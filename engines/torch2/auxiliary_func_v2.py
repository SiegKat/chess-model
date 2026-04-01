# auxiliary_func_v2.py

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chess
import chess.pgn  # FIX: Explicitly import the pgn submodule
import numpy as np
import torch

DEFAULT_MODEL_NAME = "chess_model_v2_portfolio"
DEFAULT_MODELS_DIR = "../../models"

PIECE_TO_INDEX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}
NUM_PIECE_TYPES = 6

def board_to_matrix_v2(board: chess.Board) -> np.ndarray:
    """
    Converts a board object into a richer, multi-plane representation.
    
    Representation (19 planes, 8x8):
    - Planes 0-5: White pieces (P, N, B, R, Q, K)
    - Planes 6-11: Black pieces (P, N, B, R, Q, K)
    - Plane 12: Player to move (1s for white, 0s for black)
    - Plane 13: White kingside castling right
    - Plane 14: White queenside castling right
    - Plane 15: Black kingside castling right
    - Plane 16: Black queenside castling right
    - Plane 17: Total move count (normalized)
    - Plane 18: No-progress counter (for 50-move rule, normalized)
    """
    # Total planes: 6 pieces * 2 colors + 7 feature planes = 19
    matrix = np.zeros((19, 8, 8), dtype=np.float32)
    
    # --- Piece Planes (0-11) ---
    for sq, piece in board.piece_map().items():
        row, col = divmod(sq, 8)
        piece_idx = PIECE_TO_INDEX[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else NUM_PIECE_TYPES
        matrix[piece_idx + color_offset, row, col] = 1

    # --- Feature Planes (12-18) ---
    if board.turn == chess.WHITE:
        matrix[12, :, :] = 1.0
    
    if board.has_kingside_castling_rights(chess.WHITE): matrix[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): matrix[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): matrix[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): matrix[16, :, :] = 1.0
    
    matrix[17, :, :] = board.fullmove_number / 150.0 # Normalize move count
    matrix[18, :, :] = board.halfmove_clock / 50.0 # Normalize half-move clock

    return matrix

def get_value_for_result(result: str, turn: chess.Color) -> float:
    """Converts a game result to a numerical value [-1, 1] from the current player's perspective."""
    if result == '1-0':
        return 1.0 if turn == chess.WHITE else -1.0
    elif result == '0-1':
        return -1.0 if turn == chess.WHITE else 1.0
    return 0.0

def create_input_for_nn_v2(games: List[chess.pgn.Game]) -> Tuple[List[np.ndarray], List[str], List[float]]:
    """
    Creates the input data (X) and two corresponding targets (policy y, value z).
    """
    X = [] # Board states
    y_policy = [] # Moves (policy targets)
    z_value = [] # Game outcomes (value targets)
    
    for game in games:
        board = game.board()
        result = game.headers.get('Result', '*')
        if result not in ['1-0', '0-1', '1/2-1/2']:
            continue

        for move in game.mainline_moves():
            # Store the sample for the current board state
            X.append(board_to_matrix_v2(board))
            y_policy.append(move.uci())
            z_value.append(get_value_for_result(result, board.turn))
            
            # Make the move on the board
            board.push(move)
            
    return X, y_policy, z_value

def encode_moves(moves: List[str]) -> Tuple[np.ndarray, dict, dict]:
    """Creates a mapping from all unique moves to integers and back."""
    unique_moves = sorted(list(set(moves)))
    move_to_int = {move: idx for idx, move in enumerate(unique_moves)}
    int_to_move = {idx: move for move, idx in move_to_int.items()}
    encoded_y = np.array([move_to_int[move] for move in moves], dtype=np.int64)
    return encoded_y, move_to_int, int_to_move


def _normalize_model_name(model_name: str) -> str:
    cleaned_name = "".join(char if char.isalnum() else "_" for char in model_name.strip())
    cleaned_name = "_".join(part for part in cleaned_name.split("_") if part)
    return cleaned_name or DEFAULT_MODEL_NAME


def save_training_artifacts(
    model,
    move_to_int: dict,
    model_name: str = DEFAULT_MODEL_NAME,
    models_dir: str = DEFAULT_MODELS_DIR,
    trained_at=None,
) -> dict:
    """Save a model checkpoint plus move map metadata using the shared dated naming convention."""
    trained_at = trained_at or datetime.now()
    trained_on = trained_at.strftime("%Y-%m-%d")
    normalized_name = _normalize_model_name(model_name)
    artifact_prefix = f"{normalized_name}_{trained_on}"

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    model_path = models_path / f"{artifact_prefix}.pth"
    map_path = models_path / f"{artifact_prefix}_move_map.pkl"

    torch.save(model.state_dict(), model_path)

    int_to_move = {idx: move for move, idx in move_to_int.items()}
    with open(map_path, "wb") as file:
        pickle.dump(
            {
                "move_to_int": move_to_int,
                "int_to_move": int_to_move,
                "trained_on": trained_on,
                "model_name": normalized_name,
            },
            file,
        )

    return {
        "model_path": str(model_path),
        "map_path": str(map_path),
        "trained_on": trained_on,
        "artifact_prefix": artifact_prefix,
        "model_name": normalized_name,
    }


def load_move_map_data(map_path: str) -> Dict[str, Any]:
    """
    Load a move map saved by `save_training_artifacts`.

    Older map files that only contain `move_to_int` are normalized into the new
    dictionary format so callers can rely on a stable shape.
    """
    with open(map_path, "rb") as file:
        raw_data = pickle.load(file)

    if isinstance(raw_data, dict) and "move_to_int" in raw_data:
        move_to_int = raw_data["move_to_int"]
        int_to_move = raw_data.get("int_to_move", {idx: move for move, idx in move_to_int.items()})
        return {
            "move_to_int": move_to_int,
            "int_to_move": int_to_move,
            "trained_on": raw_data.get("trained_on"),
            "model_name": raw_data.get("model_name"),
        }

    move_to_int = raw_data
    return {
        "move_to_int": move_to_int,
        "int_to_move": {idx: move for move, idx in move_to_int.items()},
        "trained_on": None,
        "model_name": None,
    }


def find_latest_training_artifacts(
    model_name: str = DEFAULT_MODEL_NAME,
    models_dir: str = DEFAULT_MODELS_DIR,
) -> Dict[str, str]:
    """
    Find the most recent dated model + move-map pair for a given artifact prefix.
    """
    normalized_name = _normalize_model_name(model_name)
    models_path = Path(models_dir)

    map_candidates = sorted(models_path.glob(f"{normalized_name}_????-??-??_move_map.pkl"))
    if not map_candidates:
        raise FileNotFoundError(
            f"No saved artifacts were found for '{normalized_name}' in '{models_path}'. "
            "Run the training notebook first or change the configured model_name."
        )

    latest_map_path = map_candidates[-1]
    artifact_prefix = latest_map_path.name[: -len("_move_map.pkl")]
    model_path = models_path / f"{artifact_prefix}.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Found move map '{latest_map_path.name}' but the matching model file "
            f"'{model_path.name}' is missing."
        )

    return {
        "artifact_prefix": artifact_prefix,
        "model_path": str(model_path),
        "map_path": str(latest_map_path),
        "model_name": normalized_name,
    }
