import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from chess import Board


def board_to_matrix(board: Board):
    # 8x8 is a size of the chess board.
    # 12 = number of unique pieces.
    # 13th board for legal moves (WHERE we can move)
    # maybe 14th for squares FROM WHICH we can move? idk
    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    # Populate first 12 8x8 boards (where pieces are)
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    # Populate the legal moves board (13th 8x8 board)
    legal_moves = board.legal_moves
    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int


def _normalize_model_name(model_name: str) -> str:
    cleaned_name = "".join(char if char.isalnum() else "_" for char in model_name.strip())
    cleaned_name = "_".join(part for part in cleaned_name.split("_") if part)
    return cleaned_name or "chess_model"


def save_training_artifacts(model, move_to_int, model_name: str, models_dir: str = "../../models", trained_at=None):
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
    }
