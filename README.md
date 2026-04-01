# Chess Engine

A chess engine powered by a **deep convolutional neural network** trained with **supervised learning** on expert-level human games. The architecture uses **residual blocks** and a **dual-head design** (policy + value), inspired by AlphaZero, but trained via **imitation learning** from the [Lichess Elite Database](https://database.nikonoel.fr/) rather than reinforcement learning self-play.

## Approach

### Supervised Deep Learning (Imitation Learning)

The model learns to imitate expert human play by observing what strong players did in each board position. The training data comes from thousands of high-rated games in PGN format. Each position in each game produces one training sample: the board state is the input, and the expert's move plus the game result are the targets.

This is **not** reinforcement learning — there is no self-play, no reward signal, and no environment interaction. The labels come directly from human expert games.

### Board Representation

Each board position is encoded as a **19-plane 8×8 tensor** (similar to a multi-channel image):

| Planes | Description |
|--------|-------------|
| 0–5 | White piece positions (Pawn, Knight, Bishop, Rook, Queen, King) |
| 6–11 | Black piece positions |
| 12 | Side to move |
| 13–16 | Castling rights (white/black, kingside/queenside) |
| 17 | Move count (normalized) |
| 18 | 50-move rule counter (normalized) |

### Model Architecture

The actively maintained model (`ChessModelV2`) is a residual convolutional neural network with a dual-head output:

- **Stem**: 19 → 128 channels via 3×3 convolution + batch normalization
- **Body**: 8 residual blocks (each with two 3×3 convolutions, batch normalization, and skip connections)
- **Policy head**: predicts a probability distribution over all possible moves (classification, trained with cross-entropy loss)
- **Value head**: predicts the expected game outcome from −1 to +1 (regression, trained with MSE loss and Tanh activation)

Combined loss: `L = policy_loss + 0.5 × value_loss`

### Training Details

- **Data**: up to 50,000 games from the Lichess Elite Database, each producing ~40–80 training samples
- **Optimizer**: Adam (learning rate 1e-4)
- **Batch size**: 128
- **Epochs**: 20
- **Device**: CUDA GPU when available

### Inference

At prediction time, the model takes a board position, produces move probabilities via the policy head, filters them to only legal moves, and picks the highest-probability candidate. The value head provides a position evaluation score.

### Comparison with AlphaZero

| Aspect | AlphaZero | This Project |
|--------|-----------|--------------|
| Training paradigm | Self-play reinforcement learning | Supervised learning from human games |
| Data source | Games the AI plays against itself | Expert human games (Lichess Elite) |
| Architecture | Residual CNN, dual-head | Residual CNN, dual-head (similar) |
| Search | Monte Carlo Tree Search (MCTS) | Direct policy prediction (no search) |

## Project Structure

```
engines/
  torch2/          # Actively maintained PyTorch workflow (V2)
    model_v2.py        # ChessModelV2 — residual dual-head network
    dataset_v2.py      # PyTorch Dataset wrapper
    auxiliary_func_v2.py  # Board encoding, move encoding, artifact I/O
    train_new.ipynb    # Training notebook
    predict.ipynb      # Inference notebook
  torch/           # Original PyTorch workflow (V1)
    model.py           # ChessModel — simple 2-layer CNN
    dataset.py         # PyTorch Dataset wrapper
    auxiliary_func.py  # Board encoding and helpers
    train.ipynb        # Training notebook
data/
  png/             # PGN game files go here (folder name is historical)
models/            # Saved model checkpoints and move-map files
```

## Setup

1. Install Python dependencies:

    ```
    pip install -r requirements.txt
    ```

2. Download PGN files from the [Lichess Elite Database](https://database.nikonoel.fr/) and place them in `data/png/`.

3. Open the training notebook:

    `engines/torch2/train_new.ipynb`

4. Adjust the `CONFIG` cell if needed (batch size, number of games, epochs, etc.) and run the notebook end to end.

Trained artifacts are saved to `models/` with a dated naming convention:

- `models/<model_name>_YYYY-MM-DD.pth`
- `models/<model_name>_YYYY-MM-DD_move_map.pkl`

## Prediction

1. Make sure a trained model exists in `models/`.
2. Open the prediction notebook:

    `engines/torch2/predict.ipynb`

3. Run the notebook — it automatically loads the most recent checkpoint for the configured model prefix.

## Performance

The V1 model (`TORCH_100EPOCHS`) showed a performance of approximately 1500 ELO during opening and middlegame phases, though it struggles after about 20 moves. The V2 dual-head architecture with residual blocks aims to improve on this through a richer board representation and position evaluation.
