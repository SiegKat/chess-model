# Torch2 Chess Pipeline

This folder contains the maintained PyTorch chess workflow that is ready to present in a portfolio.

## Files kept in this folder

- `train_new.ipynb`: supervised training notebook for the V2 model.
- `predict.ipynb`: inference notebook for loading the latest saved V2 checkpoint and testing moves on a board.
- `auxiliary_func_v2.py`: board encoding, target creation, move encoding, and artifact management helpers.
- `dataset_v2.py`: dataset wrapper used by the training notebook.
- `model_v2.py`: residual dual-head chess model used for policy and value prediction.

## Training flow

1. Put PGN files in `../../data/png/`.
2. Open `train_new.ipynb`.
3. Adjust the `CONFIG` cell if you want different batch size, number of games, epochs, or artifact prefix.
4. Run the notebook end to end.

The notebook saves two files in `../../models/`:

- `<model_name>_YYYY-MM-DD.pth`
- `<model_name>_YYYY-MM-DD_move_map.pkl`

Default `model_name`:

- `chess_model_v2_portfolio`

## Prediction flow

1. Train a model first or make sure matching artifacts already exist in `../../models/`.
2. Open `predict.ipynb`.
3. Keep the same `model_name` prefix used during training.
4. Run the notebook to automatically load the most recent dated checkpoint for that prefix.

The prediction notebook uses the policy head from `ChessModelV2` and filters predictions to legal chess moves before choosing the best move.

## Notes

- The move-map pickle stores `move_to_int`, `int_to_move`, `trained_on`, and `model_name`.
- Artifact discovery is handled by `find_latest_training_artifacts(...)` in `auxiliary_func_v2.py`.
- The folder name `data/png` is historical; the notebook still expects PGN files inside it.
