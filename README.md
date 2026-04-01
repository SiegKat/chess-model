# chess-engine

[TRY IT ONLINE](https://setday.github.io/chess-engine-online/)

A big thanks to [Alexander Serkov](https://github.com/setday) for the web implementation.

`models/TORCH_100EPOCHS` has shown a performance of approx. 1500 ELO during opening and middlegame.

It fails after about 20 moves. A simple algorithm to detect blunders is still needed.

## Setup:

- Install Python dependencies:

    ```pip install -r requirements.txt```

- Put your data (`.pgn` files) into `data/png/`.

> The [dataset](https://database.nikonoel.fr/) that I used.

- Refer to ```engines/chess_engine.ipynb``` for further instructions and actions (TensorFlow)
- Or use the maintained PyTorch workflow in `engines/torch2`.

## Torch2 portfolio workflow

- The actively maintained PyTorch workflow now lives in `engines/torch2`.

- Training notebook:

    `engines/torch2/train_new.ipynb`

- Prediction notebook:

    `engines/torch2/predict.ipynb`

- The training helpers save artifacts with a custom base name plus the training date:

    `models/<model_name>_YYYY-MM-DD.pth`

    `models/<model_name>_YYYY-MM-DD_move_map.pkl`

- The default portfolio prefix is:

    `chess_model_v2_portfolio`

- The current project keeps PGN files in `data/png/` for historical reasons, so the `torch2` notebook uses that folder as its input path.

- Additional torch2 setup, training, and inference notes are documented in:

    `engines/torch2/README.md`
