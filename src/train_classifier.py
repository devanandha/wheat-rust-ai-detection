import argparse
import json
from pathlib import Path

import tensorflow as tf

from .data import load_train_and_validation
from .model import build_classifier


def train(data_dir: Path, model_path: Path, output_dir: Path, epochs: int, batch_size: int):
    tf.keras.utils.set_random_seed(42)
    train_data, validation_data = load_train_and_validation(data_dir, batch_size)
    model = build_classifier()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    ]
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the wheat-rust classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/wheat_rust_mobilenetv2.keras"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/classifier"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.model, args.output_dir, args.epochs, args.batch_size)

