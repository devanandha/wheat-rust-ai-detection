import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

from .data import IMAGE_SIZE


def healthy_dataset(path: Path, batch_size: int, shuffle: bool):
    files = tf.data.Dataset.list_files(str(path / "*"), shuffle=shuffle, seed=42)

    def load_image(filename):
        image = tf.io.decode_image(tf.io.read_file(filename), channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32) / 255.0

    return files.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def reconstruction_dataset(dataset):
    """Return (input, target) pairs for autoencoder training."""
    return dataset.map(lambda images: (images, images), num_parallel_calls=tf.data.AUTOTUNE)


def build_autoencoder():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    outputs = tf.keras.layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="healthy_leaf_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_errors(model, dataset):
    errors = []
    for batch in dataset:
        reconstruction = model(batch, training=False)
        errors.extend(tf.reduce_mean(tf.square(batch - reconstruction), axis=(1, 2, 3)).numpy())
    return np.asarray(errors)


def load_labelled_validation(data_dir: Path, batch_size: int):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        label_mode="int",
        class_names=["Brown_rust", "Healthy", "Yellow_rust"],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))


def train(data_dir: Path, model_path: Path, output_dir: Path, epochs: int, batch_size: int):
    tf.keras.utils.set_random_seed(42)
    training = healthy_dataset(data_dir / "train" / "Healthy", batch_size, True)
    healthy_validation = healthy_dataset(data_dir / "val" / "Healthy", batch_size, False)
    model = build_autoencoder()
    model.fit(reconstruction_dataset(training), validation_data=reconstruction_dataset(healthy_validation), epochs=epochs,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

    threshold = float(np.percentile(reconstruction_errors(model, healthy_validation), 95))
    labelled = load_labelled_validation(data_dir, batch_size)
    scores, labels = [], []
    for images, class_ids in labelled:
        reconstruction = model(images, training=False)
        scores.extend(tf.reduce_mean(tf.square(images - reconstruction), axis=(1, 2, 3)).numpy())
        labels.extend((class_ids.numpy() != 1).astype(int))
    scores, labels = np.asarray(scores), np.asarray(labels)
    predictions = (scores > threshold).astype(int)
    report = classification_report(labels, predictions, target_names=["Healthy", "Disease anomaly"],
                                   output_dict=True, zero_division=0)
    report["roc_auc"] = float(roc_auc_score(labels, scores))
    report["threshold"] = threshold

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    (output_dir / "anomaly_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train healthy-only anomaly detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("models/healthy_autoencoder.keras"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/anomaly"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.data_dir, args.model, args.output_dir, args.epochs, args.batch_size)
