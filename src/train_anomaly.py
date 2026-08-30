import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

from .data import IMAGE_SIZE


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def image_dataset(files, batch_size: int, shuffle: bool):
    """Create a deterministic image dataset from an explicit list of files."""
    file_strings = [str(path) for path in files]
    if not file_strings:
        raise ValueError("The image dataset is empty.")

    dataset = tf.data.Dataset.from_tensor_slices(file_strings)
    if shuffle:
        dataset = dataset.shuffle(len(file_strings), seed=42, reshuffle_each_iteration=True)

    def load_image(filename):
        image = tf.io.decode_image(
            tf.io.read_file(filename), channels=3, expand_animations=False
        )
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32) / 255.0

    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def split_healthy_files(path: Path):
    """Split healthy training files into train, tuning and calibration sets.

    The held-out ``val`` directory is never used for model fitting, early
    stopping or threshold selection. This prevents evaluation leakage.
    """
    files = sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if len(files) < 30:
        raise ValueError("At least 30 healthy training images are required.")

    rng = np.random.default_rng(42)
    order = rng.permutation(len(files))
    files = [files[index] for index in order]

    tuning_count = max(1, int(round(len(files) * 0.10)))
    calibration_count = max(1, int(round(len(files) * 0.10)))
    training_count = len(files) - tuning_count - calibration_count
    if training_count < 1:
        raise ValueError("Not enough images remain for autoencoder training.")

    training_files = files[:training_count]
    tuning_files = files[training_count : training_count + tuning_count]
    calibration_files = files[training_count + tuning_count :]
    return training_files, tuning_files, calibration_files


def reconstruction_dataset(dataset):
    """Return (input, target) pairs for autoencoder training."""
    return dataset.map(
        lambda images: (images, images), num_parallel_calls=tf.data.AUTOTUNE
    )


def build_autoencoder():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Conv2D(
        32, 3, strides=2, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        64, 3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        128, 3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        64, 3, strides=2, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        32, 3, strides=2, padding="same", activation="relu"
    )(x)
    outputs = tf.keras.layers.Conv2DTranspose(
        3, 3, strides=2, padding="same", activation="sigmoid"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="healthy_leaf_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_errors(model, dataset):
    errors = []
    for batch in dataset:
        reconstruction = model(batch, training=False)
        batch_errors = tf.reduce_mean(
            tf.square(batch - reconstruction), axis=(1, 2, 3)
        )
        errors.extend(batch_errors.numpy())
    return np.asarray(errors)


def load_labelled_evaluation(data_dir: Path, batch_size: int):
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        label_mode="int",
        class_names=["Brown_rust", "Healthy", "Yellow_rust"],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


def score_summary(scores, mask):
    selected = scores[mask]
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "standard_deviation": float(np.std(selected)),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
    }


def train(data_dir: Path, model_path: Path, output_dir: Path, epochs: int, batch_size: int):
    tf.keras.utils.set_random_seed(42)

    training_files, tuning_files, calibration_files = split_healthy_files(
        data_dir / "train" / "Healthy"
    )
    training = image_dataset(training_files, batch_size, True)
    tuning = image_dataset(tuning_files, batch_size, False)
    calibration = image_dataset(calibration_files, batch_size, False)

    model = build_autoencoder()
    model.fit(
        reconstruction_dataset(training),
        validation_data=reconstruction_dataset(tuning),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        ],
    )

    calibration_errors = reconstruction_errors(model, calibration)
    threshold = float(np.percentile(calibration_errors, 95))

    labelled = load_labelled_evaluation(data_dir, batch_size)
    scores, disease_labels, class_ids = [], [], []
    for images, batch_class_ids in labelled:
        reconstruction = model(images, training=False)
        batch_scores = tf.reduce_mean(
            tf.square(images - reconstruction), axis=(1, 2, 3)
        ).numpy()
        ids = batch_class_ids.numpy()
        scores.extend(batch_scores)
        class_ids.extend(ids)
        disease_labels.extend((ids != 1).astype(int))

    scores = np.asarray(scores)
    disease_labels = np.asarray(disease_labels)
    class_ids = np.asarray(class_ids)

    # The intended convention is: larger reconstruction error = more anomalous.
    predictions = (scores > threshold).astype(int)
    report = classification_report(
        disease_labels,
        predictions,
        target_names=["Healthy", "Disease anomaly"],
        output_dict=True,
        zero_division=0,
    )

    raw_auc = float(roc_auc_score(disease_labels, scores))
    inverted_auc = float(roc_auc_score(disease_labels, -scores))
    report["roc_auc"] = raw_auc
    report["roc_auc_high_error_means_disease"] = raw_auc
    report["roc_auc_low_error_means_disease_diagnostic"] = inverted_auc
    report["best_direction_auc_diagnostic"] = max(raw_auc, inverted_auc)
    report["threshold"] = threshold
    report["threshold_percentile"] = 95
    report["threshold_source"] = "Separate 10% healthy calibration split from train/Healthy"
    report["split_counts"] = {
        "healthy_autoencoder_training": len(training_files),
        "healthy_early_stopping_tuning": len(tuning_files),
        "healthy_threshold_calibration": len(calibration_files),
        "held_out_evaluation": int(scores.size),
    }
    report["reconstruction_error_summary"] = {
        "Healthy": score_summary(scores, class_ids == 1),
        "Brown_rust": score_summary(scores, class_ids == 0),
        "Yellow_rust": score_summary(scores, class_ids == 2),
        "Disease_combined": score_summary(scores, disease_labels == 1),
    }
    report["interpretation"] = (
        "The primary AUC uses the intended convention that higher reconstruction "
        "error means disease. The inverted AUC is diagnostic only. If the inverted "
        "AUC is much larger, diseased images are being reconstructed with lower "
        "error than healthy images; this is model behaviour, not a label reversal "
        "inside roc_auc_score."
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    (output_dir / "anomaly_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train healthy-only anomaly detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model", type=Path, default=Path("models/healthy_autoencoder.keras")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/anomaly")
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.data_dir, args.model, args.output_dir, args.epochs, args.batch_size)
