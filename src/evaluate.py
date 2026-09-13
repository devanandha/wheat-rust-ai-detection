import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .data import EXPECTED_CLASSES, load_split


def evaluate(
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 32,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    validation, image_paths = load_split(
        data_dir / "val",
        shuffle=False,
        batch_size=batch_size,
        return_paths=True,
    )

    model = tf.keras.models.load_model(model_path)

    probabilities = model.predict(validation, verbose=1)
    predictions = probabilities.argmax(axis=1)

    labels = np.concatenate([
        batch_labels.numpy().argmax(axis=1)
        for _, batch_labels in validation
    ])

    confidence = probabilities.max(axis=1)

    report = classification_report(
        labels,
        predictions,
        target_names=EXPECTED_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    matrix = confusion_matrix(labels, predictions)

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=EXPECTED_CLASSES,
        yticklabels=EXPECTED_CLASSES,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=180,
    )

    plt.close()

    misclassified_path = (
        output_dir / "misclassified_images.csv"
    )

    with misclassified_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            "image_path",
            "actual_class",
            "predicted_class",
            "confidence",
        ])

        for image_path, actual, predicted, conf in zip(
            image_paths,
            labels,
            predictions,
            confidence,
        ):
            if actual != predicted:
                writer.writerow([
                    image_path,
                    EXPECTED_CLASSES[actual],
                    EXPECTED_CLASSES[predicted],
                    float(conf),
                ])

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained wheat-rust classifier."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "models/wheat_rust_mobilenetv2.keras"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/classifier"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        args.data_dir,
        args.model,
        args.output_dir,
    )