import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .data import EXPECTED_CLASSES, load_split
from .gradcam import make_gradcam_heatmap, overlay_heatmap


def main():
    data_dir = Path(
        r"F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data"
    )
    model_path = Path("models/wheat_rust_mobilenetv2.keras")
    output_dir = Path("outputs/failure_analysis")

    output_dir.mkdir(parents=True, exist_ok=True)

    validation, file_paths = load_split(
        data_dir / "val",
        shuffle=False,
        return_paths=True,
    )

    model = tf.keras.models.load_model(model_path)

    results = []
    path_index = 0

    for batch_images, batch_labels in validation:
        probabilities = model.predict(batch_images, verbose=0)

        actual_indices = np.argmax(batch_labels.numpy(), axis=1)
        predicted_indices = np.argmax(probabilities, axis=1)

        batch_size = batch_images.shape[0]

        for i in range(batch_size):
            image_path = Path(file_paths[path_index])
            path_index += 1

            actual_index = int(actual_indices[i])
            predicted_index = int(predicted_indices[i])

            if actual_index == predicted_index:
                continue

            actual_class = EXPECTED_CLASSES[actual_index]
            predicted_class = EXPECTED_CLASSES[predicted_index]
            confidence = float(probabilities[i][predicted_index])

            # Use the exact image tensor from the validation pipeline.
            image_batch = np.expand_dims(
                batch_images[i].numpy(),
                axis=0,
            )

            heatmap = make_gradcam_heatmap(
                model,
                image_batch,
                class_index=predicted_index,
            )

            original_image = Image.open(image_path).convert("RGB")
            overlay = overlay_heatmap(original_image, heatmap)

            output_path = (
                output_dir / f"{image_path.stem}_gradcam.jpg"
            )
            overlay.save(output_path)

            results.append(
                {
                    "image_name": image_path.name,
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "brown_rust_probability": float(probabilities[i][0]),
                    "healthy_probability": float(probabilities[i][1]),
                    "yellow_rust_probability": float(probabilities[i][2]),
                    "gradcam_output": str(output_path),
                }
            )

    report_path = output_dir / "failure_analysis.csv"

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:
        fieldnames = [
            "image_name",
            "actual_class",
            "predicted_class",
            "confidence",
            "brown_rust_probability",
            "healthy_probability",
            "yellow_rust_probability",
            "gradcam_output",
        ]

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Processed {len(results)} failure cases.")
    print(f"Saved report to: {report_path}")
    print(f"Saved Grad-CAM overlays to: {output_dir}")


if __name__ == "__main__":
    main()