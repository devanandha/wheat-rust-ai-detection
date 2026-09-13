import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import EXPECTED_CLASSES, IMAGE_SIZE
from .gradcam import make_gradcam_heatmap


def load_image(image_path: Path) -> np.ndarray:
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        interpolation="bilinear",
    )

    array = tf.keras.utils.img_to_array(image)

    return np.asarray(
        array,
        dtype=np.float32,
    )


def resize_heatmap(
    heatmap: np.ndarray,
) -> np.ndarray:
    heatmap_tensor = tf.convert_to_tensor(
        heatmap[..., np.newaxis],
        dtype=tf.float32,
    )

    resized = tf.image.resize(
        heatmap_tensor,
        IMAGE_SIZE,
        method="bilinear",
    )

    return resized.numpy().squeeze()


def make_top_mask(
    heatmap: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """
    Select the highest-activation Grad-CAM pixels.
    """

    flattened = heatmap.reshape(-1)

    pixel_count = max(
        1,
        int(
            round(
                flattened.size * fraction
            )
        ),
    )

    selected_indices = np.argpartition(
        flattened,
        -pixel_count,
    )[-pixel_count:]

    mask = np.zeros(
        flattened.size,
        dtype=bool,
    )

    mask[selected_indices] = True

    return mask.reshape(
        heatmap.shape
    )


def make_random_block_mask(
    image_shape: tuple[int, int],
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create a spatially coherent random rectangular mask
    covering approximately the requested fraction
    of the image.
    """

    height, width = image_shape

    target_pixels = max(
        1,
        int(
            round(
                height
                * width
                * fraction
            )
        ),
    )

    # Randomise the shape of the rectangle while
    # keeping approximately the same masked area.
    aspect_ratio = float(
        rng.uniform(
            0.5,
            2.0,
        )
    )

    block_height = int(
        round(
            np.sqrt(
                target_pixels
                / aspect_ratio
            )
        )
    )

    block_width = int(
        round(
            target_pixels
            / max(
                block_height,
                1,
            )
        )
    )

    block_height = min(
        max(
            block_height,
            1,
        ),
        height,
    )

    block_width = min(
        max(
            block_width,
            1,
        ),
        width,
    )

    top = int(
        rng.integers(
            0,
            height
            - block_height
            + 1,
        )
    )

    left = int(
        rng.integers(
            0,
            width
            - block_width
            + 1,
        )
    )

    mask = np.zeros(
        (
            height,
            width,
        ),
        dtype=bool,
    )

    mask[
        top:top + block_height,
        left:left + block_width,
    ] = True

    return mask


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Replace selected pixels with the image's
    mean RGB value.
    """

    masked = image.copy()

    replacement = image.mean(
        axis=(0, 1),
        keepdims=False,
    )

    masked[mask] = replacement

    return masked


def predict_class_confidence(
    model: tf.keras.Model,
    image: np.ndarray,
    class_index: int,
) -> float:

    probabilities = model(
        np.expand_dims(
            image,
            axis=0,
        ),
        training=False,
    ).numpy()[0]

    return float(
        probabilities[class_index]
    )


def select_balanced_images(
    validation_dir: Path,
    per_class: int,
    seed: int,
) -> list[Path]:

    rng = np.random.default_rng(
        seed
    )

    selected: list[Path] = []

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    for class_name in EXPECTED_CLASSES:

        class_dir = (
            validation_dir
            / class_name
        )

        candidates = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in valid_extensions
        )

        if len(candidates) < per_class:
            raise ValueError(
                f"{class_name} contains only "
                f"{len(candidates)} images, "
                f"but {per_class} were requested."
            )

        indices = rng.choice(
            len(candidates),
            size=per_class,
            replace=False,
        )

        selected.extend(
            candidates[index]
            for index in indices
        )

    return selected


def evaluate_faithfulness(
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    per_class: int = 20,
    mask_fraction: float = 0.20,
    random_repeats: int = 5,
    seed: int = 42,
):

    if not 0 < mask_fraction < 1:
        raise ValueError(
            "mask_fraction must be "
            "between 0 and 1."
        )

    if random_repeats < 1:
        raise ValueError(
            "random_repeats must be "
            "at least 1."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = tf.keras.models.load_model(
        model_path
    )

    validation_dir = (
        data_dir
        / "val"
    )

    image_paths = select_balanced_images(
        validation_dir,
        per_class=per_class,
        seed=seed,
    )

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        image = load_image(
            image_path
        )

        probabilities = model(
            np.expand_dims(
                image,
                axis=0,
            ),
            training=False,
        ).numpy()[0]

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        original_confidence = float(
            probabilities[
                predicted_index
            ]
        )

        heatmap = make_gradcam_heatmap(
            model,
            np.expand_dims(
                image,
                axis=0,
            ),
            class_index=predicted_index,
        )

        resized_heatmap = resize_heatmap(
            heatmap
        )

        gradcam_mask = make_top_mask(
            resized_heatmap,
            fraction=mask_fraction,
        )

        gradcam_masked_image = apply_mask(
            image,
            gradcam_mask,
        )

        gradcam_confidence = (
            predict_class_confidence(
                model,
                gradcam_masked_image,
                predicted_index,
            )
        )

        gradcam_drop = (
            original_confidence
            - gradcam_confidence
        )

        random_drops = []

        for _ in range(
            random_repeats
        ):

            random_mask = (
                make_random_block_mask(
                    IMAGE_SIZE,
                    fraction=mask_fraction,
                    rng=rng,
                )
            )

            random_masked_image = (
                apply_mask(
                    image,
                    random_mask,
                )
            )

            random_confidence = (
                predict_class_confidence(
                    model,
                    random_masked_image,
                    predicted_index,
                )
            )

            random_drops.append(
                original_confidence
                - random_confidence
            )

        mean_random_drop = float(
            np.mean(
                random_drops
            )
        )

        advantage = (
            gradcam_drop
            - mean_random_drop
        )

        rows.append(
            {
                "image_path": str(
                    image_path
                ),
                "actual_class": (
                    image_path.parent.name
                ),
                "predicted_class": (
                    EXPECTED_CLASSES[
                        predicted_index
                    ]
                ),
                "original_confidence": (
                    original_confidence
                ),
                "gradcam_masked_confidence": (
                    gradcam_confidence
                ),
                "gradcam_confidence_drop": (
                    gradcam_drop
                ),
                "random_block_mean_confidence_drop": (
                    mean_random_drop
                ),
                "gradcam_advantage": (
                    advantage
                ),
            }
        )

        print(
            f"[{number}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"Grad-CAM drop: "
            f"{gradcam_drop:.4f} | "
            f"Random-block drop: "
            f"{mean_random_drop:.4f}"
        )

    csv_path = (
        output_dir
        / "gradcam_faithfulness.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    gradcam_drops = np.array(
        [
            row[
                "gradcam_confidence_drop"
            ]
            for row in rows
        ],
        dtype=float,
    )

    random_drops = np.array(
        [
            row[
                "random_block_mean_confidence_drop"
            ]
            for row in rows
        ],
        dtype=float,
    )

    advantages = (
        gradcam_drops
        - random_drops
    )

    class_summaries = {}

    for class_name in EXPECTED_CLASSES:

        class_rows = [
            row
            for row in rows
            if row[
                "actual_class"
            ] == class_name
        ]

        class_gradcam = np.array(
            [
                row[
                    "gradcam_confidence_drop"
                ]
                for row in class_rows
            ],
            dtype=float,
        )

        class_random = np.array(
            [
                row[
                    "random_block_mean_confidence_drop"
                ]
                for row in class_rows
            ],
            dtype=float,
        )

        class_summaries[
            class_name
        ] = {
            "images": len(
                class_rows
            ),
            "mean_gradcam_confidence_drop": (
                float(
                    np.mean(
                        class_gradcam
                    )
                )
            ),
            "mean_random_block_confidence_drop": (
                float(
                    np.mean(
                        class_random
                    )
                )
            ),
            "mean_gradcam_advantage": (
                float(
                    np.mean(
                        class_gradcam
                        - class_random
                    )
                )
            ),
            "fraction_gradcam_drop_greater_than_random": (
                float(
                    np.mean(
                        class_gradcam
                        > class_random
                    )
                )
            ),
        }

    summary = {
        "images_evaluated": len(
            rows
        ),
        "images_per_class": (
            per_class
        ),
        "mask_fraction": (
            mask_fraction
        ),
        "random_repeats_per_image": (
            random_repeats
        ),
        "random_baseline": (
            "spatially coherent "
            "random rectangular regions"
        ),
        "mean_gradcam_confidence_drop": (
            float(
                np.mean(
                    gradcam_drops
                )
            )
        ),
        "median_gradcam_confidence_drop": (
            float(
                np.median(
                    gradcam_drops
                )
            )
        ),
        "mean_random_block_confidence_drop": (
            float(
                np.mean(
                    random_drops
                )
            )
        ),
        "median_random_block_confidence_drop": (
            float(
                np.median(
                    random_drops
                )
            )
        ),
        "mean_gradcam_advantage": (
            float(
                np.mean(
                    advantages
                )
            )
        ),
        "fraction_gradcam_drop_greater_than_random": (
            float(
                np.mean(
                    gradcam_drops
                    > random_drops
                )
            )
        ),
        "per_class": (
            class_summaries
        ),
    }

    summary_path = (
        output_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "Grad-CAM faithfulness "
        "evaluation complete."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Quantitatively evaluate "
            "Grad-CAM faithfulness using "
            "region masking."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/"
            "gradcam_faithfulness_blocks"
        ),
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--random-repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    evaluate_faithfulness(
        data_dir=args.data_dir,
        model_path=args.model,
        output_dir=args.output_dir,
        per_class=args.per_class,
        mask_fraction=args.mask_fraction,
        random_repeats=args.random_repeats,
        seed=args.seed,
    )