import argparse
from pathlib import Path

from .evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the classifier on an external validation dataset."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path containing a val/ folder with Brown_rust, Healthy, and Yellow_rust.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/wheat_rust_mobilenetv2.keras"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/external_validation"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate(
        args.data_dir,
        args.model,
        args.output_dir,
        batch_size=8,
    )