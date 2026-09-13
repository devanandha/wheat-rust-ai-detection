import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main():
    csv_path = Path("outputs/failure_analysis/failure_analysis.csv")
    output_path = Path("outputs/failure_analysis/failure_cases_grid.png")

    rows = []

    with csv_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            rows.append(row)

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=2,
        figsize=(10, len(rows) * 3.2),
    )

    for index, row in enumerate(rows):
        original_path = Path(
            r"F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data"
        ) / "val" / row["actual_class"] / row["image_name"]

        gradcam_path = Path(row["gradcam_output"])

        original_image = Image.open(original_path).convert("RGB")
        gradcam_image = Image.open(gradcam_path).convert("RGB")

        axes[index, 0].imshow(original_image)
        axes[index, 0].axis("off")
        axes[index, 0].set_title(
            f"{row['image_name']}\n"
            f"Actual: {row['actual_class']}"
        )

        axes[index, 1].imshow(gradcam_image)
        axes[index, 1].axis("off")
        axes[index, 1].set_title(
            f"Predicted: {row['predicted_class']}\n"
            f"Confidence: {float(row['confidence']):.2%}"
        )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved failure-case figure to: {output_path}")


if __name__ == "__main__":
    main()