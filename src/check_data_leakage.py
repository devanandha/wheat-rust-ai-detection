from pathlib import Path
from collections import defaultdict
import hashlib

from PIL import Image
import imagehash


DATA_DIR = Path(
    r"F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data"
)

SPLITS = ["train", "val"]
MAX_HAMMING_DISTANCE = 5


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def perceptual_hash(path: Path):
    with Image.open(path) as image:
        return imagehash.phash(image.convert("RGB"))


def main():
    exact_hashes = defaultdict(list)

    train_images = []
    val_images = []

    total_files = 0

    for split in SPLITS:
        split_dir = DATA_DIR / split

        for image_path in split_dir.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }:
                continue

            total_files += 1

            sha256 = file_hash(image_path)
            phash = perceptual_hash(image_path)

            record = {
                "split": split,
                "path": image_path,
                "sha256": sha256,
                "phash": phash,
            }

            exact_hashes[sha256].append(record)

            if split == "train":
                train_images.append(record)
            else:
                val_images.append(record)

    exact_duplicates = []

    for sha256, entries in exact_hashes.items():
        splits_present = {
            entry["split"]
            for entry in entries
        }

        if "train" in splits_present and "val" in splits_present:
            exact_duplicates.append(entries)

    print(f"Total images checked: {total_files}")
    print(
        "Exact duplicate groups across train/validation: "
        f"{len(exact_duplicates)}"
    )

    print("\nChecking perceptual near-duplicates...")

    near_duplicates = []

    for val_record in val_images:
        for train_record in train_images:
            distance = (
                val_record["phash"]
                - train_record["phash"]
            )

            if distance <= MAX_HAMMING_DISTANCE:
                near_duplicates.append(
                    {
                        "distance": distance,
                        "train_path": train_record["path"],
                        "val_path": val_record["path"],
                    }
                )

    near_duplicates.sort(
        key=lambda item: item["distance"]
    )

    print(
        "Near-duplicate train/validation pairs "
        f"(pHash distance <= {MAX_HAMMING_DISTANCE}): "
        f"{len(near_duplicates)}"
    )
    output_path = Path("outputs/failure_analysis/near_duplicate_pairs.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        import csv

        writer = csv.writer(csvfile)
        writer.writerow([
            "distance",
            "train_path",
            "val_path",
        ])

        for pair in near_duplicates:
            writer.writerow([
                pair["distance"],
                pair["train_path"],
                pair["val_path"],
            ])

    print(f"Saved near-duplicate report to: {output_path}")

    if near_duplicates:
        print("\nPossible near-duplicates:\n")

        for index, pair in enumerate(
            near_duplicates,
            start=1,
        ):
            print(
                f"Pair {index} | "
                f"distance={pair['distance']}"
            )
            print(
                f"  train: {pair['train_path']}"
            )
            print(
                f"  val:   {pair['val_path']}"
            )
            print()

    else:
        print(
            "\nNo perceptual near-duplicates were found "
            "at the selected threshold."
        )


if __name__ == "__main__":
    main()