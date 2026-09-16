"""Generate the remaining machine-readable evidence for the manuscript.

The script stores relative filenames and hashes, never absolute local paths or
dataset image bytes. It does not retrain or modify a model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEPENDENCIES = ("tensorflow", "numpy", "pandas", "Pillow", "scikit-learn", "ImageHash")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_root(path: Path, *, external: bool) -> Path:
    candidates = [path / "val", path] if external else [path / "train", path]
    for candidate in candidates:
        if all((candidate / name).is_dir() for name in CLASSES):
            return candidate
    raise FileNotFoundError(
        f"Could not find the expected class folders below {path}. "
        f"Required: {', '.join(CLASSES)}"
    )


def image_paths(root: Path):
    for class_name in CLASSES:
        for path in sorted((root / class_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield class_name, path


def inspect_images(root: Path) -> tuple[list[dict], list[dict]]:
    try:
        import imagehash
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError(
            "Missing image dependencies. Run: pip install -r requirements.txt"
        ) from error

    records: list[dict] = []
    corrupt: list[dict] = []
    for class_name, path in image_paths(root):
        relative_path = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                perceptual_hash = str(imagehash.phash(rgb))
            records.append(
                {
                    "class": class_name,
                    "relative_path": relative_path,
                    "sha256": sha256(path),
                    "phash": perceptual_hash,
                    "width": width,
                    "height": height,
                    "size_bytes": path.stat().st_size,
                }
            )
        except (UnidentifiedImageError, OSError) as error:
            corrupt.append({"relative_path": relative_path, "error": type(error).__name__})
    return records, corrupt


def inspect_original(data_dir: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    corrupt: list[dict] = []
    for split in ("train", "val"):
        root = data_dir / split
        if not all((root / name).is_dir() for name in CLASSES):
            raise FileNotFoundError(f"Missing expected class folders below {root}")
        split_records, split_corrupt = inspect_images(root)
        for record in split_records:
            record["split"] = split
        for record in split_corrupt:
            record["split"] = split
        records.extend(split_records)
        corrupt.extend(split_corrupt)
    return records, corrupt


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def package_versions() -> dict[str, str | None]:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_hashes(root: Path) -> dict[str, str]:
    models = root / "models"
    if not models.is_dir():
        return {}
    return {
        path.name: sha256(path)
        for path in sorted(models.iterdir())
        if path.is_file() and path.suffix.lower() in {".keras", ".h5"}
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-data", type=Path, required=True)
    parser.add_argument("--external-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evidence"))
    parser.add_argument("--phash-threshold", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    external_root = class_root(args.external_data, external=True)

    print("Inspecting external dataset...")
    external, external_corrupt = inspect_images(external_root)
    external_counts = Counter(record["class"] for record in external)
    inventory = {
        "dataset": "Wheat Disease Dataset - Small",
        "source_doi": "10.5281/zenodo.7307816",
        "included_classes": list(CLASSES),
        "excluded_classes": ["Mildew", "Septoria"],
        "counts": dict(external_counts),
        "total_images": len(external),
        "corrupt_files": external_corrupt,
        "files": external,
    }
    write_json(args.output_dir / "external_dataset_inventory.json", inventory)

    print("Inspecting original dataset and checking cross-dataset overlap...")
    original, original_corrupt = inspect_original(args.original_data)
    original_by_sha: dict[str, list[dict]] = defaultdict(list)
    for record in original:
        original_by_sha[record["sha256"]].append(record)

    exact_pairs = []
    near_pairs = []
    for external_record in external:
        for original_record in original_by_sha.get(external_record["sha256"], []):
            exact_pairs.append(
                {
                    "original_split": original_record["split"],
                    "original_relative_path": original_record["relative_path"],
                    "external_relative_path": external_record["relative_path"],
                    "sha256": external_record["sha256"],
                }
            )
        for original_record in original:
            distance = hamming(original_record["phash"], external_record["phash"])
            if distance <= args.phash_threshold:
                near_pairs.append(
                    {
                        "distance": distance,
                        "original_split": original_record["split"],
                        "original_relative_path": original_record["relative_path"],
                        "external_relative_path": external_record["relative_path"],
                    }
                )

    near_pairs.sort(key=lambda item: (item["distance"], item["external_relative_path"]))
    audit = {
        "original_images_checked": len(original),
        "external_images_checked": len(external),
        "original_corrupt_files": original_corrupt,
        "external_corrupt_files": external_corrupt,
        "exact_cross_dataset_duplicate_count": len(exact_pairs),
        "exact_cross_dataset_pairs": exact_pairs,
        "phash_algorithm": "ImageHash pHash",
        "phash_hamming_threshold": args.phash_threshold,
        "near_duplicate_count": len(near_pairs),
        "near_duplicate_pairs": near_pairs,
    }
    write_json(args.output_dir / "cross_dataset_duplicate_audit.json", audit)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "manuscript evidence generation and reproducibility snapshot",
        "git_commit": git_commit(project_root),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "model_sha256": model_hashes(project_root),
        "parameters": {
            "classes": list(CLASSES),
            "phash_threshold": args.phash_threshold,
            "original_image_count": len(original),
            "external_image_count": len(external),
        },
        "scope_note": (
            "This records the evidence-generation environment. Historical training "
            "hardware and package versions were not retrospectively inferred."
        ),
    }
    write_json(args.output_dir / "run_metadata.json", metadata)

    print(f"External images: {len(external)}")
    print(f"Exact cross-dataset duplicates: {len(exact_pairs)}")
    print(f"Near-duplicates (pHash <= {args.phash_threshold}): {len(near_pairs)}")
    print(f"Saved evidence to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
