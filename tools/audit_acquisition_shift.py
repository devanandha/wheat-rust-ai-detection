"""Descriptive acquisition-shift audit of separately sourced class-folder images.

No training, predictions, resizing of original files, or model selection is done.
The border metric is an image-edge proxy, NOT a verified background mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FEATURES = (
    "mean_brightness", "sd_brightness", "border_mean_brightness",
    "center_mean_brightness", "border_mean_saturation", "center_mean_saturation",
    "border_center_brightness_difference", "aspect_ratio",
)
FIELDS = ("source", "label", "image_id", "sha256", "width", "height", *FEATURES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(path: Path) -> dict[str, float | int]:
    """Calculate deterministic colour/layout proxies at resolution 128 x 128."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if width == 0 or height == 0:
            raise ValueError("Invalid image dimensions")
        reduced = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        rgb = np.asarray(reduced, dtype=np.float32) / 255.0
        hsv = np.asarray(reduced.convert("HSV"), dtype=np.float32) / 255.0
    # Standard sRGB-coded luma proxy, not physical scene luminance.
    brightness = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    sat = hsv[:, :, 1]
    mask = np.zeros((128, 128), dtype=bool)
    mask[:16, :] = True
    mask[-16:, :] = True
    mask[:, :16] = True
    mask[:, -16:] = True
    center = ~mask
    border_avg = float(brightness[mask].mean())
    center_avg = float(brightness[center].mean())
    return {
        "width": width, "height": height, "aspect_ratio": round(width / height, 6),
        "mean_brightness": round(float(brightness.mean()), 6),
        "sd_brightness": round(float(brightness.std(ddof=0)), 6),
        "border_mean_brightness": round(border_avg, 6),
        "center_mean_brightness": round(center_avg, 6),
        "border_mean_saturation": round(float(sat[mask].mean()), 6),
        "center_mean_saturation": round(float(sat[center].mean()), 6),
        "border_center_brightness_difference": round(border_avg - center_avg, 6),
    }


def parse_sources(items: list[str]) -> dict[str, Path]:
    if len(items) < 2:
        raise ValueError("Provide at least two --source NAME=FOLDER entries")
    sources: dict[str, Path] = {}
    for item in items:
        name, sep, directory = item.partition("=")
        if not sep or not name.strip() or not directory.strip() or name in sources:
            raise ValueError(f"Invalid or duplicate --source entry: {item!r}")
        path = Path(directory).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Source folder not found: {path}")
        sources[name] = path
    return sources


def write_csv(path: Path, records: list[dict], fieldnames: tuple[str, ...] | list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def audit(sources: dict[str, Path], output: Path, max_per_class: int | None = None) -> dict:
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be positive")
    rows: list[dict] = []
    errors: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for source, root in sources.items():
        counts[source] = {}
        for label in CLASSES:
            folder = root / label
            files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS) if folder.is_dir() else []
            if max_per_class is not None:
                files = files[:max_per_class]
            counts[source][label] = len(files)
            for path in files:
                identifier = f"{label}/{path.name}"
                try:
                    values = measure(path)
                    rows.append({"source": source, "label": label, "image_id": identifier,
                                 "sha256": sha256_file(path), **values})
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    errors.append({"source": source, "label": label, "image_id": identifier,
                                   "error": type(exc).__name__ + ": " + str(exc)})
    if not rows:
        raise ValueError("No readable images found in supplied source/class folders")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_image_metrics.csv", rows, FIELDS)
    summary: list[dict] = []
    for source in sources:
        for label in CLASSES:
            group = [r for r in rows if r["source"] == source and r["label"] == label]
            record: dict = {"source": source, "label": label, "count": len(group)}
            for key in FEATURES:
                values = np.array([r[key] for r in group], dtype=np.float64)
                record[f"{key}_mean"] = round(float(values.mean()), 6) if len(values) else ""
                record[f"{key}_sd"] = round(float(values.std(ddof=1)), 6) if len(values) > 1 else ""
            summary.append(record)
    write_csv(output / "source_class_summary.csv", summary,
              ["source", "label", "count", *(field for key in FEATURES for field in (f"{key}_mean", f"{key}_sd"))])
    hashes: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        hashes[r["sha256"]].add(r["source"])
    duplicate_hashes = sorted(digest for digest, owners in hashes.items() if len(owners) > 1)
    shared = [label for label in CLASSES if all(any(r["source"] == s and r["label"] == label for r in rows) for s in sources)]
    report = {
        "status": "descriptive_only_no_models_rerun",
        "sources": list(sources), "image_counts_before_corrupt_filter": counts,
        "processed_images": len(rows), "unreadable_images": errors,
        "exact_sha256_duplicate_groups_across_sources": len(duplicate_hashes),
        "exact_sha256_duplicate_hashes_across_sources": duplicate_hashes,
        "classes_represented_in_every_source": shared,
        "sampling": "all eligible images" if max_per_class is None else f"first {max_per_class} sorted files per source and class",
        "warnings": [
            "Image-edge measurements are NOT verified backgrounds or plant masks.",
            "This is a descriptive study, not an explanation of model errors or a causal experiment.",
            "Comparisons can be confounded by disease-label and acquisition-source differences.",
            "Pixel/metadata measurements are not leakage or field-independence checks; hashes detect exact-file overlap only.",
            "Previously inspected external benchmarks must not be presented as untouched evaluation sets.",
            "Do not tune models using the results from a protected final test set.",
        ],
    }
    (output / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True,
                        help="Repeat NAME=FOLDER, where FOLDER has Brown_rust/ Healthy/ Yellow_rust/ subfolders")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Optional smoke-test cap; reported results must declare this sampling")
    args = parser.parse_args()
    try:
        result = audit(parse_sources(args.source), args.output_dir, args.max_per_class)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"processed_images": result["processed_images"],
                      "shared_classes": result["classes_represented_in_every_source"],
                      "unreadable_images": len(result["unreadable_images"]),
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
