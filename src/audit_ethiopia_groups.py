"""Read-only image-integrity and similarity audit for the Ethiopian wheat-leaf dataset.

Usage (from the Wheat Rust AI repository root):
  python -m src.audit_ethiopia_groups --root "PATH_TO_ETHIOPIA_DATASET" --out-dir "experiments\\ethiopia_audit_20261016"

The detected groups are image-similarity candidates, NOT verified identifiers for
independent plants, acquisition sessions or fields. Review provenance before
claiming group-independent evaluation. Original images are never modified.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
import imagehash

CLASSES = ("Healthy", "stripe_rust", "septoria")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root: Path, out_dir: Path, threshold: int):
    if not (0 <= threshold <= 64):
        raise ValueError("Threshold must be between 0 and 64")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    if root.resolve() == out_dir.resolve() or root.resolve() in out_dir.resolve().parents:
        raise ValueError("Report output directory must be outside the dataset folder")

    rows = []
    for cls in CLASSES:
        folder = root / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTS:
                continue
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                width, height = rgb.size
                ph = int(str(imagehash.phash(rgb)), 16)
            rows.append({
                "class": cls,
                "relative_path": path.relative_to(root).as_posix(),
                "width": width,
                "height": height,
                "sha256": sha256_file(path),
                "phash": f"{ph:016x}",
                "_phash_int": ph,
            })
    if not rows:
        raise ValueError("No images found")

    # Visual-similarity connected components (conservative grouping candidates).
    parent = list(range(len(rows)))
    size = [1] * len(rows)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]

    pairs = []
    exact_pairs = 0
    cross_class_pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            exact = rows[i]["sha256"] == rows[j]["sha256"]
            distance = (rows[i]["_phash_int"] ^ rows[j]["_phash_int"]).bit_count()
            if exact or distance <= threshold:
                union(i, j)
                cross_class = rows[i]["class"] != rows[j]["class"]
                pairs.append({
                    "file_1": rows[i]["relative_path"],
                    "file_2": rows[j]["relative_path"],
                    "class_1": rows[i]["class"],
                    "class_2": rows[j]["class"],
                    "exact_sha256_match": exact,
                    "phash_distance": distance,
                    "cross_class": cross_class,
                })
                exact_pairs += int(exact)
                cross_class_pairs += int(cross_class)

    clusters = {}
    for i, row in enumerate(rows):
        clusters.setdefault(find(i), []).append(i)
    ordered = sorted(clusters.values(), key=lambda ix: rows[min(ix)]["relative_path"])
    group_summaries = []
    for idx, members in enumerate(ordered, 1):
        group_id = f"similarity_group_{idx:04d}"
        for i in members:
            rows[i]["candidate_group"] = group_id
        if len(members) > 1:
            group_summaries.append({
                "candidate_group": group_id,
                "image_count": len(members),
                "classes": dict(Counter(rows[i]["class"] for i in members)),
                "files": [rows[i]["relative_path"] for i in members],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "image_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["relative_path", "class", "width", "height", "sha256", "phash", "candidate_group"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in cols})
    with (out_dir / "possible_duplicate_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        cols = ["file_1", "file_2", "class_1", "class_2", "exact_sha256_match", "phash_distance", "cross_class"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(pairs)
    summary = {
        "status": "exploratory similarity audit; not an independent-provenance assessment",
        "dataset_root": str(root.resolve()),
        "phash_algorithm": "ImageHash pHash",
        "phash_hamming_distance_threshold": threshold,
        "image_count": len(rows),
        "counts_by_class": dict(Counter(r["class"] for r in rows)),
        "possible_duplicate_pairs": len(pairs),
        "exact_duplicate_pairs": exact_pairs,
        "cross_class_similar_pairs": cross_class_pairs,
        "similarity_groups_with_multiple_images": len(group_summaries),
        "images_in_multi_image_groups": sum(x["image_count"] for x in group_summaries),
        "multi_image_group_details": group_summaries,
        "caution": "Image-based groups may miss distinct images from the same plant or field. Sequential file names do not establish independence; seek collection metadata or document the limitation.",
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"multi_image_group_details", "caution"}}, indent=2))
    print(f"Reports saved to: {out_dir.resolve()}")
    return summary


def main():
    p = argparse.ArgumentParser(description="Audit Ethiopian wheat images for candidate related-image groups")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--phash-threshold", type=int, default=5)
    args = p.parse_args()
    audit(args.root, args.out_dir, args.phash_threshold)


if __name__ == "__main__":
    main()
