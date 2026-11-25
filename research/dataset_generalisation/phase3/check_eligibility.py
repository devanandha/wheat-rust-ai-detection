import hashlib
import json
from collections import Counter
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path.cwd()
HISTORICAL = Path(
    r"F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai"
    r"\experiments\diversity_sources_20261016\staging_manifest.json"
)
PRIMARY = Path(
    r"F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data\train"
)
CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THRESHOLD = 5


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        phash = int(str(imagehash.phash(image.convert("RGB"))), 16)
    return {
        "path": str(path),
        "sha256": digest,
        "phash": phash,
    }


def scan_folder(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return [
        fingerprint(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]


def main():
    history = json.loads(HISTORICAL.read_text(encoding="utf-8"))

    protected = {
        "original_validation": ROOT / "data" / "val",
        "jic_benchmark": ROOT / "data" / "external_jic" / "val",
        "iari_validation": Path(history["iari_original_validation"]),
        "iari_test": Path(history["iari_original_test"]),
    }

    holdout_records = history["ethiopia_holdout_file_manifest"]

    print("\n=== PROTECTED DATASETS ===")
    protected_images = []

    for name, folder in protected.items():
        images = scan_folder(folder)
        print(f"{name}: {len(images)}")
        protected_images.extend(images)

    ethiopia_holdout = []
    for record in holdout_records:
        fp = fingerprint(record["original"])
        if fp["sha256"] != record["sha256"]:
            raise ValueError("Ethiopia holdout hash mismatch")
        ethiopia_holdout.append(fp)

    protected_images.extend(ethiopia_holdout)
    print(f"ethiopia_holdout: {len(ethiopia_holdout)}")

    protected_hashes = {r["sha256"] for r in protected_images}
    protected_phashes = [r["phash"] for r in protected_images]

    print("\n=== TRAINING CANDIDATES ===")

    candidates = []

    for cls in CLASSES:
        folder = PRIMARY / cls
        images = scan_folder(folder)
        print(f"primary/{cls}: {len(images)}")
        for fp in images:
            fp.update(source="primary", cls=cls)
            candidates.append(fp)

    for record in history["source_file_manifest"]:
        fp = fingerprint(record["original"])

        if fp["sha256"] != record["sha256"]:
            raise ValueError(
                f"Historical candidate hash mismatch: {fp['path']}"
            )

        fp.update(source=record["source"], cls=record["class"])
        candidates.append(fp)

    print(f"Total candidates: {len(candidates)}")

    print("\n=== LEAKAGE SCREENING ===")

    eligible = []
    exclusions = []
    seen_hashes = set()

    for rec in candidates:
        reason = None

        if rec["sha256"] in protected_hashes:
            reason = "exact_protected_overlap"

        elif any(
            (rec["phash"] ^ h).bit_count() <= THRESHOLD
            for h in protected_phashes
        ):
            reason = "perceptual_protected_overlap"

        elif rec["sha256"] in seen_hashes:
            reason = "duplicate_training_image"

        if reason:
            exclusions.append({
                "source": rec["source"],
                "class": rec["cls"],
                "path": rec["path"],
                "reason": reason,
            })
        else:
            eligible.append(rec)
            seen_hashes.add(rec["sha256"])

    print("Excluded:", len(exclusions))
    for reason, count in sorted(
        Counter(r["reason"] for r in exclusions).items()
    ):
        print(f"  {reason}: {count}")

    print("\n=== ELIGIBLE COUNTS ===")

    counts = Counter((r["source"], r["cls"]) for r in eligible)

    for source in ("primary", "iari", "ethiopia"):
        for cls in CLASSES:
            print(f"{source:10} {cls:12} {counts[source, cls]}")

    print("\n=== MAXIMUM MIX FEASIBILITY ===")

    feasible = True

    for cls in CLASSES:
        primary_count = counts["primary", cls]
        extra_count = sum(
            counts[source, cls]
            for source in ("iari", "ethiopia")
        )

        ok = primary_count >= 300 and extra_count >= 150
        feasible &= ok

        print(
            f"{cls}: primary={primary_count}, "
            f"additional={extra_count}, "
            f"feasible={ok}"
        )

    print("\nAll 18 conditions feasible:", feasible)

    print("\nNOTE:")
    print("Read-only audit. No files or models were modified.")
    print("Training-candidate near-duplicates are not yet fully screened.")


if __name__ == "__main__":
    main()
