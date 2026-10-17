"""Create reproducible, leakage-screened, balanced wheat-rust training cohorts.

This stages image files only; it never trains a model or touches the held-out
external test images. Install project requirements (including ImageHash) first.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
from datetime import datetime, timezone

from PIL import Image

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fingerprint(path, use_phash):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    with Image.open(path) as im:
        im.verify()
    phash = None
    if use_phash:
        import imagehash
        with Image.open(path) as im:
            phash = int(str(imagehash.phash(im.convert("RGB"))), 16)
    return {"path": path, "sha256": h.hexdigest(), "phash": phash}


def inventory(root, use_phash, require_all):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Missing dataset folder: {root}")
    rows = defaultdict(list)
    for cls in CLASSES:
        folder = root / cls
        if not folder.is_dir():
            if require_all:
                raise ValueError(f"Missing expected class directory: {folder}")
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUFFIXES:
                try:
                    rows[cls].append(fingerprint(path, use_phash))
                except Exception as ex:
                    raise ValueError(f"Could not read image {path}: {ex}") from ex
    if not any(rows.values()):
        raise ValueError(f"Dataset has no class images: {root}")
    return rows


def flatten(rows):
    return [p for cls in CLASSES for p in rows[cls]]


def check_and_filter_sources(sources, internal, external, threshold):
    """Exclude training candidates overlapping either held-out dataset."""
    test_rows = flatten(internal) + flatten(external)
    test_hashes = {r["sha256"] for r in test_rows}
    if len(test_hashes) != len(test_rows):
        raise ValueError("Held-out validation/test images have exact duplicate files; inspect provenance first.")
    seen = set()
    kept = {}
    removed = []
    for name, byclass in sources.items():
        out = defaultdict(list)
        for cls in CLASSES:
            for rec in byclass[cls]:
                reason = None
                if rec["sha256"] in test_hashes:
                    reason = "exact_holdout_overlap"
                elif rec["sha256"] in seen:
                    reason = "exact_training_duplicate"
                elif threshold >= 0:
                    matched = next((
                        (other, (rec["phash"] ^ other["phash"]).bit_count())
                        for other in test_rows
                        if (rec["phash"] ^ other["phash"]).bit_count() <= threshold
                    ), None)
                    if matched is not None:
                        reason = "perceptual_holdout_overlap"
                if reason:
                    record = {"source": name, "class": cls, "file": rec["path"].name, "reason": reason}
                    if reason == "perceptual_holdout_overlap":
                        record["matched_holdout_file"] = matched[0]["path"].name
                        record["phash_distance"] = matched[1]
                    removed.append(record)
                else:
                    seen.add(rec["sha256"])
                    out[cls].append(rec)
        kept[name] = out
    return kept, removed


def _sample_extra(extra_sources, cls, need, rng):
    """Round-robin sampling makes contributions from multiple sources visible."""
    pools = {}
    for name, byclass in extra_sources.items():
        pool = list(byclass[cls]); rng.shuffle(pool)
        if pool:
            pools[name] = pool
    if sum(map(len, pools.values())) < need:
        raise ValueError(f"Not enough independent {cls} images: need {need}, available {sum(map(len,pools.values()))}")
    result = []
    while len(result) < need:
        for name in sorted(pools):
            if pools[name] and len(result) < need:
                result.append((name, pools[name].pop()))
    return result


def select_cohort(kept, size, mix, seed):
    rng = random.Random(seed)
    primary = kept["primary"]
    extra = {name: pool for name, pool in kept.items() if name != "primary"}
    k_extra = round(size * mix)
    k_primary = size - k_extra
    selected = []
    for cls in CLASSES:
        if len(primary[cls]) < k_primary:
            raise ValueError(f"Insufficient primary {cls}: need {k_primary}, have {len(primary[cls])}")
        selected.extend(("primary", cls, rec) for rec in rng.sample(primary[cls], k_primary))
        selected.extend((source, cls, rec) for source, rec in _sample_extra(extra, cls, k_extra, rng))
    return selected


def stage(selected, destination, mode):
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for cls in CLASSES:
        (destination / cls).mkdir()
    for name, cls, rec in selected:
        src = rec["path"]
        # Prefix source and content hash to avoid filename collisions.
        target = destination / cls / f"{name}_{rec['sha256'][:16]}{src.suffix.lower()}"
        if target.exists():
            raise ValueError(f"Unexpected filename collision: {target}")
        if mode == "hardlink":
            os.link(src, target)  # Requires same filesystem; failure is deliberate.
        else:
            shutil.copy2(src, target)
        rows.append({"source": name, "class": cls, "original_file": src.name,
                     "staged_file": target.name, "sha256": rec["sha256"],
                     "phash": None if rec["phash"] is None else f"{rec['phash']:016x}"})
    return rows


def source_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=CLASS_FOLDER_ROOT")
    name, path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name == "primary":
        raise argparse.ArgumentTypeError("Use a simple unique name, not 'primary'")
    return name, Path(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary-train", type=Path, required=True)
    p.add_argument("--internal-val", type=Path, required=True)
    p.add_argument("--external-test", type=Path, required=True)
    p.add_argument("--additional-source", type=source_arg, action="append", default=[])
    p.add_argument("--per-class", type=int, nargs="+", default=[150, 300])
    p.add_argument("--mix", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    p.add_argument("--seed", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--out-dir", type=Path, default=Path("experiments/generalisation"))
    p.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    p.add_argument("--phash-threshold", type=int, default=5,
                   help="Hamming distance cutoff; -1 disables pHash ONLY for preliminary testing")
    args = p.parse_args(argv)
    if args.out_dir.exists():
        p.error(f"Output already exists; choose a fresh --out-dir: {args.out_dir}")
    if args.phash_threshold < -1 or args.phash_threshold > 64:
        p.error("--phash-threshold must be -1..64")
    if any(n <= 0 for n in args.per_class) or any(not 0 <= m <= 1 for m in args.mix):
        p.error("Sizes must be positive and mixing fractions must be within 0..1")
    names = [n for n, _ in args.additional_source]
    if len(set(names)) != len(names):
        p.error("Duplicate --additional-source names")
    if not names and any(x > 0 for x in args.mix):
        p.error("Mixing needs one or more --additional-source entries")
    roots = [args.primary_train, args.internal_val, args.external_test] + [v for _, v in args.additional_source]
    if len(set(p.resolve() for p in roots)) != len(roots):
        p.error("Training, validation and test sources must have different roots")
    phash = args.phash_threshold >= 0
    if phash:
        try:
            import imagehash  # noqa: F401
        except ImportError:
            p.error("Install project requirements first: pip install -r requirements.txt (requires ImageHash)")
    internal = inventory(args.internal_val, phash, True)
    external = inventory(args.external_test, phash, True)
    sources = {"primary": inventory(args.primary_train, phash, True)}
    for name, root in args.additional_source:
        sources[name] = inventory(root, phash, False)
    kept, removed = check_and_filter_sources(sources, internal, external, args.phash_threshold)
    specs = []
    for size in sorted(set(args.per_class)):
        for mix in sorted(set(args.mix)):
            for seed in sorted(set(args.seed)):
                picked = select_cohort(kept, size, mix, seed)
                code = f"n{size}_mix{int(round(mix*100)):03d}_seed{seed}"
                specs.append((code, size, mix, seed, picked))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    run_manifest = {"protocol": "fixed-class-balanced size/diversity ablation, external test never trained on",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "classes": list(CLASSES), "train_primary": str(args.primary_train.resolve()),
                    "train_additional": {k: str(v.resolve()) for k, v in args.additional_source},
                    "internal_validation": str(args.internal_val.resolve()),
                    "external_heldout_test": str(args.external_test.resolve()),
                    "overlap_screen": "SHA-256 and ImageHash pHash" if phash else "SHA-256 ONLY - not publishable",
                    "phash_threshold": args.phash_threshold, "source_counts_before_filter": {
                        name: {c: len(pool[c]) for c in CLASSES} for name, pool in sources.items()},
                    "source_counts_after_filter": {
                        name: {c: len(pool[c]) for c in CLASSES} for name, pool in kept.items()},
                    "filtered_candidates": removed, "conditions": []}
    for code, size, mix, seed, picked in specs:
        path = args.out_dir / code
        rows = stage(picked, path / "train", args.copy_mode)
        (path / "selected_images.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        counts = Counter((r["source"], r["class"]) for r in rows)
        run_manifest["conditions"].append({"id": code, "per_class": size, "mix": mix,
            "seed": seed, "train_dir": str((path / "train").resolve()),
            "counts_by_source_class": [{"source": n, "class": c, "count": v}
                for (n, c), v in sorted(counts.items())]})
    (args.out_dir / "study_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"Created {len(specs)} cohorts in {args.out_dir}")
    print(f"Excluded {len(removed)} candidate images due to holdout overlap or duplicates")
    if not phash:
        print("WARNING: perceptual overlap screening was disabled; do not use results for external expert review.")

if __name__ == "__main__":
    main()
