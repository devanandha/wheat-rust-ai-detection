import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .data import EXPECTED_CLASSES


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(data_dir: Path):
    counts = Counter()
    corrupt = []
    hashes = defaultdict(list)
    dimensions = Counter()

    for split in ("train", "val"):
        for class_name in EXPECTED_CLASSES:
            folder = data_dir / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")
            for path in sorted(folder.iterdir()):
                if not path.is_file():
                    continue
                counts[f"{split}/{class_name}"] += 1
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        dimensions[image.size] += 1
                    hashes[file_hash(path)].append(str(path))
                except (UnidentifiedImageError, OSError) as error:
                    corrupt.append({"path": str(path), "error": str(error)})

    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "counts": dict(counts),
        "corrupt_files": corrupt,
        "exact_duplicate_groups": duplicates,
        "unique_dimensions": len(dimensions),
        "most_common_dimensions": [
            {"width": size[0], "height": size[1], "count": count}
            for size, count in dimensions.most_common(10)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset integrity and structure.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit.json"))
    args = parser.parse_args()
    result = audit(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

