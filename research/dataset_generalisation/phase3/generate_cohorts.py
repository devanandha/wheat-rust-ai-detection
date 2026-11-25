import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "research" / "dataset_generalisation" / "phase3"
INPUT = BASE / "audit_outputs" / "eligibility_manifest_v1.json"
OUTPUT = BASE / "audit_outputs" / "cohorts_v1"
CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
SEEDS = (42, 43, 44)
SIZES = (150, 300)
MIXES = (0, 25, 50)

EXPECTED_MANIFEST_SHA256 = "10b80ee46fe6ab9708395cc140099aef851105a0890408b99a2012014f8322e9"

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha256_file(INPUT) != EXPECTED_MANIFEST_SHA256:
    raise SystemExit("STOP: Eligibility manifest fingerprint has changed.")

manifest = json.loads(INPUT.read_text(encoding="utf-8"))
eligible = [r for r in manifest["candidates"] if r["eligible"]]

assert len(eligible) == 3705

pools = {}
for cls in CLASSES:
    pools[(cls, "primary")] = sorted(
        [r for r in eligible if r["class"] == cls and r["source"] == "primary"],
        key=lambda r: r["path"]
    )
    pools[(cls, "additional")] = sorted(
        [r for r in eligible if r["class"] == cls and r["source"] != "primary"],
        key=lambda r: (r["source"], r["path"])
    )

if OUTPUT.exists():
    raise SystemExit(f"STOP: Output folder already exists: {OUTPUT}")

plans = []

for size in SIZES:
    for mix in MIXES:
        additional_count = {0: 0, 25: size // 4 + (1 if size % 4 >= 2 else 0), 50: size // 2}[mix]
        primary_count = size - additional_count

        for seed in SEEDS:
            selected = []

            for cls in CLASSES:
                for source_type, count in (
                    ("primary", primary_count),
                    ("additional", additional_count)
                ):
                    pool = pools[(cls, source_type)]

                    if len(pool) < count:
                        raise SystemExit(
                            f"STOP: Insufficient {source_type} images for {cls}"
                        )

                    rng = random.Random(
                        f"phase3-v1|{size}|{mix}|{seed}|{cls}|{source_type}"
                    )
                    selected.extend(rng.sample(pool, count))

            assert len(selected) == size * len(CLASSES)
            assert len({r["path"] for r in selected}) == len(selected)

            plans.append({
                "condition": f"n{size}_mix{mix:02d}_seed{seed}",
                "per_class_size": size,
                "additional_fraction_percent": mix,
                "seed": seed,
                "primary_per_class": primary_count,
                "additional_per_class": additional_count,
                "selected": selected,
            })

print("Verifying selected image hashes...")
unique = {r["path"]: r for plan in plans for r in plan["selected"]}

for index, record in enumerate(unique.values(), 1):
    path = Path(record["path"])
    if not path.is_file():
        raise SystemExit(f"STOP: Missing image: {path}")
    if sha256_file(path) != record["sha256"]:
        raise SystemExit(f"STOP: Image SHA-256 mismatch: {path}")
    if index % 500 == 0:
        print(f"Verified {index}/{len(unique)} unique images")

print(f"Verified all {len(unique)} unique selected images.")

OUTPUT.mkdir(parents=True, exist_ok=False)
summary = []

for plan in plans:
    name = plan["condition"]
    destination = OUTPUT / f"{name}.json"

    destination.write_text(
        json.dumps(plan, indent=2),
        encoding="utf-8"
    )

    counts = Counter(
        (r["source"], r["class"]) for r in plan["selected"]
    )

    summary.append({
        "condition": name,
        "total_images": len(plan["selected"]),
        "per_class_size": plan["per_class_size"],
        "mix_percent": plan["additional_fraction_percent"],
        "seed": plan["seed"],
        "counts": [
            {"source": source, "class": cls, "count": count}
            for (source, cls), count in sorted(counts.items())
        ],
        "manifest_sha256": sha256_file(destination),
    })

summary_path = OUTPUT / "cohort_summary.json"
summary_path.write_text(
    json.dumps({
        "eligibility_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sampling_algorithm": "Python random.Random seeded by condition/class/source string",
        "rounding_rule": "25% of 150 = 38 additional images per class",
        "conditions": summary,
    }, indent=2),
    encoding="utf-8"
)

print("\nCOHORT GENERATION: PASS")
print("Conditions:", len(plans))
print("Unique selected images:", len(unique))
print("Output:", OUTPUT)

for item in summary:
    print(item["condition"], "images:", item["total_images"])
