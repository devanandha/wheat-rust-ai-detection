import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path.cwd()
BASE = ROOT / "research" / "dataset_generalisation" / "phase3"
COHORTS = BASE / "audit_outputs" / "cohorts_v1"
STAGED = BASE / "staged_training"
MANIFEST = BASE / "study_manifest_phase3.json"

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not STAGED.is_dir():
    raise SystemExit("STOP: staged_training directory is missing.")

if any(STAGED.iterdir()):
    raise SystemExit("STOP: staged_training is not empty.")

if MANIFEST.exists():
    raise SystemExit("STOP: Study manifest already exists.")

summary = json.loads(
    (COHORTS / "cohort_summary.json").read_text(encoding="utf-8")
)

conditions = summary["conditions"]
if len(conditions) != 18:
    raise SystemExit("STOP: Expected exactly 18 conditions.")

entries = []

for item in conditions:
    name = item["condition"]
    plan_path = COHORTS / f"{name}.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    if sha256(plan_path) != item["manifest_sha256"]:
        raise SystemExit(f"STOP: Cohort manifest changed: {name}")

    records = plan["selected"]
    train_dir = STAGED / name / "train"

    if len(records) != plan["per_class_size"] * 3:
        raise SystemExit(f"STOP: Incorrect cohort size: {name}")

    for record in records:
        source = Path(record["path"])

        if not source.is_file():
            raise SystemExit(f"STOP: Missing source image: {source}")

        if sha256(source) != record["sha256"]:
            raise SystemExit(f"STOP: Source hash mismatch: {source}")

        destination = train_dir / record["class"] / source.name

        if destination.exists():
            raise SystemExit(f"STOP: Duplicate destination: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        if sha256(destination) != record["sha256"]:
            raise SystemExit(f"STOP: Copied image hash mismatch: {destination}")

    entries.append({
        "id": name,
        "seed": plan["seed"],
        "train_dir": str(train_dir.resolve()),
        "per_class_size": plan["per_class_size"],
        "additional_fraction_percent": plan["additional_fraction_percent"],
        "cohort_manifest_sha256": sha256(plan_path),
    })

    print(f"Staged and verified: {name} ({len(records)} images)", flush=True)

study = {
    "study": "wheat_rust_phase3",
    "overlap_screen": "SHA-256 and ImageHash pHash",
    "internal_validation": str((ROOT / "data" / "val").resolve()),
    "external_heldout_test": str((ROOT / "data" / "external_jic" / "val").resolve()),
    "external_benchmark_previously_evaluated": True,
    "eligibility_manifest_sha256": summary["eligibility_manifest_sha256"],
    "conditions": entries,
}

MANIFEST.write_text(json.dumps(study, indent=2), encoding="utf-8")

print("\nSTAGING COMPLETE")
print("Conditions:", len(entries))
print("Study manifest:", MANIFEST)
print("Study manifest SHA256:", sha256(MANIFEST))
