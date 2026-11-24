import os
import json
import hashlib
from pathlib import Path
from collections import Counter

phase4 = Path(__file__).resolve().parent
phase3 = phase4.parent / "phase3"

dataset = Path(os.environ["PHASE4_DATASET_DIR"]).expanduser()

selection_path = phase4 / "phase4_selection_manifest.json"
protocol_path = phase4 / "phase4_evaluation_protocol.json"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

selection = json.loads(selection_path.read_text(encoding="utf-8"))
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

errors = []

if sha256(selection_path) != protocol["selection_manifest_sha256"]:
    errors.append("Selection manifest SHA-256 mismatch")

expected_classes = ["Brown_rust", "Healthy", "Yellow_rust"]

if protocol["class_order"] != expected_classes:
    errors.append("Incorrect class ordering")

if protocol["image_size"] != [224, 224]:
    errors.append("Unexpected image dimensions")

if protocol["preprocessing"] != "embedded_in_saved_model":
    errors.append("Preprocessing configuration mismatch")

if protocol["additional_normalisation"] is not False:
    errors.append("Additional normalisation unexpectedly enabled")

included = [
    record for record in selection["records"]
    if record["status"] == "included"
]

counts = Counter(record["class"] for record in included)

if len(included) != 199:
    errors.append("Expected exactly 199 images")

if dict(counts) != protocol["class_counts"]:
    errors.append("Class counts do not match protocol")

print("=== PHASE 4 PRE-INFERENCE VERIFICATION ===")

print("\nChecking selected images...")

for record in included:
    relative = Path(record["relative_path"])

    if relative.is_absolute() or ".." in relative.parts:
        errors.append("Unsafe relative image path")
        continue

    path = dataset / relative

    if not path.is_file():
        errors.append("Missing selected image")
        continue

    if sha256(path) != record["sha256"]:
        errors.append("Selected image hash mismatch")

print("Selected images:", len(included))
print("Class counts:", dict(counts))

print("\nChecking canonical model checkpoints...")

models = protocol["models"]

if len(models) != 18:
    errors.append("Expected exactly 18 models")

if len({m["condition"] for m in models}) != len(models):
    errors.append("Duplicate model condition")

for model in models:
    name = model["condition"]

    if "/" in name or "\\" in name or ".." in name:
        errors.append("Unsafe model condition")
        continue

    path = phase3 / name / "results" / "best_model.keras"

    if not path.is_file():
        errors.append("Missing model checkpoint")
        continue

    if sha256(path) != model["checkpoint_sha256"]:
        errors.append("Model checkpoint SHA-256 mismatch")

print("Models checked:", len(models))

print("\n=== VERIFICATION RESULT ===")

if errors:
    print("FAILED")
    for error in sorted(set(errors)):
        print("ERROR:", error)
    raise SystemExit(1)

print("All selected image hashes verified")
print("All model checkpoint hashes verified")
print("Class ordering verified")
print("Preprocessing configuration verified")
print("PHASE 4 INPUT VERIFICATION PASSED")
