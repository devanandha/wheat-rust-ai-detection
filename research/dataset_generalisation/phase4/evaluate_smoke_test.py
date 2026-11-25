import os
import csv
import json
import hashlib
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

phase4 = Path(__file__).resolve().parent
phase3 = phase4.parent / "phase3"

dataset = Path(os.environ["PHASE4_DATASET_DIR"]).expanduser()

selection_path = phase4 / "phase4_selection_manifest.json"
protocol_path = phase4 / "phase4_evaluation_protocol.json"

condition = "n150_mix00_seed42"
output = phase4 / "smoke_n150_mix00_seed42"

classes = ["Brown_rust", "Healthy", "Yellow_rust"]
class_to_id = {name: i for i, name in enumerate(classes)}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if output.exists():
    raise SystemExit("STOP: Smoke-test results already exist.")

selection = json.loads(selection_path.read_text(encoding="utf-8"))
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

if sha256(selection_path) != protocol["selection_manifest_sha256"]:
    raise SystemExit("STOP: Selection manifest changed.")

if protocol["class_order"] != classes:
    raise SystemExit("STOP: Class order mismatch.")

if protocol["image_size"] != [224, 224]:
    raise SystemExit("STOP: Image size mismatch.")

if protocol["preprocessing"] != "embedded_in_saved_model":
    raise SystemExit("STOP: Preprocessing mismatch.")

records = [
    r for r in selection["records"]
    if r["status"] == "included"
]

records.sort(key=lambda r: r["relative_path"])

if len(records) != 199:
    raise SystemExit("STOP: Expected 199 evaluation images.")

checkpoint = phase3 / condition / "results" / "best_model.keras"

model_record = next(
    (m for m in protocol["models"] if m["condition"] == condition),
    None
)

if model_record is None:
    raise SystemExit("STOP: Model missing from frozen protocol.")

if sha256(checkpoint) != model_record["checkpoint_sha256"]:
    raise SystemExit("STOP: Model checkpoint hash mismatch.")

print("Loading model:", condition, flush=True)

model = tf.keras.models.load_model(
    checkpoint,
    compile=False
)

print("Model loaded successfully.", flush=True)
print("Evaluating 199 screened images...", flush=True)

y_true = []
y_pred = []
all_probabilities = []

batch_size = 8

for start in range(0, len(records), batch_size):
    batch_records = records[start:start + batch_size]
    images = []

    for record in batch_records:
        path = dataset / record["relative_path"]

        if not path.is_file():
            raise RuntimeError("Selected image missing.")

        if sha256(path) != record["sha256"]:
            raise RuntimeError("Selected image hash mismatch.")

        image = tf.keras.utils.load_img(
            path,
            target_size=(224, 224),
            color_mode="rgb",
            interpolation="bilinear"
        )

        images.append(
            tf.keras.utils.img_to_array(image)
        )

        y_true.append(class_to_id[record["class"]])

    batch = np.stack(images).astype("float32")

    # Raw 0-255 RGB input: preprocessing is already in the model.
    probabilities = model.predict(batch, verbose=0)

    if probabilities.shape != (len(batch_records), 3):
        raise RuntimeError("Unexpected model output shape.")

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Non-finite prediction probabilities.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError("Invalid softmax probabilities.")

    y_pred.extend(np.argmax(probabilities, axis=1).tolist())
    all_probabilities.extend(probabilities.tolist())

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred,
    labels=[0, 1, 2],
    average="macro",
    zero_division=0
)

report = classification_report(
    y_true, y_pred,
    labels=[0, 1, 2],
    target_names=classes,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    y_true, y_pred,
    labels=[0, 1, 2]
)

output.mkdir(parents=True, exist_ok=False)

with (output / "predictions.csv").open(
    "x", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "relative_path", "actual", "predicted",
        *[f"prob_{name}" for name in classes]
    ])

    for record, prediction, probabilities in zip(
        records, y_pred, all_probabilities
    ):
        writer.writerow([
            record["relative_path"],
            record["class"],
            classes[prediction],
            *probabilities
        ])

(output / "classification_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

np.savetxt(
    output / "confusion_matrix.csv",
    matrix, fmt="%d", delimiter=","
)

print("\n=== PHASE 4 SMOKE TEST RESULTS ===")
print("Model:", condition)
print("Images evaluated:", len(records))
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro-F1: {macro_f1 * 100:.2f}%")

for name in classes:
    print(
        f"{name} recall: "
        f"{report[name]['recall'] * 100:.2f}%"
    )

print("\nConfusion matrix:")
print(matrix)

print("\nPHASE 4 SMOKE TEST COMPLETE")
