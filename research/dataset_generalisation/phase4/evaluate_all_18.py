import os
import csv
import gc
import json
import hashlib
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

phase4 = Path(__file__).resolve().parent
phase3 = phase4.parent / "phase3"

dataset = Path(os.environ["PHASE4_DATASET_DIR"]).expanduser()

selection_path = phase4 / "phase4_selection_manifest.json"
protocol_path = phase4 / "phase4_evaluation_protocol.json"
output = phase4 / "evaluation_all_18"

classes = ["Brown_rust", "Healthy", "Yellow_rust"]
class_ids = {name: i for i, name in enumerate(classes)}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if output.exists():
    raise SystemExit("STOP: Evaluation output already exists.")

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

if protocol["additional_normalisation"] is not False:
    raise SystemExit("STOP: Normalisation mismatch.")

records = sorted(
    [r for r in selection["records"] if r["status"] == "included"],
    key=lambda r: r["relative_path"]
)

if len(records) != 199 or len(protocol["models"]) != 18:
    raise SystemExit("STOP: Unexpected image or model count.")

if len({m["condition"] for m in protocol["models"]}) != 18:
    raise SystemExit("STOP: Duplicate model conditions.")

expected_counts = {
    c: sum(r["class"] == c for r in records)
    for c in classes
}

if expected_counts != protocol["class_counts"]:
    raise SystemExit("STOP: Class counts differ from protocol.")

# Verify model files before inference.
for item in protocol["models"]:
    name = item["condition"]

    if not name.startswith(("n150_mix", "n300_mix")):
        raise SystemExit("STOP: Unexpected condition.")

    path = phase3 / name / "results/best_model.keras"

    if not path.is_file() or sha256(path) != item["checkpoint_sha256"]:
        raise SystemExit(f"STOP: Checkpoint verification failed: {name}")

# Verify and load the fixed image set once.
print("Loading 199 verified RGB images...", flush=True)

images = []
y_true = []

for record in records:
    relative = Path(record["relative_path"])

    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("STOP: Unsafe image path.")

    path = dataset / relative

    if not path.is_file() or sha256(path) != record["sha256"]:
        raise SystemExit("STOP: Image verification failed.")

    image = tf.keras.utils.load_img(
        path,
        target_size=(224, 224),
        color_mode="rgb",
        interpolation="bilinear"
    )

    images.append(tf.keras.utils.img_to_array(image))
    y_true.append(class_ids[record["class"]])

images = np.stack(images).astype("float32")
y_true = np.asarray(y_true)

print("Images verified:", len(images), flush=True)

output.mkdir(parents=True, exist_ok=False)
summary = []

for index, item in enumerate(protocol["models"], start=1):
    name = item["condition"]
    checkpoint = phase3 / name / "results/best_model.keras"

    print(f"\n[{index}/18] Evaluating {name}", flush=True)

    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(
        checkpoint,
        compile=False
    )

    probabilities = model.predict(
        images,
        batch_size=8,
        verbose=0
    )

    if probabilities.shape != (199, 3):
        raise RuntimeError(f"Unexpected output shape: {name}")

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(f"Invalid predictions: {name}")

    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-3):
        raise RuntimeError(f"Softmax check failed: {name}")

    y_pred = probabilities.argmax(axis=1)

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

    model_dir = output / name
    model_dir.mkdir()

    (model_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    np.savetxt(
        model_dir / "confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=","
    )

    with (model_dir / "predictions.csv").open(
        "x", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "relative_path", "actual", "predicted",
            *[f"prob_{c}" for c in classes]
        ])

        for record, pred, probs in zip(
            records, y_pred, probabilities
        ):
            writer.writerow([
                record["relative_path"],
                record["class"],
                classes[int(pred)],
                *[float(x) for x in probs]
            ])

    row = {
        "condition": name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "brown_recall": report["Brown_rust"]["recall"],
        "healthy_recall": report["Healthy"]["recall"],
        "yellow_recall": report["Yellow_rust"]["recall"],
    }

    summary.append(row)

    print(
        f"Accuracy: {accuracy*100:.2f}% | "
        f"Macro-F1: {macro_f1*100:.2f}%",
        flush=True
    )

    del model
    gc.collect()

with (output / "summary_all_18.csv").open(
    "x", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)

(output / "evaluation_metadata.json").write_text(
    json.dumps({
        "protocol_sha256": sha256(protocol_path),
        "selection_manifest_sha256": sha256(selection_path),
        "images_evaluated": len(records),
        "models_evaluated": len(summary),
        "evaluation_status": "screened_exploratory_external"
    }, indent=2),
    encoding="utf-8"
)

print("\n=== PHASE 4 EVALUATION SUMMARY ===")

for row in summary:
    print(
        row["condition"],
        f"accuracy={row['accuracy']*100:.2f}%",
        f"macro_f1={row['macro_f1']*100:.2f}%"
    )

print("\nALL 18 PHASE 4 EVALUATIONS COMPLETE")
