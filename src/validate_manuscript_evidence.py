"""Check whether manuscript evidence artifacts required by the repository are present."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "E01 MobileNetV2 internal classification report": "outputs/classifier/classification_report.json",
    "E02 Leakage-controlled autoencoder anomaly report": "outputs/anomaly_v2/anomaly_report.json",
    "E03 Original dataset inventory": "outputs/dataset_audit.json",
    "E04 External dataset inventory": "outputs/evidence/external_dataset_inventory.json",
    "E05 MobileNetV2 external report": "outputs/external_validation/classification_report.json",
    "E06 EfficientNetB0 internal report": "outputs/baselines/efficientnetb0/internal/classification_report.json",
    "E07 EfficientNetB0 external report": "outputs/baselines/efficientnetb0/external/classification_report.json",
    "E08 ResNet50 internal report": "outputs/baselines/resnet50/internal/classification_report.json",
    "E09 ResNet50 external report": "outputs/baselines/resnet50/external/classification_report.json",
    "E10 Train-validation duplicate audit": "outputs/failure_analysis/near_duplicate_pairs.csv",
    "E11 Cross-dataset duplicate audit": "outputs/evidence/cross_dataset_duplicate_audit.json",
    "E12 Grad-CAM image-level results": (
        "outputs/gradcam_faithfulness_blocks/gradcam_faithfulness.csv"
    ),
    "E13 Run metadata": "outputs/evidence/run_metadata.json",
}


def main() -> int:
    missing = []
    for label, relative_path in REQUIRED.items():
        exists = (ROOT / relative_path).is_file()
        print(f"{'PASS' if exists else 'MISS'}  {label}: {relative_path}")
        if not exists:
            missing.append(label)
    print(f"\nVerified: {len(REQUIRED) - len(missing)}/{len(REQUIRED)}")
    print(f"Missing: {len(missing)}/{len(REQUIRED)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
