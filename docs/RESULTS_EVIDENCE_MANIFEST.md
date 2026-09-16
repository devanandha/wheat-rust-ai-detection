# Results Evidence Manifest

Status labels: **verified** = machine-readable evidence is committed; **documented** = reported in project documentation but underlying output is not committed; **missing** = required before submission.

| ID | Result or artifact | Status | Current evidence / required action |
|---|---|---|---|
| E01 | MobileNetV2 internal result | verified | `outputs/classifier/classification_report.json`, `confusion_matrix.csv`, `training_history.json` |
| E02 | Autoencoder anomaly result | verified | `outputs/anomaly/anomaly_report.json` and `outputs/anomaly_v2/anomaly_report.json` |
| E03 | Original dataset counts and split | verified | `outputs/dataset_audit.json` records all train/validation class counts, corrupt-file checks and exact-duplicate groups |
| E04 | External dataset inventory and class mapping | verified | `outputs/evidence/external_dataset_inventory.json` records the DOI, inclusion/exclusion mapping, 489 relative filenames, dimensions, sizes, SHA-256 and pHash values |
| E05 | MobileNetV2 external predictions/report | verified | `outputs/external_validation/` contains the report, confusion matrix, summary and sanitised misclassification rows |
| E06 | EfficientNetB0 internal outputs | verified | `outputs/baselines/efficientnetb0/internal/` report and confusion matrix; training history retained separately |
| E07 | EfficientNetB0 external outputs | verified | `outputs/baselines/efficientnetb0/external/` report and confusion matrix |
| E08 | ResNet50 internal outputs | verified | `outputs/baselines/resnet50/internal/` report and confusion matrix; interrupted-training limitation documented |
| E09 | ResNet50 external outputs | verified | `outputs/baselines/resnet50/external/` report and confusion matrix |
| E10 | Train/validation exact and near-duplicate audit | verified | `outputs/failure_analysis/near_duplicate_pairs.csv`, `outputs/dataset_audit.json`, and the evaluation in `outputs/classifier_without_near_duplicate/` |
| E11 | Cross-dataset duplicate audit | verified | `outputs/evidence/cross_dataset_duplicate_audit.json` records 3,679 original and 489 external images checked, with zero exact and zero pHash matches at distance <= 5 |
| E12 | Grad-CAM coherent-block image-level results | verified | `outputs/gradcam_faithfulness_blocks/gradcam_faithfulness.csv` and `summary.json` |
| E13 | Reproducibility metadata | verified with limitation | `outputs/evidence/run_metadata.json` records the evidence-generation environment, dependency versions, commit and model hashes; historical training hardware/package versions remain explicitly unknown |

## Submission gate

All numerical result groups now have committed machine-readable evidence. The historical training-environment limitation recorded under E13 must remain disclosed. Dataset images themselves should not be committed when licensing does not permit redistribution.
