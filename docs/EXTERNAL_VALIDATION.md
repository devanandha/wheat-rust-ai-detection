# External Validation and Generalisation Analysis

## Objective

The original Wheat Rust AI MobileNetV2 classifier achieved 98.78% accuracy on its internal validation set. However, strong performance on data drawn from the same source does not necessarily demonstrate that a model will generalise to images collected under different conditions.

To investigate this limitation, the trained model was evaluated without retraining on an external-source wheat disease dataset.

## External Dataset

External validation used images from the **Wheat Disease Dataset - Small**, associated with work by researchers at the John Innes Centre.

Official dataset DOI:

https://doi.org/10.5281/zenodo.7307816

The complete dataset contains five classes:

- Brown Rust
- Healthy
- Mildew
- Septoria
- Yellow Rust

Because the existing classifier was trained only to distinguish Brown Rust, Healthy, and Yellow Rust, external classification accuracy was evaluated only on the three directly compatible classes.

External evaluation subset:

| Class | Images |
|---|---:|
| Brown Rust | 128 |
| Healthy | 122 |
| Yellow Rust | 239 |
| **Total** | **489** |

The external images were not used for model training or model selection.

## Cross-Dataset Overlap Checks

To reduce the possibility that external performance was influenced by duplicated images, the external subset was compared with the original 3,679-image dataset.

Two checks were performed:

1. SHA-256 exact-file comparison.
2. Perceptual hashing (pHash), using a Hamming-distance threshold of <= 5 to identify visually near-duplicate images.

Results:

- Exact cross-dataset duplicates: **0**
- Near-duplicate cross-dataset pairs (pHash <= 5): **0**

These checks found no exact or close perceptual matches at the selected threshold.

## External Validation Results

The original trained MobileNetV2 model was evaluated on all 489 compatible external images without retraining.

### Overall Performance

| Evaluation | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Original internal validation | 728 | 737 | 98.78% |
| External validation | 188 | 489 | 38.45% |

The substantial performance reduction demonstrates that the high internal validation accuracy did not transfer to this external image distribution.

### Per-Class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Brown Rust | 0.332 | 0.578 | 0.422 | 128 |
| Healthy | 0.396 | 0.811 | 0.532 | 122 |
| Yellow Rust | 0.938 | 0.063 | 0.118 | 239 |

Macro-average F1-score: **0.357**

Weighted-average F1-score: **0.301**

## Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | Brown Rust | Healthy | Yellow Rust |
|---|---:|---:|---:|
| Brown Rust | 74 | 53 | 1 |
| Healthy | 23 | 99 | 0 |
| Yellow Rust | 126 | 98 | 15 |

The largest failure occurred for Yellow Rust.

Of 239 externally labelled Yellow Rust images:

- 15 were correctly classified as Yellow Rust.
- 126 were classified as Brown Rust.
- 98 were classified as Healthy.

This produced a Yellow Rust recall of approximately 6.28%.

## Confidence of Incorrect Predictions

The external failures were not limited to low-confidence decisions.

For example:

- Yellow Rust -> Brown Rust: 126 errors, mean confidence 83.89%, median confidence 89.82%.
- Yellow Rust -> Healthy: 98 errors, mean confidence 82.68%, median confidence 90.00%.
- Brown Rust -> Healthy: 53 errors, mean confidence 84.24%, median confidence 85.90%.

Several incorrect predictions had confidence values close to 1.0.

This indicates that under the external image distribution, prediction confidence alone is not a reliable indicator of correctness.

## Grad-CAM Failure Inspection

Representative high-confidence external failures were inspected using Grad-CAM.

Examples included:

- YellowRust2280: Yellow Rust -> Healthy.
- YellowRust2405: Yellow Rust -> Brown Rust.
- BrownRust2075: Brown Rust -> Healthy.

The visualisations showed different attention behaviours. Some activation patterns were distributed across plant and contextual regions, while other cases showed stronger attention on leaf structures despite an incorrect class prediction.

Grad-CAM indicates image regions that influenced a model prediction. It does not establish that highlighted regions correspond to biologically confirmed disease symptoms. Expert spatial annotations would be required for quantitative biological localisation claims.

## Interpretation

The external evaluation identifies a substantial generalisation gap between the original validation distribution and a separate external image source.

The result suggests that the original 98.78% validation accuracy should not be interpreted as evidence of equivalent performance under different field or acquisition conditions.

In particular, the combination of:

- substantially lower external accuracy,
- very low external Yellow Rust recall,
- high-confidence incorrect predictions, and
- differing Grad-CAM attention patterns

motivates further investigation into dataset shift, robustness, model calibration, and disease-feature representation.

## Limitations

This evaluation has several limitations.

First, only the three classes shared with the original classifier were included. Mildew and Septoria were excluded because the current model was not trained to recognise those classes.

Second, the absence of exact or pHash near-duplicates does not prove complete statistical independence between datasets; it only provides evidence that direct image overlap was not detected using the implemented checks.

Third, Grad-CAM visualisations are qualitative explanations and have not been validated against expert disease-region annotations.

Finally, this evaluation measures performance on one additional external dataset and should not be interpreted as establishing performance across all real-world wheat-growing environments.

## Next Steps

The external-validation results motivate several follow-up experiments:

1. Compare MobileNetV2 with additional controlled baseline architectures.
2. Investigate dataset and acquisition differences between the original and external datasets.
3. Analyse external failure cases systematically.
4. Investigate prediction calibration under distribution shift.
5. Evaluate explainability quantitatively if suitable expert spatial annotations become available.
6. Explore methods for improving cross-dataset generalisation without compromising evaluation independence.

The broader research question emerging from this work is:

> **How well do deep-learning models for wheat-rust classification generalise beyond their training distribution, and can explainability and failure-case analysis reveal the visual factors underlying model errors?**