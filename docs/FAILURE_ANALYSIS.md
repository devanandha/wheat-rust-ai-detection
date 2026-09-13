# Wheat Rust AI — Failure Analysis

## Objective

This analysis investigates the failure cases of the MobileNetV2 wheat-rust classifier and examines whether potential train/validation image leakage materially affects the reported validation performance.

## Validation Performance

The MobileNetV2 classifier was evaluated on 737 validation images across three classes:

- Brown Rust: 226 images
- Healthy: 279 images
- Yellow Rust: 232 images

Overall validation accuracy was **98.78% (728/737)**, with **9 misclassified images**.

## Failure Distribution

Of the 9 misclassified images:

- 6 were Brown Rust images
  - 3 predicted as Yellow Rust
  - 3 predicted as Healthy
- 1 Healthy image was predicted as Brown Rust
- 2 Yellow Rust images were predicted as Brown Rust

Several errors occurred with relatively high confidence. The most notable case was `Brown_rust826.jpg`, which was labelled Brown Rust but predicted as Healthy with approximately **99.16% confidence**.

This demonstrates that high aggregate classification accuracy can coexist with confidently incorrect individual predictions.

## Grad-CAM Failure Analysis

Grad-CAM visualisations were generated for all 9 misclassified validation images using the exact image tensors from the evaluation pipeline.

The visualisations provide insight into image regions influencing each predicted class. They should not, however, be interpreted as proof that highlighted regions correspond to biologically confirmed disease symptoms without expert spatial annotations.

The failure cases show different activation patterns, including attention concentrated on leaf regions as well as broader scene/context regions. These observations motivate further investigation into model robustness and the visual features influencing predictions.

## Train/Validation Leakage Investigation

All 3,679 images across the training and validation splits were checked for exact duplicates using SHA-256 hashing.

**Exact duplicate train/validation groups found: 0.**

A second analysis used perceptual hashing (pHash) to identify visually similar images across the two splits.

Using a Hamming-distance threshold of 5, one near-duplicate pair was detected:

- Training: `Healthy203.jpg`
- Validation: `Healthy217.jpg`
- pHash distance: 2

Manual inspection showed that the two images depict essentially the same scene with minor framing differences.

## Sensitivity Analysis

To determine whether this near-duplicate materially affected the reported validation performance, `Healthy217.jpg` was excluded and the evaluation metrics were recalculated.

Original evaluation:

- Images: 737
- Correct: 728
- Errors: 9
- Accuracy: **98.7788%**

Evaluation excluding `Healthy217.jpg`:

- Images: 736
- Correct: 727
- Errors: 9
- Accuracy: **98.7772%**

The difference was approximately **0.0017 percentage points**, indicating that removal of this single identified near-duplicate had a negligible effect on aggregate validation accuracy.

## Current Conclusion

The analysis identified no exact duplicate leakage and one highly similar train/validation image pair. Removing that validation image did not materially alter overall classification accuracy.

However, the presence of several confidently incorrect predictions demonstrates that aggregate accuracy alone is insufficient to characterise model reliability.

These findings motivate further evaluation of generalisation, robustness, dataset splitting methodology, external validation, and explainability.