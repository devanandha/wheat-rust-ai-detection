# External Technical Review — Implementation Record

## Purpose

This document records how independent technical feedback on the Wheat Rust AI project was translated into concrete experimental and engineering improvements.

The external review identified five main areas for strengthening the work:

1. Independent external validation
2. Investigation of possible data leakage
3. Comparison with stronger classifier baselines
4. More rigorous evaluation of Grad-CAM explainability
5. A stronger research question focused on robustness, generalisation, and failure analysis

Rather than treating the original high internal validation accuracy as the final result, the project was extended to investigate these limitations directly.

---

# 1. External Validation

## Reviewer Recommendation

Evaluate the classifier on an independent dataset rather than relying primarily on the original train/validation split.

## Implementation

A separate external wheat-disease dataset was introduced for cross-dataset evaluation.

The compatible subset contained:

- 128 Brown rust images
- 122 Healthy images
- 239 Yellow rust images
- 489 images in total

Only classes represented by the existing three-class classifier were included.

The external dataset was kept outside the repository, while a reproducible evaluation workflow was implemented in the project.

## Result

The original MobileNetV2 classifier achieved:

| Evaluation | Accuracy |
|---|---:|
| Internal validation | 98.78% |
| External validation | 38.45% |

The external confusion matrix was:

| Actual / Predicted | Brown rust | Healthy | Yellow rust |
|---|---:|---:|---:|
| Brown rust | 74 | 53 | 1 |
| Healthy | 23 | 99 | 0 |
| Yellow rust | 126 | 98 | 15 |

The substantial reduction in performance demonstrated that the original internal validation result did not transfer reliably to the external dataset.

This changed the direction of the project from reporting high validation accuracy to investigating cross-dataset generalisation.

## Evidence

Relevant implementation and documentation include:

- `src/evaluate_external.py`
- `docs/EXTERNAL_VALIDATION.md`
- Commit `768119e` — Add reproducible external validation workflow
- Commit `7c71c58` — Document external validation and generalisation analysis

---

# 2. Data Leakage and Image Similarity

## Reviewer Recommendation

Investigate whether visually similar images, crops, or images originating from the same source could occur across training and validation sets.

## Implementation

Two complementary checks were performed.

### Exact Duplicate Detection

SHA-256 hashes were compared between training and validation images.

Result:

**0 exact train/validation duplicates detected.**

### Perceptual Similarity Detection

Perceptual hashing was used to identify visually similar train/validation image pairs.

At a perceptual-hash distance threshold of 5 or less, one near-duplicate pair was detected:

- `Healthy203` — training set
- `Healthy217` — validation set

The images appeared to represent the same or highly similar scene with a small difference in crop or framing.

The validation evaluation was repeated after excluding the affected validation image.

| Evaluation | Accuracy |
|---|---:|
| Original validation set | 98.7788% |
| Validation excluding near-duplicate | 98.7772% |

The difference was negligible.

## Interpretation

The detected near-duplicate did not materially explain the high internal validation accuracy.

However, this check does not establish complete source-level or field-level independence because the original dataset does not provide all provenance information required for such verification.

## Evidence

Relevant implementation and documentation include:

- failure and leakage analysis workflow
- perceptual hashing dependency
- `docs/FAILURE_ANALYSIS.md`
- Commit `5c72136` — Add failure analysis and leakage checks
- Commit `3725e1d` — Add ImageHash dependency for leakage analysis

---

# 3. Stronger Classifier Baselines

## Reviewer Recommendation

Compare MobileNetV2 with additional relevant architectures to determine whether the observed behaviour is specific to one classifier.

## Implementation

The classifier pipeline was extended to support:

- MobileNetV2
- EfficientNetB0
- ResNet50

Each architecture used ImageNet initialization, a frozen convolutional backbone, global average pooling, dropout, and a three-class softmax output.

Architecture-appropriate preprocessing was used.

The models were evaluated on both the internal validation set and the same external dataset.

## Results

| Architecture | Internal Accuracy | External Accuracy | Accuracy Drop |
|---|---:|---:|---:|
| MobileNetV2 | 98.78% | 38.45% | 60.33 pp |
| EfficientNetB0 | 99.05% | 41.31% | 57.74 pp |
| ResNet50 | 99.59% | 38.04% | 61.55 pp |

EfficientNetB0 produced the highest external accuracy, but its external performance remained substantially below its internal validation performance.

ResNet50 achieved the highest internal validation accuracy but the lowest external accuracy of the three models.

## Interpretation

Higher internal validation accuracy did not correspond to stronger external generalisation.

Changing the classifier architecture alone did not resolve the cross-dataset performance gap.

This result strengthened the conclusion that internal accuracy by itself was insufficient evidence of robust performance.

## Evidence

Relevant implementation and documentation include:

- configurable classifier architecture support
- `docs/BASELINE_COMPARISON.md`
- Commit `307dd40` — Add configurable classifier baseline architectures
- Commit `b36cbbe` — Document classifier baseline comparison

---

# 4. Quantitative Grad-CAM Faithfulness Evaluation

## Reviewer Recommendation

Strengthen the Grad-CAM analysis beyond qualitative visual examples and, ideally, compare highlighted regions with expert annotations.

## Implementation

Expert lesion annotations were not available, so biological localisation accuracy could not be evaluated directly.

Instead, a quantitative perturbation experiment was implemented to test **model faithfulness**.

For each selected image:

1. The model's original predicted-class confidence was recorded.
2. Grad-CAM was generated for the predicted class.
3. The highest-activation 20% of Grad-CAM pixels were masked.
4. Prediction confidence was recalculated.
5. Spatially coherent random regions covering approximately the same fraction of the image were masked as a control.
6. Five random-region repetitions were performed per image.
7. Grad-CAM and random-region confidence reductions were compared.

The experiment used:

- 60 validation images
- 20 images per class
- 20% masking fraction
- 5 random-region repetitions
- deterministic seed of 42

## Results

| Metric | Result |
|---|---:|
| Mean Grad-CAM confidence drop | 0.2091 |
| Mean random-region confidence drop | 0.0386 |
| Mean Grad-CAM advantage | +0.1705 |
| Grad-CAM drop greater than random | 68.3% |

Class-level results were:

| Class | Grad-CAM Drop | Random Drop | Advantage | Grad-CAM > Random |
|---|---:|---:|---:|---:|
| Brown rust | 0.1847 | 0.0766 | +0.1080 | 70% |
| Healthy | -0.0079 | 0.0075 | -0.0155 | 35% |
| Yellow rust | 0.4506 | 0.0316 | +0.4190 | 100% |

## Interpretation

Under this perturbation protocol, Grad-CAM-highlighted regions were more influential to model confidence than randomly positioned spatial regions overall.

The result was strongest for Yellow rust and positive but more variable for Brown rust.

Healthy images did not demonstrate the same pattern.

This provides quantitative evidence about model faithfulness, but it does **not** demonstrate that Grad-CAM regions correspond to biologically verified wheat-rust lesions.

Expert spatial annotations remain a potential future extension.

## Evidence

Relevant implementation and documentation include:

- `src/evaluate_gradcam_faithfulness.py`
- `docs/GRADCAM_FAITHFULNESS.md`
- Commit `ad875cc` — Add quantitative Grad-CAM faithfulness evaluation

---

# 5. Failure Analysis and Research Direction

## Reviewer Recommendation

Analyse failure cases in more detail and develop a research question that goes beyond maximising validation accuracy.

## Implementation

The original internal validation errors were extracted and analysed rather than reporting only aggregate accuracy.

The MobileNetV2 classifier correctly classified:

**728 of 737 internal validation images**

leaving:

**9 misclassified images**

The errors included:

- 6 Brown rust errors
- 1 Healthy image classified as Brown rust
- 2 Yellow rust images classified as Brown rust

Grad-CAM visualisations were generated for representative failure cases to investigate which image regions influenced incorrect predictions.

External validation then provided a substantially larger set of cross-dataset failures, including high-confidence incorrect predictions.

## Research Question

The project was reframed around the following question:

> **How well do deep-learning models for wheat-rust classification generalise beyond their training distribution, and can explainability and failure-case analysis reveal the visual factors underlying model errors?**

This shifts the focus from:

> How high can validation accuracy become?

toward:

> Does the model remain reliable when the image distribution changes, and what can its failures reveal about the learned decision process?

## Evidence

Relevant evidence includes:

- `docs/FAILURE_ANALYSIS.md`
- `docs/EXTERNAL_VALIDATION.md`
- `docs/BASELINE_COMPARISON.md`
- `docs/GRADCAM_FAITHFULNESS.md`
- generated Grad-CAM failure-case analysis
- Commit `5c72136` — Add failure analysis and leakage checks

---

# Summary of Review Implementation

| External Review Area | Action Taken | Status |
|---|---|---|
| Independent external validation | 489-image external evaluation | Implemented |
| Data leakage investigation | SHA-256 + perceptual similarity analysis | Implemented |
| Stronger classifier baselines | MobileNetV2 vs EfficientNetB0 vs ResNet50 | Implemented |
| Quantitative explainability | Grad-CAM perturbation faithfulness experiment | Implemented |
| Expert lesion annotation | No verified annotations available | Future work |
| Failure-case analysis | Internal and external errors investigated | Implemented |
| Stronger research question | Reframed around generalisation, robustness and explainability | Implemented |

---

# Overall Outcome

The external technical feedback materially changed the direction of the project.

The original project demonstrated strong performance on its internal validation set, but the subsequent investigation showed that this result alone did not establish robust generalisation.

External validation revealed a substantial performance reduction across a different image source. Additional classifier architectures achieved similarly high internal accuracy without resolving the external performance gap.

Leakage analysis indicated that the observed internal result was not materially explained by the single detected near-duplicate validation image.

Failure analysis exposed specific error patterns and high-confidence incorrect predictions, while quantitative Grad-CAM perturbation testing provided initial evidence about which regions influence model decisions and highlighted important class-level differences.

The resulting project therefore documents not only successful model performance, but also:

- generalisation limitations;
- reproducible external evaluation;
- leakage investigation;
- architecture comparison;
- failure analysis;
- quantitative explainability;
- negative and mixed findings;
- methodological limitations; and
- future research directions.

This implementation record is intended to provide a transparent link between external technical review and subsequent project development.