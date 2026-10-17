# Protocol: Dataset Size × Source Diversity in Wheat-Rust Classification

**Date:** 16 October 2026
**Purpose:** Investigate the effects of training-set size and source diversity following independent scientific feedback.
**Status:** Archival planning protocol written before the additional experiments; results are documented separately.

## Study question

Under otherwise matched MobileNetV2 training settings, how do the number of training images and the diversity of independent collection sources relate to performance on a fixed external wheat-rust benchmark?

## Historical reference (NOT results from this extension)

- Existing original-source MobileNetV2 internal validation accuracy: **98.78%**, 737 images.
- Existing external evaluation accuracy: **38.45%**, 489 images (JIC Wheat Disease Dataset - Small, DOI 10.5281/zenodo.7307816).
- Existing full-training 2,942-image models and original published manuscript are retained separately.
- Old model results should not be directly attributed to a new controlled ablation: balanced 150/300 images per class means a different training composition and training size.

## Candidate sources and class mapping

| Role | Dataset | Permitted label selection | Notes |
|---|---|---|---|
| Primary train | Original Kaggle sinadunk23 / Behzad Safari Jalal | Brown_rust, Healthy, Yellow_rust | Use original training split, not validation. |
| Additional train A | IARI leaf rust + control (Mendeley DOI: 10.17632/th422bg4yd.1) | Leaf rust -> Brown_rust; healthy control -> Healthy | Confirm labels and exclude N-deficiency photographs; CC BY 4.0. |
| Additional train B | Ethiopia wheat leaf images (Mendeley DOI: 10.17632/wgd66f8n6h.1) | Stripe/Strip Rust -> Yellow_rust; Healthy Wheat Leaf -> Healthy | Confirm labels; exclude Septoria; CC BY 4.0. |
| Internal validation | Original dataset's independent validation folder | Three classes | Used for early stopping and checkpoint selection, not fitting. |
| Fixed external benchmark | JIC Wheat Disease Dataset - Small (DOI: 10.5281/zenodo.7307816) | Brown rust, Healthy, Yellow rust (489 images) | Previously evaluated; not used for training or model selection in this extension. |

**Caution:** Data from different publishers do not guarantee different source photographs. The scripts filter exact and pHash-near matches but cannot establish plant, field, photographer or acquisition-event independence. Label taxonomy requires manual verification, ideally by someone familiar with plant pathology.

## Prespecified design

- **Architecture:** ImageNet-initialised MobileNetV2, frozen backbone, identical augmentation, loss, training length and early stopping across arms.
- **Training sizes:** 150 or 300 images **per class**, totalling 450 or 900 training images respectively.
- **Source-diversity fraction:** 0%, 25% or 50% additional-source images in EACH class; fill the rest from the original training source.
- **Random seeds:** 42, 43 and 44, independently sampled cohorts and repeated training.
- **Comparison:** At a fixed training size, compare 0%, 25% and 50% external-source mixing. At a fixed mix, compare sizes 150 vs 300.
- **Internal evaluation:** Held-out original validation (737 images). This set controls early stopping/checkpoint selection only.
- **External evaluation:** Existing JIC 489-image set. Keep fixed, never use for fitting, early stopping or selecting a configuration.
- **Primary reporting:** external macro-F1 and accuracy, including per-class recall and confusion matrices. Report internal/external accuracy gap as secondary, plus balanced cohort/source counts.
- **Limitations:** source mixtures, class mix and capture differences may be confounded; no biological annotation audit, no proof of field-level independence, external set previously inspected. These are exploratory controlled ablations, not causal identification of generalisation mechanisms.

## Controls and leakage rules

1. Each condition must contain the same class counts and a recorded image manifest, seed and source fraction.
2. Training candidates matching internal validation or external benchmark by SHA-256 or pHash Hamming distance <= 5 are **excluded before sampling** and recorded.
3. Exclude identical cross-source training duplicates. Record every exclusion; inspect image-level near matches and potential subject-level relatedness when provenance permits.
4. Create different output folders per cohort. Retain incomplete runs, failed runs and all reports.
5. Train only on the staged train subset; validation is used for model selection, not gradients; external test is used for final evaluation only.
6. Avoid using the fixed JIC dataset to choose a hyperparameter or preferred experimental arm.
7. Report class-level label uncertainty and any domain conflicts explicitly.

## Analysis and interpretation

- Calculate per-seed accuracy and macro-F1 for internal and external datasets.
- Report mean and standard deviation across seeds for each (training size, mixture) cell; do not call this a 95% CI.
- Where all models evaluate the same JIC images, use paired image-level comparisons and suitable uncertainty estimates in a follow-up analysis. Confidence intervals over images can be over-optimistic if image observations from the same fields/subjects are correlated.
- If diversity improves external scores, state the observed association conditional on this protocol. Do not claim field readiness or biological localisation.
- If diversity fails to improve external scores, report that negative finding. This is still scientifically informative.
- Clearly distinguish this exploratory extension from the submitted journal manuscript, which remains unchanged unless editors request a revision.

## Review package after experiments

Create a short report: 1) question and preregistered protocol, 2) source metadata/licences, 3) cohort composition and checks, 4) all results including statistical limitations, 5) improvements vs historical design, 6) code commit / environment / hashes, 7) questions for independent reviewers about whether the evidence supports the proposed contribution.

Do not solicit a favourable conclusion; invite an independent judgement of strengths, weaknesses, publication potential and significance. Scientific reviewers should be able to form their own independent assessment of the work.
