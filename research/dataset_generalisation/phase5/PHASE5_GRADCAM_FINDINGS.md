# Phase 5 — Cross-Seed Grad-CAM Analysis

Date: 24 November 2026

## Objective

Investigate persistent Yellow Rust to Brown Rust
classification errors observed during the Phase 4
screened exploratory external evaluation.

## Experimental design

Three frozen MobileNetV2 checkpoints from the
n300_mix50 condition were examined:

- n300_mix50_seed42
- n300_mix50_seed43
- n300_mix50_seed44

No model retraining, fine-tuning, checkpoint modification,
or retrospective changes to the Phase 4 evaluation
were performed.

Three Yellow Rust images were purposively selected:

- lolr(184).JPG: misclassified across all three seeds
- lolr(194).JPG: misclassified across all three seeds
- lolr(200).JPG: seed-dependent classification

The images are case studies, not a representative
random sample.

## Grad-CAM methodology

Grad-CAM was calculated from the MobileNetV2 backbone's
final out_relu activation, with feature-map dimensions
7 x 7 x 1280.

Class-specific gradients were calculated using
pre-softmax classifier scores (logits).

The original prediction outputs were verified against
the reconstructed logit-based model and the saved
Phase 4 prediction records.

Grad-CAM maps were independently normalised and
resized for visual inspection.

Independent normalisation means heatmap intensities
cannot be directly compared as absolute evidence
strength between classes or models.

## Prediction results

| Image | Seed | P(Brown) | P(Yellow) | Prediction |
|---|---|---:|---:|---|
| lolr(184) | 42 | 0.995 | 0.004 | Brown |
| lolr(184) | 43 | 0.929 | 0.065 | Brown |
| lolr(184) | 44 | 0.982 | 0.008 | Brown |
| lolr(194) | 42 | 0.976 | 0.024 | Brown |
| lolr(194) | 43 | 0.951 | 0.048 | Brown |
| lolr(194) | 44 | 0.971 | 0.029 | Brown |
| lolr(200) | 42 | 0.466 | 0.534 | Yellow |
| lolr(200) | 43 | 0.580 | 0.420 | Brown |
| lolr(200) | 44 | 0.720 | 0.277 | Brown |

Probabilities are rounded.

## Findings

### Case 1: lolr(184)

All three models consistently classified the image as
Brown Rust.

Brown Rust Grad-CAM activation overlapped portions
of the visibly pustule-bearing leaf tissue.

Yellow Rust maps showed different and more localised
activation patterns, varying across seeds.

### Case 2: lolr(194)

All three models again predicted Brown Rust.

Brown Rust Grad-CAM highlighted a substantial portion
of the symptomatic leaf area.

Yellow Rust activation also overlapped some symptomatic
regions but differed spatially from the Brown Rust maps.

### Case 3: lolr(200)

Seed 42 predicted Yellow Rust, while seeds 43 and 44
predicted Brown Rust.

Both Brown Rust and Yellow Rust Grad-CAM maps overlapped
visible lesion-bearing regions.

This example demonstrates model-instance sensitivity
for a selected borderline classification.

## Comparison with external evaluation

At 300 images per class and 50% source mixing:

- Phase 3 JIC mean Yellow Rust recall: 53.42%
- Phase 4 Kaggle mean Yellow Rust recall: 5.62%

In Phase 4, an average of 77 of 83 Yellow Rust images
were classified as Brown Rust under this condition.

The discrepancy indicates substantial dataset-dependent
variation in class-specific performance.

## Interpretation

The visualisations suggest that some incorrect predictions
were associated with activation over symptomatic tissue,
rather than an obvious failure to localise the leaf.

This is consistent with difficulties transferring
discriminative features between visually similar
rust disease classes.

However, the analysis does not establish the causal
mechanism of these errors.

Possible contributing factors include image acquisition
differences, source-specific visual characteristics,
disease-label inconsistencies, preprocessing effects
and unresolved source-class confounding.

## Limitations

- Three purposively selected images only
- Coarse 7 x 7 spatial feature representation
- Grad-CAM is not a causal explanation
- Heatmaps independently normalised
- No lesion-level expert annotations
- Full Kaggle image provenance remains unverified
- Phase 4 is an exploratory screened benchmark
- No evidence of deployment readiness

## Figures

### Figure 1 — lolr(184)

Cross-seed logit Grad-CAM comparison for lolr(184). The figure is retained locally pending confirmation of source-image redistribution rights.

### Figure 2 — lolr(194)

Cross-seed logit Grad-CAM comparison for lolr(194). The figure is retained locally pending confirmation of source-image redistribution rights.

### Figure 3 — lolr(200)

Cross-seed logit Grad-CAM comparison for lolr(200). The figure is retained locally pending confirmation of source-image redistribution rights.

## Conclusion

Across three selected Yellow Rust images, cross-seed
logit-based Grad-CAM revealed two consistently
misclassified examples and one seed-dependent example.

Class-specific activation often overlapped visibly
symptomatic tissue, suggesting that classification
errors cannot be explained solely by failure to
highlight infected regions.

These observations complement the Phase 3 and Phase 4
results but remain exploratory.

A larger, prespecified evaluation with independent
disease-label verification and lesion annotations
is required before stronger conclusions are justified.


## Dataset attribution and figure availability

Source dataset: Wheat Leaf Disease Dataset for DL Classification,
published on Kaggle by yilikalfikru.

https://www.kaggle.com/datasets/yilikalfikru/wheat-leaf-disease-dataset-for-dl-classification

The Grad-CAM figures incorporate source dataset photographs.
They are retained locally and are not redistributed in this
public report pending verification of the applicable image rights.

The selected examples were chosen for exploratory error analysis,
not as an unbiased evaluation sample.
