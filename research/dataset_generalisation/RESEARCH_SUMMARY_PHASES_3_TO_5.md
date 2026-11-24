# Wheat Rust AI: Cross-Dataset Generalisation and Disease-Class Confusion

**Technical Research Summary — Phases 3–5**

**Researcher:** Devanandha V. S.
**Date:** 24 November 2026
**Repository:** https://github.com/devanandha/wheat-rust-ai-detection
**Research branch:** `research/phase3-source-diversity`

## Executive Summary

This exploratory research investigates how training-source diversity
affects the external generalisation of deep-learning classifiers
for wheat rust recognition.

Eighteen MobileNetV2 classifiers were trained using controlled
variations in training-set size, source composition and random seed.

The models were evaluated on two previously examined or screened
external benchmarks, followed by targeted cross-seed Grad-CAM
analysis of recurring classification errors.

The principal finding is that increasing training-source diversity
was associated with improved aggregate accuracy on both benchmarks,
but did not consistently improve individual disease-class recall.

Substantial differences in Brown Rust and Yellow Rust predictions
were observed between the evaluated datasets.

These findings demonstrate why internal validation accuracy
and aggregate external accuracy alone are insufficient
to establish reliable disease-specific generalisation.

## 1. Research Question

How does increasing training-source diversity affect the
external generalisation and class-specific error patterns
of MobileNetV2 wheat rust classifiers?

Three classes were examined:

- Brown Rust
- Healthy
- Yellow Rust

The study prioritises the analysis of generalisation failures
over maximising internal validation accuracy.

## 2. Phase 3 — Controlled Source-Diversity Experiment

### Experimental Design

- Architecture: MobileNetV2 with frozen pretrained backbone
- Training images per class: 150 or 300
- Additional-source mixing: 0%, 25% or 50%
- Random seeds: 42, 43 and 44
- Total trained models: 18
- External benchmark: JIC, 489 images

Additional-source images included IARI and Ethiopian datasets.

Source identity was partially associated with disease class,
creating a potential source-class confounding problem.

### JIC External Results

For 300 images per class:

| Source mixing | Mean accuracy | Brown recall | Healthy recall | Yellow recall |
|---|---:|---:|---:|---:|
| 0% | 36.54% | 54.69% | 81.97% | 3.63% |
| 25% | 51.53% | 26.04% | 88.25% | 46.44% |
| 50% | 53.24% | 14.84% | 93.17% | 53.42% |

Values are means across three random seeds.

Increasing source mixing from 0% to 50% was associated
with a mean accuracy increase of approximately 16.70
percentage points.

However, Yellow Rust recall increased by 49.79 percentage
points while Brown Rust recall decreased by 39.85 points.

This demonstrates that the aggregate improvement concealed
a substantial class-specific trade-off.

### Dataset Integrity Audit

The subsequent audit verified:

- Eligibility manifest hash integrity
- All 18 historical cohort manifests
- All 18 cohort membership and composition checks
- 12,150 staged training-image SHA-256 checks
- Recorded image eligibility and protected-set screening

The 12,150 checks represent image occurrences across
experiments rather than unique images.

The audit supports traceability, but does not prove
plant-level, field-level or complete biological independence.

## 3. Phase 4 — Screened External Dataset Evaluation

The 18 frozen Phase 3 checkpoints were evaluated without
retraining on a separately downloaded Kaggle wheat-leaf
disease test subset.

Initial three-class test images: 239

Screening exclusions:

- 38 exact-overlap exclusions
- 2 additional conservative perceptual-hash exclusions

Final evaluated images: 199

| Class | Images |
|---|---:|
| Brown Rust | 39 |
| Healthy | 77 |
| Yellow Rust | 83 |

### Aggregate Results

| Images/class | Source mixing | Mean accuracy | Mean macro-F1 |
|---|---|---:|---:|
| 150 | 0% | 26.80% | 22.42% |
| 150 | 25% | 37.35% | 34.43% |
| 150 | 50% | 37.36% | 35.22% |
| 300 | 0% | 27.47% | 23.31% |
| 300 | 25% | 46.06% | 43.34% |
| 300 | 50% | 46.74% | 43.48% |

### Dataset-Dependent Class Recall

At 300 images per class and 50% source mixing:

| Disease class | Phase 3 JIC | Phase 4 Kaggle |
|---|---:|---:|
| Brown Rust | 14.84% | 95.73% |
| Healthy | 93.17% | 66.23% |
| Yellow Rust | 53.42% | 5.62% |

The same frozen models exhibited markedly different
class-specific behaviour between the benchmarks.

On the Phase 4 dataset, an average of 77 of 83
Yellow Rust images were classified as Brown Rust.

Aggregate accuracy gains were mainly associated
with improved Healthy recognition.

The original Kaggle dataset contained duplicate images
across its splits. Although the evaluated subset was
screened, collection provenance and label consistency
have not been independently established.

The benchmark is therefore exploratory, not confirmed
independent field validation.

## 4. Phase 5 — Cross-Seed Grad-CAM Investigation

Three purposively selected Yellow Rust images were
investigated using the frozen 300-images-per-class,
50%-mixing models.

The models corresponded to seeds 42, 43 and 44.

Grad-CAM used the final MobileNetV2 `out_relu`
activation with spatial dimensions 7 x 7.

Gradients were calculated from pre-softmax class scores.
Reconstructed predictions were verified against the
original model outputs and saved Phase 4 predictions.

### Selected Examples

**lolr(184).JPG**

All three models classified the image as Brown Rust.
Brown Rust activation overlapped portions of visibly
symptomatic leaf tissue.

**lolr(194).JPG**

All three models again predicted Brown Rust.
Class-specific activation overlapped some pustule-bearing
regions but differed spatially across classes and seeds.

**lolr(200).JPG**

Seed 42 predicted Yellow Rust, while seeds 43 and 44
predicted Brown Rust.

This selected example demonstrated model-instance
sensitivity despite identical architecture and
experimental conditions.

### Interpretation

The visualisations suggest that some misclassifications
occurred even when class-specific activation overlapped
visible disease symptoms.

These observations are consistent with limitations in
transferring discriminative visual features between
Brown Rust and Yellow Rust.

However, Grad-CAM does not identify a causal mechanism,
and these three selected images cannot establish a
dataset-wide explanation.

The source photographs and derived Grad-CAM figures
are retained locally pending verification of image
redistribution rights.

## 5. Principal Findings

1. Increasing source mixing was associated with improved
   aggregate external accuracy on two exploratory benchmarks.

2. Improvements were not uniform across disease classes.

3. Brown Rust and Yellow Rust recognition patterns changed
   substantially between the two external datasets.

4. Some consistently misclassified Yellow Rust examples
   exhibited Grad-CAM activation overlapping symptomatic tissue.

5. Internal validation and aggregate accuracy alone
   did not adequately represent cross-dataset performance.

## 6. Scientific Limitations

- Only three random seeds per training condition
- Partial source-class confounding in training composition
- JIC had previously been evaluated
- Kaggle collection provenance remains incompletely verified
- Potential differences in disease-label consistency
- Unequal class sizes in the external benchmarks
- No verified plant-level or field-level independence
- Three purposively selected Grad-CAM case studies
- Grad-CAM does not establish causal feature reliance
- Public reproduction requires additional local assets
- No established deployment readiness

The observed associations should not be interpreted
as proof that source mixing alone caused the gains.

## 7. Conclusion

Increasing training-source diversity was associated
with improved aggregate accuracy, but individual
disease-class performance varied substantially
between evaluation datasets.

The results demonstrate that average accuracy can
conceal severe, dataset-dependent classification failures.

The study contributes a controlled experimental comparison,
dataset-integrity verification, screened external evaluation
and targeted interpretability investigation.

Its central contribution is transparent documentation
of generalisation limitations, rather than a claim
of producing a reliable field-ready disease detector.

## 8. Questions for Independent Technical Review

Independent experts are invited to assess:

1. Whether the experimental design adequately addresses
   source-class confounding.
2. Which analyses could best explain the reversal in
   Brown Rust and Yellow Rust recognition.
3. Whether disease labels are biologically consistent
   across datasets.
4. How a genuinely independent validation study
   should be designed.
5. What additional evidence would strengthen the
   study for formal scientific publication.

## 9. Supporting Evidence

### Phase 3 — Source-Diversity Experiment

[Phase 3 analysis and findings](phase3/analysis_20261123/PER_CLASS_FINDINGS.md)

[Phase 3 dataset integrity audit](phase3/analysis_20261123/DATA_INTEGRITY_AUDIT.md)

### Phase 4 — Screened External Evaluation

[Phase 4 findings](phase4/PHASE4_FINDINGS.md)

[Phase 4 results and evaluation documentation](phase4/README.md)

### Phase 5 — Grad-CAM Investigation

[Phase 5 cross-seed interpretability findings](phase5/PHASE5_GRADCAM_FINDINGS.md)

---

Research status: Exploratory technical research.
Independent expert review and further external validation
are required before stronger conclusions can be drawn.
