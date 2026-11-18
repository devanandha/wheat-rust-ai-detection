# Frozen MobileNetV2: paired resolution-perturbation results

**Date:** 18 November 2026  
**Status:** Completed local inference experiment; the user supplied `paired_sensitivity_summary.json`, `original_val_paired_predictions.csv` and `jic_paired_predictions.csv`. **Exploratory research result, not a blind independent test or evidence of a causal shortcut.**

## Objective and method

Following the [source-image metadata audit](EXPLORATORY_FINDINGS.md), evaluate whether the **same frozen, saved classifier** is sensitive to an artificial reduction in input spatial detail, holding the image and class label fixed. Inference was performed with [`tools/probe_resolution_sensitivity.py`](../../tools/probe_resolution_sensitivity.py), using the repository's `src.data.load_split` and preprocessing already embedded within the saved MobileNetV2 classifier. No extra normalization, retraining or image modifications on disk.

- TensorFlow: **2.18.1**.
- Saved model SHA-256: `2c18f6061bd3f26427bdb05d21ea1b3bd85fc82f92596194c3e36f6b2c1513e4`.
- Baseline: standard 224 × 224 RGB input.
- Perturbed: same 224 × 224 input, bilinear resize to **112 × 112** then bilinear resize back to **224 × 224**. This is an artificial loss of spatial detail; **not** a recreation of native low-resolution acquisition.
- Class order: `Brown_rust`, `Healthy`, `Yellow_rust`.
- Original validation **n = 737**, previously inspected external JIC **n = 489**.
- Baseline reproduced both historical counts *exactly*: original 728/737, JIC 188/489.

## Paired overall outcome

| Measure | Original validation (n = 737) | JIC (n = 489) |
|---|---:|---:|
| Baseline correct | 728 (98.78%) | 188 (38.45%) |
| Perturbed correct | 707 (95.93%) | 233 (47.65%) |
| Accuracy change | **−2.85 pp** | **+9.20 pp** |
| Any predicted-class change | 29 (3.93%) | 154 (31.49%) |
| Incorrect → correct | 4 | 83 |
| Correct → incorrect | 25 | 38 |
| Mean change in assigned true-class probability | −0.0481 | +0.0806 |

Predictions changed about **8 times as frequently** on JIC as on original validation. This is a within-image paired comparison with the same model. Because accuracy and class composition differ substantially across sources, the difference in change rates should not be interpreted as a source-level causal estimate.

## Which classes changed?

Numbers are the *number correctly classified*, out of the fixed class support in each source.

| Source | True class | Baseline correct | Perturbed correct | Baseline recall | Perturbed recall |
|---|---|---:|---:|---:|---:|
| Original validation | Brown Rust | 220 / 226 | 220 / 226 | 97.35% | 97.35% |
| Original validation | Healthy | 278 / 279 | 255 / 279 | 99.64% | 91.40% |
| Original validation | Yellow Rust | 230 / 232 | 232 / 232 | 99.14% | 100.00% |
| JIC | Brown Rust | 74 / 128 | 60 / 128 | 57.81% | 46.88% |
| JIC | Healthy | 99 / 122 | 98 / 122 | 81.15% | 80.33% |
| JIC | **Yellow Rust** | **15 / 239** | **75 / 239** | **6.28%** | **31.38%** |

- On **JIC Yellow Rust**, 85 predictions switched class; 63 formerly incorrect cases became correct and 3 formerly correct cases became incorrect (net **+60** correct). Among 63 new correct Yellow Rust predictions, **36** changed from Brown Rust and **27** from Healthy.
- On **JIC Brown Rust**, 38 predictions switched; 6 errors were corrected and 20 originally correct became incorrect (net **−14** correct).
- On **JIC Healthy**, 31 switched; 14 were corrected and 15 became incorrect (net **−1**).
- On **original Healthy**, 23 correctly classified images became incorrect: 13 switched to Brown Rust and 10 to Yellow Rust. This accounted for almost all of the internal accuracy drop.

### JIC confusion matrices

Rows are true labels and columns are predicted labels: Brown Rust, Healthy, Yellow Rust.

**Before perturbation:**

```text
                 Pred Brown  Pred Healthy  Pred Yellow
True Brown             74           53            1
True Healthy           23           99            0
True Yellow           126           98           15
```

**After perturbation:**

```text
                 Pred Brown  Pred Healthy  Pred Yellow
True Brown             60           51           17
True Healthy           14           98           10
True Yellow            93           71           75
```

The change broadly increases the model's tendency to predict Yellow Rust, helping some JIC Yellow Rust samples but sacrificing some Brown/Healthy samples. **JIC Yellow Rust recall remains poor (31.38%; 164/239 missed)** despite the improvement.

## Cross-check against the acquisition-metadata audit

The earlier metadata audit of 4,168 images found that **original training** source-image sizes strongly covary with class: 72.6% of Brown Rust, 1.2% of Healthy and 94.9% of Yellow Rust had at least one dimension below 224 px. None of the JIC samples did. Training and validation show similar size distributions, and JIC images are less colour-saturated by disease class.

The per-image prediction and audit-metric CSV files join exactly on `(source, image_id)` for **737/737 original validation** and **489/489 JIC** records, with consistent labels. These sources are eligible for future *pre-specified* paired analysis.

However, downsampling an image that was already resized is an **artificial blur/detail perturbation**. It neither isolates original resolution metadata nor reproduces camera/sampling conditions. The cross-source performance difference could also involve plant symptoms, background, cultivar, cropping, label conventions, acquisition lighting, saturation, image quality or curation.

## Interpretation and research-integrity boundaries

1. The historic JIC accuracy **did not improve through retraining**. A single, artificial inference-time perturbation changed predictions and raised apparent JIC accuracy from 38.45% to 47.65%; internal accuracy fell.
2. These results provide **direct evidence of the frozen model's sensitivity to loss of image detail**, with effects strongly dependent on source and true class.
3. They **do not prove** that the trained classifier learned a shortcut from native source-image resolution, or that resolution is the cause of its original external failure.
4. This JIC subset was **already inspected in prior experiments**. Treat the reported 112 px perturbation and all subsequent comparisons as **exploratory**; do **not** present 47.65% as a newly validated generalisation score.
5. Exact-file hash checks and reproduced baseline scores are useful consistency checks, but do not certify independent plants, sessions or fields.
6. Do not use repeated JIC probing to select a supposedly optimal downsampling level or tune a new model without a genuinely independent final holdout.

## Recommended next controlled experiment

- **Prespecify competing mechanisms:** (a) interpolation/spatial-detail artefacts, (b) image colour/saturation and (c) source/context.
- **Separate inference sensitivity from learning shortcuts.** Assess multiple perturbations on held-out original data and the already studied JIC data strictly as exploratory, while keeping interventions individually controlled and verifying baseline on each run; avoid tuning to JIC.
- To test **learning shortcuts**, train controlled replicates (fixed architecture, seeds, data budget, split and class composition) with a pre-specified image standardization/interpolation policy and compare to an otherwise unchanged baseline.
- Acquire a truly **untouched** external evaluation collection with matching labels and provenance, ideally independent by plant/field/session; lock the hypothesis and protocol before opening it.
- Report **all three class recalls, macro metrics and confusion matrices**, not overall accuracy alone. Preserve adverse effects (e.g. Brown Rust degradation and original Healthy losses).

**Evidence retained locally:** The uploaded JSON includes model hash, versions and all matrices; prediction CSVs include each image's class and before/after probabilities. These output files are not raw training images and have not been published to GitHub in this update.
