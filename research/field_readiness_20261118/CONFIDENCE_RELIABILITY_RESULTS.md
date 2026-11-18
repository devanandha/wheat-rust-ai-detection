# Frozen-model confidence reliability audit — exploratory results

**Date:** 18 November 2026  
**Status:** Completed by local user; `reliability_report.json` and six accompanying per-image, confidence-bin, and referral-tradeoff CSVs were uploaded and analysed. Results have been derived from those outputs, **not from newly collected field images**.  
**Experiment protocol:** [CONFIDENCE_RELIABILITY_AUDIT.md](CONFIDENCE_RELIABILITY_AUDIT.md)  
**Code:** [`tools/audit_prediction_reliability.py`](../../tools/audit_prediction_reliability.py)  
**Unit tests:** Four synthetic tests passed on the user's Windows Python 3.11.2.  
**Frozen model SHA-256:** `2c18f6061bd3f26427bdb05d21ea1b3bd85fc82f92596194c3e36f6b2c1513e4`  
**Runtime:** TensorFlow 2.18.1. No weights, photos, data partitions, probability scaling, thresholds or classifier outputs were modified in this study.

## Executive finding

The **maximum softmax score is not a reliable safety gate on the previously examined external JIC data**. At score ≥0.90, **135/233 accepted JIC predictions (57.94%) were incorrect**. At the stricter ≥0.99 cutoff, **55/83 accepted predictions (66.27%) were incorrect**, including 30 leaves with true disease labels predicted **Healthy**. Therefore do not use a simple 0.9 or 0.99 rule to declare crop health or to enable autonomous field advice.

## Source-aware classification reproduction

| Metric | Original validation | External JIC |
|---|---:|---:|
| Images | 737 | 489 |
| Correct | 728 (98.78%) | 188 (38.45%) |
| Incorrect | 9 | 301 |
| Balanced accuracy | 98.71% | 48.41% |
| Macro F1 | 98.74% | 35.72% |
| Actual Brown Rust predicted Healthy | 3 | 53 |
| Actual Yellow Rust predicted Healthy | 0 | 98 |
| **All diseased predicted Healthy** | **3** | **151** |
| Total wrong with score ≥0.90 | 1 | 135 |
| Total wrong with score ≥0.99 | 1 | 55 |

The JIC subset has **367 actually diseased images** (Brown Rust 128, Yellow Rust 239), of which **151/367 = 41.14%** were assigned Healthy. Of the 250 JIC samples predicted Healthy at all confidence levels, **151/250 = 60.4% were actually diseased**. These are dataset-specific closed-set outcomes, not population risk estimates or an assessment of unknown diseases.

The JIC confusion matrix (true rows, predicted columns Brown Rust / Healthy / Yellow Rust):

```text
                 predicted Brown   predicted Healthy   predicted Yellow
True Brown             74                 53                  1
True Healthy           23                 99                  0
True Yellow           126                 98                 15
```

## Calibration and highly confident mistakes

| Measurement | Original validation | JIC |
|---|---:|---:|
| Mean maximum score | 0.9850 | 0.8292 |
| 10-bin max-score expected calibration error (ECE) | 0.00986 | 0.44474 |
| Multiclass Brier score | 0.02042 | 1.00326 |
| Negative log likelihood | 0.04099 | 2.57432 |
| 0.90–1.00 score-bin sample count | 706 | 233 |
| Mean score within 0.90–1.00 bin | 0.9946 | 0.9672 |
| Observed accuracy within 0.90–1.00 bin | 99.86% | **42.06%** |

The JIC high-score bin displays an extreme gap between the displayed model scores (96.72% mean) and correctness (42.06%). This is **a descriptive model-calibration result under one observed dataset shift**, not evidence that confidence can be directly interpreted as a diagnostic probability or that ECE values transfer to field populations. ECE depends on binning and the sampled class/source composition.

## Fixed-threshold referral analysis on JIC

Threshold was predetermined before rerunning the model, and **no threshold was tuned on JIC**. “Accepted” means maximum score ≥ threshold; all others are **referred to expert review**. Here 'accepted' is a descriptive counting label, **not an approval for field use**.

| Minimum max score | Accepted | Sent to review | Error among accepted | Diseased → Healthy among accepted | Yellow Rust correct / accepted Yellow |
|---|---:|---:|---:|---:|---:|
| 0.00 | 489 | 0 | 301/489 (61.55%) | 151 | 15/239 |
| 0.50 | 468 | 21 | 288/468 (61.54%) | 144 | 12/226 |
| 0.70 | 374 | 115 | 226/374 (60.43%) | 117 | 8/179 |
| 0.90 | 233 | 256 | 135/233 (57.94%) | **71** | 1/111 |
| 0.95 | 160 | 329 | 93/160 (58.13%) | **48** | 1/78 |
| 0.99 | 83 | 406 | 55/83 (66.27%) | **30** | **0/49** |

At threshold 0.90, the 71 high-score diseased-as-Healthy errors consist of **22 Brown Rust** and **49 Yellow Rust**. Even among the **122 images predicted Healthy and accepted at ≥0.90, 71 (58.20%) are actually diseased**. At 0.99, 30 of the 47 accepted Healthy predictions were diseased (63.83%).

Crucial caveat: the column “Yellow Rust correct / accepted Yellow” measures **recall conditional on acceptance**, not recall over *all* originally presented Yellow Rust images. The latter would be lower still at any threshold that defers cases.

On original validation, by contrast, threshold ≥0.90 retained **706/737**, made **1/706** errors (0.14%), and assigned one diseased image as Healthy. This contrast illustrates why a threshold appearing effective on the original source cannot simply be transported to JIC or field images.

## What this experiment does and does not show

**Observed:** The previously saved model reproduces the historical evaluation exactly. It makes many highly confident errors on JIC, notably true disease images classified Healthy. Simply increasing a maximum-score acceptance threshold is **not an adequate safety gate** for this dataset; the wrong-prediction rate even increases between thresholds 0.9 and 0.99.

**Not established:** No new farmer or agricultural-field images were gathered; no external site/season independent labels were evaluated. The model is **closed-set** and cannot detect all alternative wheat diseases, mixed infections, non-wheat objects or normal nutrient-related changes. None of these thresholds has been prospectively validated; none should be deployed. The dataset was reused many times and must **not** be described as a fresh blind test or improved performance after model adaptation.

## Next step for real-world research

1. **Preserve all negative findings** and reject naive “≥90% confidence = safe diagnosis” behaviour in the prototype. Keep demo output non-diagnostic, and require expert review in Research Mode.
2. **Stop optimising thresholds against JIC.** The existing audit supports a research hypothesis (high-confidence error under acquisition shift), not selection of a deployment cutoff.
3. **Prospectively collect a small pilot** with a real wheat-pathology collaborator under appropriate image/data permissions, expert-reviewed labels, and site/field/plant/session/device identifiers; include out-of-scope/mixed cases. Pilot-scale counts are for feasibility, not proof of safe accuracy.
4. **Prepare a separate development/calibration partition**, with group-disjoint data and a locked final external site/season evaluation. Compare separately calibrated scores, quality/review policies, and error/coverage on untouched field images. Maintain all per-class metrics and confidence-bin audits.
5. Before any field-facing advice, define and satisfy **independent readiness and risk gates** agreed with agronomic reviewers; evaluate false Healthy reassurance, Yellow Rust misses, abstention burden, unknown conditions and operational use on target phones.
6. A distinctive publishable contribution would require **new, reproducible, independent, causally informative findings** compared with prior selective-prediction/domain-shift research. The current error report alone does not establish novelty.

No raw image data or per-image filenames from the supplied CSV files were committed as part of this results summary.
