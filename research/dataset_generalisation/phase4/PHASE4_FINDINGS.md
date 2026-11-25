# Wheat Rust AI — Phase 4 Exploratory External Evaluation

Date: 24 November 2026

## Objective

Evaluate all 18 frozen Phase 3 MobileNetV2 checkpoints on a
separately downloaded, screened Kaggle wheat-disease test subset.

## Dataset

Original three-class test images: 239
Exact-overlap exclusions: 38
Additional pHash 6–8 review exclusions: 2
Final screened images: 199

Final class supports:
- Brown_rust: 39
- Healthy: 77
- Yellow_rust: 83

The dataset's original collection provenance is not fully
established. These results are exploratory and are not
presented as independent field validation.

## Mean performance across three seeds

| Images/class | Source mix | Accuracy | Macro-F1 |
|---|---|---|---|
| 150 | 0% | 26.80% | 22.42% |
| 150 | 25% | 37.35% | 34.43% |
| 150 | 50% | 37.36% | 35.22% |
| 300 | 0% | 27.47% | 23.31% |
| 300 | 25% | 46.06% | 43.34% |
| 300 | 50% | 46.74% | 43.48% |

## Phase 4 mean per-class recall

| Images/class | Mix | Brown Rust | Healthy | Yellow Rust |
|---|---|---|---|---|
| 150 | 0% | 99.15% | 6.06% | 12.05% |
| 150 | 25% | 94.87% | 41.56% | 6.43% |
| 150 | 50% | 96.58% | 42.42% | 4.82% |
| 300 | 0% | 98.29% | 12.55% | 8.03% |
| 300 | 25% | 95.73% | 62.77% | 7.23% |
| 300 | 50% | 95.73% | 66.23% | 5.62% |

## Confusion-matrix findings

For 300 images/class at 50% mixing, an average of 77.0
of 83 actual Yellow Rust images were classified as Brown Rust.

Only 4.7 Yellow Rust images were correctly classified on
average across three seeds.

Healthy recall improved substantially with source mixing.

## Comparison with Phase 3

At 300 images/class and 50% mixing:

Phase 3 JIC:
- Brown Rust recall: 14.84%
- Healthy recall: 93.17%
- Yellow Rust recall: 53.42%

Phase 4 screened Kaggle:
- Brown Rust recall: 95.73%
- Healthy recall: 66.23%
- Yellow Rust recall: 5.62%

The same frozen models display substantially different
class-specific performance across these benchmarks.

## Interpretation

Training-source mixing was associated with improved aggregate
accuracy on both evaluated benchmarks.

However, improvements were not consistent across disease
classes or datasets. Phase 4 gains were largely associated
with improved Healthy recall, while Yellow Rust recall
remained extremely low.

These results demonstrate that aggregate accuracy can conceal
substantial dataset-specific classification failures.

## Limitations

- The Phase 4 benchmark is exploratory, not confirmed
  independently collected field data.
- Dataset provenance and class-label consistency require review.
- The Kaggle source contained duplicates between splits.
- The screened set has only 39 Brown Rust images.
- Source-class confounding remains unresolved.
- Three random seeds were evaluated per condition.
- The evaluation results must not be used to retroactively
  tune or select models and then claim independent confirmation.
- No claim of deployment readiness is supported.

## Reproducibility

All 18 canonical Phase 3 model checkpoints were evaluated
without retraining.

The Phase 4 selection manifest, frozen evaluation protocol,
model hashes, individual predictions, confusion matrices,
and classification reports are preserved locally.

No historical Phase 3 results were modified.
