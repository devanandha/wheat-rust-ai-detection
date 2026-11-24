# Wheat Rust AI — Phase 3: Training Data Quantity and Source Diversity

## Overview

This study investigates whether increasing training dataset size or introducing additional-source images improves cross-dataset generalisation in wheat-rust classification.

The study extends earlier research demonstrating strong internal validation performance but weak external performance.

## Experimental design

- Architecture: MobileNetV2 with frozen ImageNet-pretrained backbone
- Classes: Brown_rust, Healthy, Yellow_rust
- Training images per class: 150 and 300
- Additional-source mixing: 0%, 25%, 50%
- Random seeds: 42, 43, 44
- Maximum epochs: 10
- Batch size: 8
- Internal validation: 737 images
- Previously evaluated JIC external benchmark: 489 images

Total: 18 controlled training experiments.

## Results

Mean performance across three seeds per condition:

| Images/class | Source mix | Internal accuracy | External accuracy | External macro F1 |
|---|---|---|---|---|
| 150 | 0% | 97.65% | 38.31% | 35.07% |
| 150 | 25% | 96.38% | 50.65% | 48.24% |
| 150 | 50% | 95.52% | 51.12% | 47.30% |
| 300 | 0% | 97.96% | 36.54% | 33.08% |
| 300 | 25% | 97.11% | 51.53% | 48.67% |
| 300 | 50% | 96.61% | 53.24% | 47.53% |

## Paired-seed comparisons

Mean improvements in external accuracy relative to the corresponding 0%-mixing condition:

| Images/class | Source mix | Improvement |
|---|---|---|
| 150 | 25% | +12.34 percentage points |
| 150 | 50% | +12.82 percentage points |
| 300 | 25% | +15.00 percentage points |
| 300 | 50% | +16.70 percentage points |

All 12 matched-seed comparisons showed improvements on the previously evaluated JIC benchmark.

## Interpretation

Increasing training-source diversity was associated with improved external benchmark performance under the tested conditions.

Doubling the quantity of same-source training data did not produce a corresponding improvement in mean external accuracy.

The findings suggest that training-data composition is important when investigating cross-dataset generalisation.

## Limitations

- Three random seeds were evaluated per condition.
- The JIC benchmark had previously been evaluated and must not be represented as an untouched independent test.
- The observed improvements are exploratory and require independent confirmation.
- High internal accuracy does not establish field deployment readiness.
- Dataset provenance, licensing, overlap screening, and class-label consistency require careful documentation.
- The original experimental manifest contains machine-specific absolute paths.

## Reproducibility

The original study manifest is preserved unchanged because its SHA-256 hash is recorded in the run metadata.

The original training runner is located at:

`research/source_exclusion_20261023/src/run_generalisation_study.py`

To reproduce the experiments, researchers must obtain appropriately licensed datasets and generate a local copy of the manifest with paths adapted to their environment.

Do not modify the archived original manifest or historical results when preparing a local reproduction.

The repository includes:

- Original Phase 3 preparation scripts
- Study manifest
- Training histories
- Experiment metadata and file hashes
- Classification reports and confusion matrices
- Individual-run and six-condition statistical summaries
- Paired-seed comparisons

Original training images and model checkpoints are not distributed in this directory.

## Research status

Controlled training and statistical aggregation completed in November 2026.

Independent confirmation and further error analysis remain pending.

These Phase 3 experiments are an extension of the research and should not be confused with results already included in an earlier journal submission.
