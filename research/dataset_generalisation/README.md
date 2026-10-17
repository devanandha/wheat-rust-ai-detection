# Wheat rust AI - dataset size and source-mixture extension (18 runs)

**Experimental extension completed 17 October 2026.** This work extends the [Wheat Rust AI Detection](https://github.com/devanandha/wheat-rust-ai-detection) repository with dataset-size and source-mixture comparisons.

## Research question

At two fixed class-balanced training sizes (450, 900 images), how does substituting 0%, about 25% or 50% images from additional datasets affect three-class wheat rust classification results? We ran three seeds per condition (42, 43, 44), totalling 18 MobileNetV2 models. Source mixture and training size vary together with per-source class content; reported associations do **not** establish an isolated causal effect of diversity.

## Key results (mean accuracy of three seeds)

| Training images | Nominal added-source share | JIC (489) | IARI test (129) | Ethiopia (61) |
|---:|---:|---:|---:|---:|
| 450 | 0% | 36.61% | 42.64% | 3.28% |
| 450 | 25% | 51.06% | 82.69% | 81.42% |
| 450 | 50% | 47.17% | 95.09% | 90.71% |
| 900 | 0% | 37.76% | 42.64% | 4.92% |
| 900 | 25% | 51.12% | 87.86% | 89.62% |
| 900 | 50% | 50.10% | 95.09% | 96.17% |

The JIC benchmark was **previously evaluated during research development**. It is *not a newly blind external test*. The IARI and Ethiopian test images were reserved, but training for mixed-source models also included other images from the same source datasets. Therefore, strong IARI/Ethiopia results are **within-source, image-held-out results**, not generalisation to entirely unseen acquisition domains. Each holdout contains only two of the three classes. Source and disease labels are partly confounded. Hash-based overlap checks cannot prove plant, field or camera-session independence.

## Included files

- `../../src/`: reproducibility scripts sourced from this study and evaluated on the author's laptop. No original dataset images or model weights included.
- `technical_review.pdf`: detailed technical report with results, figures and constraints.
- `per_run_results_18.csv`: one row per model with JIC, IARI and Ethiopia metrics; JIC values transcribed from 18 PowerShell-printed `run_metadata.json` records, not copied from the physical metadata files.
- `condition_summary.csv`: condition means and **sample standard deviations across three seeds**.
- `sanitized_study_summary.json`: path-free aggregate from the provided manifests.
- `figures/`: summary figures; error bars show seed SD, **not confidence intervals**.
- `dataset_provenance.md`: data rights, mapping and leakage notes.

## Reproducing locally

Use a fresh environment compatible with **Python and TensorFlow 2.18.1** (the tested study runtime). The scripts also use NumPy, scikit-learn, Pillow and ImageHash. See each script's `--help` and the `extension_protocol_original.md` for CLI details. You must obtain the licensed datasets from their original publishers and configure local dataset paths; the manifest provided here deliberately contains no machine-specific paths. The extension scripts are in `../../src/`. To reproduce, obtain the third-party datasets from their original publishers and configure local paths. Retain the historical model checkpoints and experiments separately.

The original `study_manifest.json` and staging logs have absolute local Windows paths and are **not** redistributed in this package; a sanitized aggregate is provided. Exact replay additionally needs the author's private cohort image inventories/checkpoint metadata and the original dataset releases. This package is primarily a transparent methods-and-results release, not a fully self-contained re-execution bundle.

## Release / review status

The 18 JIC observations were reconciled against values printed from local run metadata in PowerShell; the original 18 JSON files have **not** been uploaded independently. The raw 18-run holdout CSV and two preprocessing manifests were received, and key confusion matrices were checked from user-supplied PowerShell output. A reviewer should verify hashes and input provenance independently where possible.

This extension has not yet received an independent scientific assessment. Feedback on earlier work should not be interpreted as an endorsement of the new results.
