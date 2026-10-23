# Wheat Rust AI: Source-exclusion experiment addendum
**Devanandha Vellaramkuzhiyil Shaji | 23 October 2026**

## Objective and design
Test whether replacing a portion of the original training data with images from one additional source improves performance on another source absent from training. This exploratory extension comprises nine models: original-only baseline, original + IARI, and original + Ethiopia, each using seeds 42, 43 and 44.

Every cohort contains 450 images, balanced at 150 per class (Brown rust, Healthy, Yellow rust). In each added-source cohort, supported classes contain 112 original images and 38 added-source images; the class absent from that source contains 150 original images. IARI supplies Brown rust and Healthy; Ethiopia supplies Healthy and Yellow rust. Thus, “25% mixture” is a nominal per-supported-class setting, not 25% of the entire cohort: 76/450 images (16.89%) come from the added source.

All runs used the same frozen ImageNet MobileNetV2 recipe, augmentation, maximum 10 epochs, batch size 8, and validation-loss checkpoint selection on 737 internal images. Visible run metadata match on runner SHA-256, TensorFlow 2.18.1 and NumPy 2.0.2. JIC (489 images) was evaluated after checkpoint selection. Protected evaluations used IARI test (55 Brown rust, 74 Healthy) and Ethiopia holdout (20 Healthy, 41 Yellow rust).

## Results
Accuracy (%) by seed, with arithmetic means. Seeds vary both cohort sampling and model training; the variation cannot be attributed solely to model initialisation.

| Training condition | IARI: seeds 42 / 43 / 44 | IARI mean | Ethiopia: seeds 42 / 43 / 44 | Ethiopia mean |
|---|---|---:|---|---:|
| Original only | 42.64 / 42.64 / 42.64 | 42.64 | 6.56 / 1.64 / 1.64 | 3.28 |
| Original + IARI | 90.70 / 75.19 / 96.12 | 87.34 | 4.92 / 11.48 / 4.92 | 7.10 |
| Original + Ethiopia | 54.26 / 2.33 / 42.64 | 33.07 | 90.16 / 88.52 / 85.25 | 87.98 |

For excluded-source transfer, adding IARI increased mean Ethiopia accuracy by 3.83 percentage points, but absolute performance remained poor. Adding Ethiopia decreased mean IARI accuracy by 9.56 points, with a wide seed range. Differences are calculated from unrounded results. These are descriptive comparisons, not statistically established effects.

## Error analysis
The original-only baseline predicted Brown rust for every IARI test image at all three seeds, correctly identifying all 55 Brown rust images but none of the 74 Healthy images. On Ethiopia, baseline Yellow rust recall was zero.

Adding IARI did not recover excluded-source Yellow rust recognition: all 41 Ethiopian Yellow rust images were predicted as Brown rust at every seed. Its small Ethiopia accuracy gain came from Healthy-image recognition. Adding Ethiopia produced unstable transfer to excluded IARI: seed 43 predicted the absent Yellow rust class for 122/129 images.

Target-source representation was associated with stronger holdout results, but these results do not establish plant-, field- or acquisition-session independence.

## Audit and comparability
Preparation screened candidates using SHA-256 and pHash (Hamming distance threshold 5). Added-source manifests explicitly record protection of IARI validation/test and the Ethiopia holdout. The baseline manifest omitted that extra record, so a retrospective audit checked the actual three selected 450-image baseline cohorts and 737 internal validation images against 319 protected images (IARI validation/test plus Ethiopia holdout). It found zero exact-match pairs and zero pHash-only flags. All baseline selected-training hashes matched their saved selection manifests.

The retrospective audit resolves the missing baseline screening documentation for these selected images. It does not prove that the full filtered primary pools are identical, nor establish independent plants or fields. Dataset roots, reported filtered primary counts and training settings match; training-image replacement is an intentional condition difference.

## Interpretation and limitations
The supported finding is that target-source inclusion improves performance on that source's held-out images in this experiment, while adding a different source does not reliably improve excluded-source generalisation. Persistent class collapse and seed variation show that the generalisation problem remains unresolved.

Source membership and class availability are confounded: each added source lacks one disease class, so the study does not isolate a pure source-diversity effect. Holdouts are small, only three seeds were used, the JIC benchmark and source holdouts have already been examined, and label mapping (including IARI diseased to Brown rust) remains an assumption requiring authoritative provenance. pHash screening cannot rule out shared plants, fields or sessions. Present-class macro-F1 is reported by the evaluator and must be labelled as such; accuracy and full three-class confusion matrices are retained to show absent-class predictions.

These findings should be reported as an exploratory diagnostic extension, without claiming a solution to unseen-domain generalisation. Any next intervention should use a documented development protocol and a genuinely untouched final evaluation where feasible.

## Evidence records
- Baseline: `experiments/source_holdout_baseline_20261021_162210`
- New cohorts/models: `experiments/source_excluded_20261022_153351`
- Baseline evaluation: `experiments/source_holdout_baseline_eval_20261021_171534/all_holdout_results.csv`
- New evaluation: `experiments/source_excluded_holdout_eval_20261022_170954/all_holdout_results.csv`
- Retrospective baseline audit: `experiments/baseline_selected_audit_20261023_075815`

Prepared from supplied console outputs, saved-result CSV excerpts, confusion matrices, source code and metadata. Raw image datasets and model files were not independently accessed by the report authoring assistant.
