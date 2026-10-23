# Source-exclusion experiment: supporting evidence

23 October 2026 · Devanandha Vellaramkuzhiyil Shaji

[Research addendum](../../docs/WHEAT_RUST_SOURCE_EXCLUSION_20261023.md)

This package preserves the supplied text evidence for nine runs: three original-only baselines and six added-source conditions. Each uses 450 training images, three classes, seeds 42/43/44, MobileNetV2, maximum 10 epochs and batch size 8.

## Findings

| Training data | IARI test mean accuracy | Ethiopia holdout mean accuracy |
|---|---:|---:|
| Original only | 42.64% | 3.28% |
| Original + IARI | 87.34% | 7.10% |
| Original + Ethiopia | 33.07% | 87.98% |

Means are descriptive across three seeds. Including a target source improved that source's held-out performance, but adding a different source did not reliably improve excluded-source generalisation. Source/class confounding, small holdouts and missing plant/field identifiers limit interpretation. Nominal 25% mixing applies only to supported classes: added-source images constitute 76/450 (16.89%) of a complete cohort.

## Contents and verification

- `experiments/`: both consolidated result CSVs; per-image predictions; classification reports; full confusion matrices; selected-image manifests; run settings and histories; retrospective audit summary and empty flag table.
- `src/`: byte-preserved snapshots of four supplied preparation, training, evaluation and audit tools, isolated here to avoid changing the main application's code.
- `prepare_source_excluded.py`: parameterised reconstruction of the historical inline PowerShell/Python preparation procedure. It was not the file executed for the archived results.
- `FILE_INTEGRITY.json`: SHA-256 of each of the 155 supplied files, both before and after publication normalisation. New explanatory/reconstruction files are not part of that original-file index.

Before publication, all 18 holdout prediction tables were checked against their consolidated accuracies and full confusion matrices. All nine cohort selection manifests contain 150 images per class. All nine training records match the supplied runner's SHA-256 and the stated architecture/epoch/batch settings. The baseline audit records zero exact and zero pHash-only matches against 319 protected images; it also checked 737 validation images and verified selected baseline hashes.

The raw source images/checkpoints were not supplied in this package, so these checks validate the consistency of recorded evidence, not independent reproduction of training or image-level audit execution.

## Path normalisation and integrity

Machine-specific roots in JSON/CSV are replaced with `${PROJECT_ROOT}`, `${ADDITIONAL_DATA_ROOT}` and `${JIC_ROOT}`. Original numerical results, image identifiers, model hashes and recorded original manifest hashes are retained. JSON formatting and CSV line endings may change. A stored `manifest_sha256` describes the original executed manifest, not the path-normalised published copy; use `FILE_INTEGRITY.json` to distinguish these versions.

These placeholders are documentation tokens, not automatically expanded by the archived scripts. Make separate working copies and replace them with real paths before execution; retain the published evidence unchanged. Dataset directory case/layout may also require adaptation on a different operating system.

## Reproduction scope

The supplied snapshots import NumPy, TensorFlow, scikit-learn, Pillow and ImageHash. Recorded training versions are TensorFlow 2.18.1 and NumPy 2.0.2; other dependency versions were not captured in this upload. Use a compatible environment and record its complete versions for a rerun. Exact numerical identity across machines is not guaranteed.

Run commands from this package directory (where the archived `src/` package lives). With separately obtained original/JIC images and staged IARI/Ethiopia source pools, rebuild source-excluded cohorts using:

```bash
python prepare_source_excluded.py --baseline-manifest /path/to/local-baseline/study_manifest.json --staging-root /path/to/diversity_sources_20261016 --iari-root /path/to/WheatLeafRust --out-dir /path/to/new-source-excluded
```

The baseline manifest supplied to this command must contain valid local paths. This reconstructs the sampling/screening logic, but the upstream staging inputs and their construction are not included in this upload. Consequently this package is a supporting evidence archive with partial reproduction tooling, not a complete end-to-end dataset rebuild.

Train each generated condition separately, for example:

```bash
python -m src.run_generalisation_study --manifest /path/to/new-source-excluded/add_iari_exclude_ethiopia/study_manifest.json --condition n150_mix025_seed42 --architecture mobilenetv2 --epochs 10 --batch-size 8
```

Repeat for seeds 43/44 and the other group. The runner selects checkpoints using internal validation, then evaluates the previously examined JIC benchmark. Use the archived holdout evaluator with both new group roots, IARI test and Ethiopia holdout; set `--expected-models 6`. Use fresh output directories; existing results are intentionally preserved.

The original `audit_wheat_cross_sources.py` snapshot performs the broader cross-source audit. The specific retrospective selected-baseline audit was run inline; its outputs are archived here, but its original executed script file did not exist. The audit does not establish plant, field or session independence. Do not tune subsequent experiments against these already examined protected holdout results.
