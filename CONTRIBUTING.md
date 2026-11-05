# Contributing and independent evaluation

Thanks for your interest in **Wheat Rust AI Detection**. We welcome independent reproduction attempts, constructive criticism, failure reports, bug fixes and documented improvements. **Positive feedback or an endorsement is not expected.**

**Start here:** [Independent evaluation invitation — Issue #3](https://github.com/devanandha/wheat-rust-ai-detection/issues/3).

## Pick a contribution path

### A. Check archived numbers without downloading datasets (easiest)

This task needs only Python 3 and the files in the repository. No TensorFlow, training, or image dataset download is required.

From the repository root:

```bash
python tools/verify_generalisation_aggregates.py
```

The script checks that the 18 archived run rows form six groups with seeds 42, 43 and 44, and that the means and **sample standard deviations** in `research/dataset_generalisation/condition_summary.csv` match the values in `per_run_results_18.csv`.

**Scope:** This is a consistency check of author-published tabular evidence. It does **not** independently reproduce training, inference, source images, class labels, checkpoint selection or data independence. A passing check must not be presented as third-party confirmation of experimental validity.

Related study and provenance notes:

- [18-run extension overview](research/dataset_generalisation/README.md)
- [Dataset provenance and permissions](research/dataset_generalisation/dataset_provenance.md)
- [Source-exclusion audit and limits](research/source_exclusion_20261023/README.md)

### B. Re-run model evaluation using independently obtained data (more involved)

Read the [main README](README.md), [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [external validation](docs/EXTERNAL_VALIDATION.md) and [source-exclusion addendum](docs/WHEAT_RUST_SOURCE_EXCLUSION_20261023.md) before beginning.

- Obtain third-party wheat-leaf datasets **from their original publishers**; respect their licences and attribution requirements. Source images are not redistributed here.
- Preserve a strict separation between training, model selection and independently held-out test data. Some historical benchmark data had already been examined and must **not** be described as a newly blind test.
- Record source, dataset version, image identifiers/hashes where permitted, class mapping, exclusions, sample counts, split rule, preprocessing, model weights/commit, software versions and random seeds.
- Use separate output directories; do not overwrite committed historical results.
- Report per-class confusion matrices and metrics, not only accuracy. State when a holdout contains only two observed classes, and do not infer plant/field/session independence from image-level hashes.
- If you cannot reproduce a historical run because raw cohort inventories, checkpoints or datasets are unavailable, **report that limitation explicitly** rather than guessing replacements or inventing a matching result.

The [18-run extension](research/dataset_generalisation/README.md) is a documented results release but **not** a completely self-contained end-to-end training reproduction package. The [source-exclusion evidence](research/source_exclusion_20261023/README.md) likewise includes partial reconstruction tools rather than all original image inputs.

### C. Methodological feedback (no code required)

Comments on domain shift, sampling bias, source/class confounding, Grad-CAM, evaluation design, uncertainty, calibration, failure cases or data provenance are welcome. Negative findings and methodological disagreements are especially useful when substantiated.

## What to include when reporting an independent check

1. **Scope:** archived arithmetic audit, new inference, independent dataset evaluation, code review or methodological critique.
2. **Version:** repository commit SHA and model checkpoint/checksum if used.
3. **Environment:** operating system, Python and library versions.
4. **Data:** origin, licence, version, class mapping and sizes (do not upload non-redistributable images).
5. **Method:** commands/code, seed, splits, preprocessing and whether data were previously examined.
6. **Outcome:** numerical results or specific observations, deviations, errors and limitations.

Open a GitHub issue for a new finding, reply to [Issue #3](https://github.com/devanandha/wheat-rust-ai-detection/issues/3) for discussion, or propose a focused pull request. Keep contributions respectful, relevant and reproducible. Avoid posting private data or copyrighted datasets without permission.

## Attribution and interpretation

You retain credit for your own independent contributions. By participating, you are **not** being asked to endorse the author, agree with the conclusions or provide immigration-related evidence. Any future public acknowledgement or quotation of your feedback should use accurate attribution and your permission. The project is a **research prototype**, not a validated field-diagnostic service.
