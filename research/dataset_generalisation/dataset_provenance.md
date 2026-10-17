# Dataset provenance, label mappings and evaluation constraints

- **Primary training source**: 2,942 original training images before overlap screening, covering Brown rust (902), Healthy (1,116), Yellow rust (924). Original internal validation comprises 737 images. Cite the original public Wheat Rust AI project for its dataset acquisition details.
- **IARI dataset**: [Mendeley Data, `th422bg4yd`](https://data.mendeley.com/datasets/th422bg4yd/1). Staged training set: 238 labelled `diseased` mapped to `Brown_rust`, 279 `control` mapped to `Healthy`. Original supplied test: 55 diseased/Brown rust, 74 control/Healthy. **The disease-to-Brown-rust mapping is a source-dependent assumption that merits independent label verification.**
- **Ethiopia dataset**: [Mendeley Data, `wgd66f8n6h`](https://data.mendeley.com/datasets/wgd66f8n6h/1). Staged training: 81 Healthy, 166 `stripe_rust` mapped to Yellow rust; Septoria omitted. Reserved 61-image holdout: 20 Healthy, 41 Yellow rust. The original dataset lacked an equivalent predefined test partition for this study; holdout selection followed local duplicate grouping, then a seeded split.
- **JIC evaluation benchmark**: 489 images, earlier examined repeatedly. No JIC image was intentionally used for these training mixtures or internal checkpoint selection. Repeated inspection makes the benchmark exploratory rather than newly blind.

## Image similarity screens

SHA-256 exact-match and perceptual hash near-match screens used a pHash distance threshold of 5. The staging manifest reports 2 excluded Ethiopian duplicate-group members (one exact and one perceptual), 63 IARI candidate exclusions due to similarities with protected IARI test/validation, and 21 candidate exclusions due to training-pool similarities. The final mixture audit excluded one original Healthy training candidate based on a pHash distance of 2 from a holdout image. These are flagged similarities, **not confirmed duplicated plants or fields**, and false-positive pHash matches can occur across class labels. The user also carried out a separate cross-source screen in the study process; its full raw output is not present in this review package.

## Interpretation and remaining work

An evaluation source represented in mixed-source training is *not* an unseen-domain assessment. The present source datasets contain different disease-label subsets, making additional-source proportion and content partially confounded. Neither test set is three-class complete; do not interpret present-class macro-F1 as full three-class macro-F1. To strengthen domain generalisation claims, secure a truly independent new dataset or collection site, verify class mappings and field/session grouping, pre-register evaluation and avoid iterative selection using test results.

Dataset-image redistribution rights should be verified against original licenses before any third-party photographs are included in a public release. No image files are in this package.
