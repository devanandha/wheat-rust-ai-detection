# Exploratory acquisition-shift findings — original validation versus JIC

**Date:** 18 November 2026  
**Status:** Measured from user-run local script and uploaded output CSV/JSON files; research images were not accessed by this report author. **Descriptive only — no model rerun, experiment, or causal claim.**

## Input and counts

The author ran `tools/audit_acquisition_shift.py` on the complete original internal `val` collection and the 3-class JIC external `val` collection. All reported measurements came from `source_class_summary.csv`, `per_image_metrics.csv`, and `audit_report.json`, which are retained locally. These are **previously examined** evaluation datasets, not new blind tests.

| Source | Brown Rust | Healthy | Yellow Rust | Total |
|---|---:|---:|---:|---:|
| Original validation | 226 | 279 | 232 | 737 |
| JIC external | 128 | 122 | 239 | 489 |
| **Combined** | **354** | **401** | **471** | **1,226** |

All 1,226 images were readable. Cross-source exact-file SHA-256 duplicates detected: **0**. These checks do not rule out near-duplicates, plant/field overlaps, label errors or dataset construction biases.

## Finding A — image dimensions are strongly associated with classes in the original validation set

The classifier resizes images to 224 × 224. The original supplied validation classes have different source-image sizes; **the model is not explicitly given the original width/height**, but resizing could preserve scale/interpolation artifacts that act as shortcuts.

| Class | Source | Median width (px) | Median height (px) | Images with at least one original side < 224 px |
|---|---|---:|---:|---:|
| Brown Rust | Original | 175 | 205 | 178/226 (78.8%) |
| Healthy | Original | 892 | 790 | 2/279 (0.7%) |
| Yellow Rust | Original | 174 | 147 | 220/232 (94.8%) |
| Brown Rust | JIC | 567 | 756 | 0/128 (0.0%) |
| Healthy | JIC | 567 | 756 | 0/122 (0.0%) |
| Yellow Rust | JIC | 709 | 945 | 0/239 (0.0%) |

Width and height are **separate medians**, and need not form a real image. In original validation, 114/226 Brown Rust and 154/232 Yellow Rust images have a side < 128 pixels; none of the original Healthy images does.

**Interpretation:** The class-dependent original resolution distribution is a concrete candidate shortcut mechanism that deserves investigation **in the original training split first**. These descriptive results **do not prove** the original classifier learned a resolution shortcut, or that it caused external errors.

## Finding B — within-class saturation differs strongly by source

All figures are mean centre-region saturation on a normalized 0–1 HSV scale, computed after resizing each image to 128 × 128 for the audit.

| Class | Original validation | JIC | Difference (JIC minus original) |
|---|---:|---:|---:|
| Brown Rust | 0.5387 | 0.2841 | -0.2546 |
| Healthy | 0.4190 | 0.2729 | -0.1461 |
| Yellow Rust | 0.5177 | 0.2830 | -0.2347 |

Additional descriptive differences: in Yellow Rust images, the mean border-minus-centre brightness proxy is -0.0657 (original) versus +0.0026 (JIC). In Brown Rust images, mean brightness is 0.4681 (original) versus 0.5100 (JIC).

**Interpretation:** Colour and image-edge statistics differ within the same disease labels between these two collections. Image-edge metrics are **not validated plant/background masks**, and sRGB values are not calibrated illumination. Differences could arise from cameras, preprocessing, acquisition conditions, source curation or other confounded factors.

## Relation to observed errors (previous studies, not rerun here)

The original MobileNetV2 internal validation was 728/737 (98.78%) versus JIC 188/489 (38.45%). JIC Yellow Rust recall was only 15/239 (6.28%), with 126 images called Brown Rust and 98 called Healthy. The present audit does not link acquisition measurements to individual prediction errors and therefore cannot explain this failure.

## Next controlled steps

1. **Audit original training, original validation and JIC separately** with the same descriptive tool. Check whether the same within-class size/saturation relationships hold in training rather than inferring from the validation subset.
2. Establish provenance for image resolution and cropping, and investigate whether source-image dimensions or upscaling artifacts are class-correlated. Inspect representative original images and labels manually, without drawing conclusions from image-size metadata alone.
3. Distinguish the hypotheses of resolution/upscaling, colour/saturation and contextual background cues; prespecify one targeted intervention, avoiding arbitrary testing of many changes on JIC.
4. Hold the model architecture, dataset composition, random seeds, train/validation protocol and evaluation definitions constant for a controlled comparison, and preserve all negative findings.
5. Treat the already inspected JIC set as **exploratory only**. For external claims seek a genuinely untouched dataset with compatible classes and documented provenance and, where possible, plant/field/session identifiers.
6. Do not claim novelty simply from cross-dataset performance decline or simple image statistics: articulate a specific new controlled finding relative to prior literature.

## Reproduction command for next audit

From the repository root on Windows:

```cmd
python tools\audit_acquisition_shift.py --source "original_train=F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data\train" --source "original_val=F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data\val" --source "jic=F:\Github\external-wheat-validation\val" --output-dir "outputs\acquisition_shift_train_val_jic"
```

The expected original training count is 2,942, provided the local `train` folder still contains the same original sample set. The audit produces descriptive CSV/JSON only; it does not retrain any model.

**Research integrity note:** The audit's SHA-256 cross-source checks cannot establish plant/field independence. Folder-level image counts and model accuracy are distinct forms of evidence. The interpretation remains exploratory until independent images and controlled interventions are evaluated.
