# Exploratory acquisition-shift findings — original training, validation, and JIC

**Date:** 18 November 2026  
**Status:** Measured from user-run local script and uploaded output CSV/JSON files. A second audit on 18 November 2026 covered all original training, original validation and JIC images. Source images were not accessed in this analysis. **Exploratory only — no neural-network rerun, independent blind test, or causal claim.**

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

## Follow-up audit — the pattern persists in original training (4,168 images)

A second local audit of `original_train`, `original_val` and `jic` processed **2,942 + 737 + 489 = 4,168 readable images**, with no unreadable files and **zero cross-source exact SHA-256 duplicate groups**. The latter does not establish source, plant, field or session independence. All samples were included.

### Original pixel dimensions are class-associated in both training and validation

| Class | Dataset | Number | Separate medians width × height (px) | At least one side < 224 px |
|---|---|---:|---:|---:|
| Brown Rust | Original training | 902 | 190 × 222 | 655/902 (72.6%) |
| Brown Rust | Original validation | 226 | 175 × 205 | 178/226 (78.8%) |
| Brown Rust | JIC | 128 | 567 × 756 | 0/128 (0%) |
| Healthy | Original training | 1,116 | 855 × 749 | 13/1,116 (1.2%) |
| Healthy | Original validation | 279 | 892 × 790 | 2/279 (0.7%) |
| Healthy | JIC | 122 | 567 × 756 | 0/122 (0%) |
| Yellow Rust | Original training | 924 | 172 × 150 | 877/924 (94.9%) |
| Yellow Rust | Original validation | 232 | 174 × 147 | 220/232 (94.8%) |
| Yellow Rust | JIC | 239 | 709 × 945 | 0/239 (0%) |

The medians for width and height are calculated separately. Crucially, in **the actual training data**, most rust images have a source dimension below the model's 224 px resizing target, while very few Healthy images do. It is plausible that source/resampling cues correlate with labels, although the neural network receives resized pixel arrays, not original width/height metadata.

### Colour differences also appear in both original splits

Mean **centre-region HSV saturation** (0–1), resized to 128 × 128 for this *audit*:

| Class | Original train | Original validation | JIC |
|---|---:|---:|---:|
| Brown Rust | 0.5444 | 0.5387 | 0.2841 |
| Healthy | 0.4212 | 0.4190 | 0.2729 |
| Yellow Rust | 0.5201 | 0.5177 | 0.2830 |

The original train/validation statistics are similar; JIC saturation is consistently lower within each class. These could reflect acquisition and/or curation/preprocessing differences rather than disease pathology.

### Exploratory metadata-only diagnostic

To test **whether class labels can be predicted from original image dimensions at all**, a small `sklearn.tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)` was fitted using **only original image width and height** and labels from `original_train`. No RGB pixels, pretrained features or neural-network predictions were used; no hyperparameter selection against a holdout was performed in this one-off calculation.

| Data | Accuracy | Balanced accuracy |
|---|---:|---:|
| Original training | 73.69% | 71.67% |
| Original validation | **71.51%** | **69.35%** |
| JIC external | 24.95% | 33.33% |

The fitted tree predicted **Healthy for every JIC image** (122/489 correct). For context, simply choosing the largest original-validation class (Healthy) would give 279/737 = **37.86%** accuracy. This supports the existence of a strong **class–metadata association** and its lack of transfer, **not** a claim that the MobileNetV2 model uses the same rule. The test was exploratory and uses already examined evaluation sets; it is not a new blinded result. Exact values depend on the described sklearn implementation/version.

## Relation to observed errors (previous studies, not rerun here)

The original MobileNetV2 internal validation was 728/737 (98.78%) versus JIC 188/489 (38.45%). JIC Yellow Rust recall was only 15/239 (6.28%), with 126 images called Brown Rust and 98 called Healthy. The present audit does not link acquisition measurements to individual prediction errors and therefore cannot explain this failure.

## Next controlled steps

1. **Completed:** Audit original training, validation and JIC separately. The class-associated source image size and saturation patterns occur in the original training set as well as validation. Next, examine the data-collection/preprocessing history and verify class labels and image provenance.
2. Establish provenance for image resolution and cropping, and investigate whether source-image dimensions or upscaling artifacts are class-correlated. Inspect representative original images and labels manually, without drawing conclusions from image-size metadata alone.
3. Distinguish the hypotheses of resolution/upscaling, colour/saturation and contextual background cues; prespecify one targeted intervention, avoiding arbitrary testing of many changes on JIC.
4. Hold the model architecture, dataset composition, random seeds, train/validation protocol and evaluation definitions constant for a controlled comparison, and preserve all negative findings.
5. Treat the already inspected JIC set as **exploratory only**. For external claims seek a genuinely untouched dataset with compatible classes and documented provenance and, where possible, plant/field/session identifiers.
6. Do not claim novelty simply from cross-dataset performance decline or simple image statistics: articulate a specific new controlled finding relative to prior literature.

## Reproduction command for the completed three-source audit

From the repository root on Windows:

```cmd
python tools\audit_acquisition_shift.py --source "original_train=F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data\train" --source "original_val=F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data\val" --source "jic=F:\Github\external-wheat-validation\val" --output-dir "outputs\acquisition_shift_train_val_jic"
```

The completed audit found 2,942 training images and 4,168 total images. It produces descriptive CSV/JSON only; it does not retrain any model. The metadata-only decision tree described above was computed separately from the uploaded CSV and is not part of the audit script.

**Research integrity note:** The audit's SHA-256 cross-source checks cannot establish plant/field independence. Folder-level image counts and model accuracy are distinct forms of evidence. The interpretation remains exploratory until independent images and controlled interventions are evaluated.
