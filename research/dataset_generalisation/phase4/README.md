# Wheat Rust AI — Phase 4 External Evaluation

## Overview

Phase 4 evaluates all 18 frozen Phase 3 MobileNetV2 checkpoints
against a screened subset of a separately downloaded Kaggle
wheat-leaf disease dataset.

This is an exploratory cross-dataset evaluation, not confirmed
independent field validation.

## Evaluation dataset

Original three-class test split: 239 images.

Screening exclusions:
- 38 images with exact SHA-256 overlaps.
- 2 additional images with perceptual-hash distances of 6–8,
  conservatively excluded pending review.

Final evaluation subset: 199 images.

| Class | Images |
|---|---:|
| Brown_rust | 39 |
| Healthy | 77 |
| Yellow_rust | 83 |

Historical training candidates, original validation, JIC, and
Kaggle training/validation images were included in the
documented overlap screening.

This screening does not prove field-level independence or
complete independence from all previously published imagery.

## Model configuration

- Architecture: MobileNetV2
- Input: 224 x 224 RGB images
- Pixel values: 0–255
- Preprocessing: embedded in the saved model
- Class order: Brown_rust, Healthy, Yellow_rust
- No retraining or Phase 4-based model selection

## Results

The evaluation_all_18 directory contains:

- summary_all_18.csv
- Individual classification reports
- Confusion matrices for all 18 models

See PHASE4_FINDINGS.md for interpretation and limitations.

## Reproducibility requirements

The Python scripts are archived from the original local
evaluation environment. They contain machine-specific
Windows paths and are not directly portable.

Independent execution requires:

1. Access to appropriately licensed source images.
2. Reproduction of the documented image-overlap screening.
3. A selection manifest identifying the 199 included images.
4. The original 18 Phase 3 model checkpoints.
5. Adjustment of local paths in the evaluation scripts.
6. Compatible TensorFlow, Keras and other dependencies.

The historical selection manifest and frozen protocol are
preserved locally but are not included in this public package.
Consequently, this public package documents the evaluation
and its aggregate outputs but is not a self-contained
reproduction of the complete experiment.

Original images, model checkpoints and image-level predictions
are not distributed in this directory.

## Scientific limitations

The Kaggle dataset contains duplicate images across its
original train, validation and test partitions.

Its full image provenance has not been independently verified.

Source-class confounding in Phase 3 remains unresolved.

The resulting scores should not be interpreted as evidence
of real-world deployment readiness.

The principal observation is that improvements in aggregate
accuracy from source mixing did not transfer consistently
across individual disease classes.

## Configuring the dataset location

The public evaluation scripts use the PHASE4_DATASET_DIR
environment variable instead of a hardcoded dataset path.

Example for PowerShell:

    $env:PHASE4_DATASET_DIR = "D:\datasets\wheat-leaf\Datasets"

This specifies the root of the extracted Kaggle dataset.

Note: The scripts also require the original screened
selection manifest, frozen evaluation protocol and
saved model checkpoints. Those assets are not included
in this public research package.

The scripts are therefore archived evaluation code,
not a fully self-contained reproduction.
