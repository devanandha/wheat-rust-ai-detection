# Experiment Protocol

## Scope

This protocol records the experiments required to reproduce the manuscript. Dataset images are not redistributed in the repository.

## Datasets

- Original dataset: 2,942 training images and 737 validation images across Brown rust, Healthy and Yellow rust.
- External dataset: compatible 489-image subset of Wheat Disease Dataset - Small (DOI: 10.5281/zenodo.7307816), comprising 128 Brown rust, 122 Healthy and 239 Yellow rust images.

## Models

- Healthy-only convolutional autoencoder.
- ImageNet-pretrained MobileNetV2.
- ImageNet-pretrained EfficientNetB0.
- ImageNet-pretrained ResNet50.

All architecture-specific settings, random seeds, preprocessing, stopping rules and checkpoint-selection rules must be exported in machine-readable run metadata before submission.

## Evaluation

For every classifier and evaluation dataset, retain:

- ordered class names;
- sample counts;
- predictions and probabilities;
- confusion matrix as CSV;
- classification report as JSON;
- overall accuracy and macro/weighted precision, recall and F1;
- model/checkpoint identifier;
- command and run timestamp.

## Dataset integrity

Run exact SHA-256 checks and perceptual-hash checks across training, internal validation and external evaluation sets. Retain pair-level CSV/JSON output, threshold, file identifiers and aggregate counts.

## Grad-CAM faithfulness

Evaluate a deterministic balanced sample of 60 internal validation images: 20 per class. Mask the highest-activation 20% of Grad-CAM pixels using image mean RGB and compare the confidence change with five spatially coherent random-region masks per image. Use seed 42. Retain image-level CSV, summary JSON and code version.

## Statistical reporting

Add bootstrap 95% confidence intervals for internal and external accuracy and macro-F1. Where model predictions are available for the same images, use paired comparisons rather than treating model results as independent.

## Reproducibility rule

A numerical manuscript claim is verified only when it can be traced to a committed machine-readable artifact or regenerated from the documented datasets and committed code.
