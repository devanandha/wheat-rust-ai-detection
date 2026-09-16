# High Internal Accuracy, Weak External Generalisation: A Reproducible Evaluation of Deep Learning for Wheat-Rust Classification

## Abstract

Deep-learning classifiers for plant disease recognition can achieve strong performance on held-out images drawn from the same source as their training data, yet this may not translate to independent image collections. This study evaluates a three-class wheat-leaf classifier for Brown rust, Healthy and Yellow rust images, extending an earlier MSc project into an analysis of external generalisation, failure behaviour and explainability. A MobileNetV2 model achieved 98.78% accuracy on a 737-image internal validation set but 38.45% on a compatible 489-image external dataset. EfficientNetB0 and ResNet50 also achieved approximately 99% internal accuracy while remaining below 42% externally, indicating that architecture replacement alone did not resolve the distribution shift. External Yellow rust recall was particularly weak. Exact-file and perceptual-hash checks did not identify cross-dataset duplicates under the stated thresholds. A Grad-CAM perturbation experiment on 60 balanced internal-validation images found a larger mean confidence reduction after masking Grad-CAM-selected regions than after spatially coherent random masking, although the effect was strongly class-dependent and weak for Healthy images. These findings show that high within-source accuracy was insufficient evidence of robust wheat-rust recognition and support routine use of external validation, class-level failure analysis and quantitative explanation checks. The system is a research prototype and is not validated for agronomic diagnosis.

## 1. Introduction

Wheat rust diseases threaten crop health and motivate research into automated image-based recognition. Transfer learning allows convolutional neural networks to obtain high classification accuracy with comparatively modest task-specific datasets. However, performance measured on a random or predefined split from one image source may be inflated by source-specific visual regularities and may fail to represent performance on images acquired under different conditions.

This study examines that problem through a reproducible extension of a 2024 MSc Artificial Intelligence dissertation. The original work explored anomaly detection and supervised transfer learning for three wheat-leaf classes. The extension changes the emphasis from maximising internal validation accuracy to testing whether that accuracy transfers to an independent dataset, determining whether the behaviour persists across established CNN architectures, examining possible image duplication, analysing high-confidence errors, and testing whether Grad-CAM-highlighted regions influence model predictions.

The primary research question is: **How well do deep-learning models for wheat-rust classification generalise beyond their training distribution, and can explainability and failure-case analysis reveal the visual factors underlying model errors?**

The contributions are: (1) a direct internal-versus-external evaluation; (2) a controlled comparison of MobileNetV2, EfficientNetB0 and ResNet50; (3) explicit exact and perceptual duplicate checks; (4) class-level and high-confidence failure analysis; and (5) a quantitative Grad-CAM perturbation experiment using a spatial random baseline.

## 2. Materials and Methods

### 2.1 Datasets

The original dataset contains 3,679 images: 2,942 training images and 737 validation images. The validation set contains 226 Brown rust, 279 Healthy and 232 Yellow rust images. The independent-source evaluation used a compatible subset of Wheat Disease Dataset - Small (DOI: 10.5281/zenodo.7307816), containing 128 Brown rust, 122 Healthy and 239 Yellow rust images. External images were not used for training or model selection.

### 2.2 Models and evaluation

The experiments included a healthy-only convolutional autoencoder and three ImageNet-pretrained classifiers: MobileNetV2, EfficientNetB0 and ResNet50. Performance was evaluated using accuracy, precision, recall, F1-score and confusion matrices. Internal and external results were analysed separately. Full run-level hyperparameters and machine-readable outputs remain subject to the evidence gate in `RESULTS_EVIDENCE_MANIFEST.md`.

### 2.3 Integrity and explainability checks

SHA-256 exact-file comparison and perceptual hashing were used to screen for duplicated or near-duplicated images. Grad-CAM faithfulness was evaluated on a deterministic balanced sample of 60 internal-validation images. The highest-activation 20% of pixels were masked and compared with five spatially coherent random masks per image using random seed 42. This measures influence on model confidence, not biological correctness.

## 3. Results

### 3.1 Internal and external classification

MobileNetV2 correctly classified 728 of 737 internal validation images (98.78%). Its external accuracy was 38.45% (188 of 489). EfficientNetB0 achieved 99.05% internally and 41.31% externally. ResNet50 achieved 99.59% internally and 38.04% externally. Thus, the model with the highest internal accuracy did not achieve the highest external accuracy.

External Yellow rust recall was 6.28% for MobileNetV2, 12.97% for EfficientNetB0 and 0.42% for ResNet50. These results identify a substantial and class-specific generalisation failure.

### 3.2 Duplicate analysis

The documented audit found no exact train/validation duplicates and one near-duplicate pair at the selected perceptual-hash threshold. Removing the affected validation image did not materially change MobileNetV2 accuracy. No exact or perceptual-hash matches were documented between the original and external datasets at the selected threshold.

### 3.3 Grad-CAM faithfulness

Across 60 images, the mean confidence reduction was 0.2091 after Grad-CAM masking and 0.0386 after coherent random-region masking, giving a mean advantage of +0.1705. Grad-CAM masking produced the larger reduction for 68.3% of images. The pattern was strongest for Yellow rust, positive but variable for Brown rust, and weak for Healthy images.

## 4. Discussion

The central result is the divergence between near-99% internal accuracy and below-42% external accuracy across all three CNN architectures. Because changing architecture did not eliminate the gap, internal model ranking alone is an inadequate basis for claims of robust disease recognition. The especially low external Yellow rust recall also shows why aggregate accuracy should be accompanied by class-level analysis.

The Grad-CAM perturbation result indicates that highlighted regions influenced model confidence overall, but faithfulness was not uniform across classes. Moreover, an explanation can accurately describe the internal basis of an incorrect prediction. Explainability therefore complements but cannot replace external performance evaluation or expert biological localisation.

## 5. Limitations

The study uses one original dataset and one external dataset, supports only three classes, and does not assess field deployment. Dataset-source differences have not been isolated causally. The Grad-CAM experiment uses one attribution method, one masking fraction and no expert lesion annotations. Several reported extension results are currently documented but require committed machine-readable artifacts before submission.

## 6. Conclusion

Strong internal validation performance did not transfer to an independent wheat-disease image source. The result persisted across MobileNetV2, EfficientNetB0 and ResNet50, and the largest failures were class-specific. Quantitative Grad-CAM testing provided limited, class-dependent evidence of model faithfulness but did not establish biological correctness. External validation, transparent negative results, artifact-level reproducibility and expert validation are necessary before such systems can support agronomic use.

## Data and code availability

Code and selected non-dataset artifacts are available at https://github.com/devanandha/wheat-rust-ai-detection. The archived project DOI is https://doi.org/10.5281/zenodo.22726590. Dataset images are not redistributed; source records and access instructions are documented in the repository.

## Submission status

This is a structured working draft, not a submitted or peer-reviewed paper. Numerical claims governed by `RESULTS_EVIDENCE_MANIFEST.md` remain provisional until their machine-readable artifacts are committed or independently regenerated.
