# High Internal Accuracy, Weak External Generalisation: A Reproducible Evaluation of Deep Learning for Wheat-Rust Classification

**Devanandha Vellaramkuzhiyil Shaji**

Independent Researcher, Bangor, Northern Ireland, United Kingdom

Work initiated during the MSc Artificial Intelligence programme at Ulster University

Email: devanandhadevv@gmail.com

## Abstract

Deep-learning systems for plant-disease recognition frequently report high accuracy on held-out images drawn from the same source as their training data. Such results may not represent performance on images acquired under different field, camera, background or curation conditions. This study evaluates a three-class wheat-leaf classification task—Brown rust, Healthy and Yellow rust—with emphasis on external generalisation, failure behaviour, dataset integrity and explainability. A leakage-controlled healthy-only convolutional autoencoder experiment was conducted using separate training, early-stopping, threshold-calibration and held-out evaluation subsets. It failed in the intended direction: disease recall was 0 and the area under the receiver-operating-characteristic curve was 0.027 when higher reconstruction error was interpreted as disease. Three ImageNet-pretrained classifiers were then evaluated on a 737-image internal validation set and a separate 489-image external subset. MobileNetV2 achieved 98.78% internal accuracy but 38.45% externally. EfficientNetB0 achieved 99.05% internally and 41.31% externally, while ResNet50 achieved 99.59% internally and 38.04% externally. External Yellow rust recall ranged from 0.42% to 12.97%. Exact SHA-256 and perceptual-hash checks found no cross-dataset matches at a Hamming-distance threshold of five. A Grad-CAM perturbation experiment on 60 balanced internal-validation images produced a mean confidence reduction of 0.209 after masking highly activated regions, compared with 0.039 after spatially coherent random masking. The effect was strongly class-dependent and absent for Healthy images. The results demonstrate that near-perfect within-source accuracy was insufficient evidence of robust wheat-rust recognition. External validation, class-level failure analysis, dataset-audit artifacts and quantitative explanation checks should accompany headline accuracy claims. The system remains a research prototype and is not validated for agronomic diagnosis.

**Keywords:** wheat rust; plant disease classification; external validation; domain shift; transfer learning; Grad-CAM; explainable AI; reproducibility

## 1. Introduction

Wheat rust diseases are visually expressed on plant tissue and are therefore plausible candidates for image-based decision support. Convolutional neural networks (CNNs) and transfer learning have made it possible to construct high-performing plant-disease classifiers using considerably fewer task-specific images than would be required to train large visual models from scratch. A prominent early demonstration trained CNNs on 54,306 controlled-background PlantVillage images and reported 99.35% accuracy on a held-out split [1]. However, performance fell substantially when the same models were evaluated on images obtained under different conditions [1]. This distinction between within-source validation and cross-source performance is central to the present study.

Plant-disease images may encode signals unrelated to pathology, including background, lighting, scale, camera processing, framing and dataset-specific curation. Barbedo [2] identified dataset size, visual diversity, symptom representation, image acquisition and background conditions as important determinants of plant-disease recognition performance. More recent cross-dataset research has similarly shown that models trained under controlled conditions may generalise poorly to field imagery and that architecture choice alone does not guarantee robustness [3]. Consequently, a high internal-validation score is not, by itself, evidence that a model has learned biologically meaningful or transferable disease features.

This paper extends a 2024 MSc Artificial Intelligence project on wheat-rust recognition. The original project investigated a healthy-only autoencoder and supervised transfer learning. The extension reframes the work around a stricter question:

> **How well do deep-learning models for wheat-rust classification generalise beyond their training distribution, and can explainability and failure-case analysis reveal the visual factors underlying model errors?**

The study makes five contributions:

1. It reports the unsuccessful anomaly-detection experiment rather than suppressing a negative result.
2. It compares internal and independent-source performance for MobileNetV2, EfficientNetB0 and ResNet50.
3. It audits exact duplicates and perceptual near-duplicates within and across datasets.
4. It analyses class-level and high-confidence external failures.
5. It evaluates Grad-CAM with a quantitative perturbation test instead of relying solely on selected heatmaps.

The objective is not to propose a deployable diagnostic system. It is to document how apparently excellent validation performance can coexist with severe cross-dataset failure, and to provide machine-readable evidence for every principal numerical result.

## 2. Related Work

### 2.1 Deep learning for plant-disease recognition

Mohanty et al. [1] demonstrated the potential of transfer learning for plant-disease classification using the PlantVillage dataset. Their work also exposed a generalisation problem: performance on controlled images did not transfer at the same level to images collected from other sources. Barbedo [2] subsequently examined factors that can make plant-disease experiments appear stronger than their likely operational performance, including limited diversity and acquisition-specific characteristics. Ahmad et al. [3] directly studied cross-dataset and controlled-to-field generalisation across multiple plant-disease datasets, reinforcing the need to test models beyond the distribution from which training and validation samples were drawn.

The present work applies that external-validation principle to wheat rust. The independent dataset contains wheat-disease images acquired in realistic growth conditions and includes Brown rust, Healthy, Mildew, Septoria and Yellow rust categories [4]. Only the three categories shared with the trained classifier were included in closed-set evaluation. Mildew and Septoria were excluded rather than incorrectly mapped to one of the supported classes.

### 2.2 Transfer-learning architectures

MobileNetV2 uses inverted residual blocks, linear bottlenecks and depthwise convolutions to provide a computationally efficient visual backbone [5]. EfficientNet introduces compound scaling of network width, depth and resolution [6]. ResNet uses residual learning to facilitate optimisation of deeper networks [7]. These architectures represent distinct capacity and design trade-offs. Comparing them under an unchanged class task helps test whether the external-performance gap is specific to the original MobileNetV2 backbone or persists across established CNN families.

### 2.3 Explainability and faithfulness

Grad-CAM uses gradients flowing into the final convolutional layer to produce a class-discriminative localisation map [8]. A visually plausible heatmap, however, is not sufficient evidence that an explanation is faithful to the model. Adebayo et al. [9] showed that visual inspection alone can be misleading and argued for behavioural tests of saliency methods. The present study therefore measures whether masking Grad-CAM-selected regions changes model confidence more than masking a spatially coherent random region of comparable area. This tests influence on the model, not biological agreement with expert-labelled lesions.

### 2.4 Confidence under distribution shift

Modern neural networks can be poorly calibrated, such that predicted confidence does not reliably correspond to the probability of correctness [10]. This matters under dataset shift, where a classifier may be confidently wrong. The present external failure analysis therefore retains confidence values for misclassified images and does not interpret softmax confidence as diagnostic certainty.

## 3. Materials and Methods

### 3.1 Study design

The study comprised four linked experiments: (1) healthy-only anomaly detection; (2) supervised three-class internal validation; (3) cross-source external validation and architecture comparison; and (4) Grad-CAM faithfulness evaluation. All principal numerical claims are traceable to committed JSON or CSV artifacts. Dataset images are not redistributed.

### 3.2 Original dataset

The original dataset contained 3,679 images arranged in predefined training and validation folders. The training split contained 2,942 images: 902 Brown rust, 1,116 Healthy and 924 Yellow rust. The validation split contained 737 images: 226 Brown rust, 279 Healthy and 232 Yellow rust. An integrity audit found no corrupt images and no exact duplicate groups. Images exhibited substantial variation in their original dimensions; all were resized to 224 × 224 pixels when loaded by the model pipeline.

### 3.3 External dataset

External validation used the compatible subset of *Wheat Disease Dataset – Small* [4]. The repository’s evidence inventory records each included image using its relative path, dimensions, byte size, SHA-256 digest and perceptual hash. The subset contained 489 images: 128 Brown rust, 122 Healthy and 239 Yellow rust. The images were never used for training, early stopping, threshold selection or architecture selection. The dataset’s Mildew and Septoria images were excluded because the trained classifier did not support those categories.

### 3.4 Dataset-overlap checks

Exact overlap was assessed by SHA-256 hashing. Perceptual similarity was assessed using ImageHash pHash and a Hamming-distance threshold of five. The audit compared all 3,679 original images with all 489 external images. Within the original train/validation split, the same perceptual threshold identified one Healthy near-duplicate pair at distance two. The affected validation image was removed in a sensitivity evaluation.

These procedures identify identical files and close perceptual matches under the chosen algorithm and threshold. They do not prove that datasets are statistically independent at the field, plant, photographer or source level.

### 3.5 Leakage-controlled healthy-only anomaly detector

The convolutional autoencoder accepted 224 × 224 RGB images with pixel values scaled to the range [0,1]. The encoder contained three stride-two convolutional layers with 32, 64 and 128 filters. The decoder used transposed-convolutional layers with 64, 32 and three output filters. The model was trained using mean squared reconstruction error as its objective function.

To prevent information leakage, the 1,116 Healthy images from the original training set were deterministically divided using random seed 42 into three mutually exclusive subsets: 892 images for model training, 112 images for early-stopping decisions and 112 images for anomaly-threshold calibration. The anomaly threshold was defined as the 95th percentile of reconstruction errors calculated on the separate calibration subset.

The predefined 737-image validation set was held out from model fitting, early stopping and threshold selection. It was used only for final evaluation. Under the prespecified anomaly rule, an image was classified as diseased when its reconstruction error exceeded the calibrated threshold; otherwise, it was classified as Healthy.

### 3.6 Supervised classifiers

MobileNetV2 [5], EfficientNetB0 [6] and ResNet50 [7] were initialised with ImageNet weights. The convolutional base was frozen. Each model used architecture-specific preprocessing followed by global average pooling, dropout of 0.25 and a three-unit softmax layer. Training augmentation comprised horizontal flipping, rotation of 0.08, zoom of 0.10 and contrast variation of 0.10.

Models were optimised with Adam at a learning rate of 0.001 and categorical cross-entropy loss. The default batch size was 32, except ResNet50, which used batch size eight because of local memory constraints. Training used seed 42, early stopping with patience three and restoration of the best weights, checkpointing on validation performance, and learning-rate reduction by a factor of 0.3 after two non-improving epochs. MobileNetV2 was trained for 10 epochs. ResNet50 completed four full epochs before a memory-allocation failure during epoch five; the best checkpoint saved during completed training was retained and evaluated. This interruption is treated as a reproducibility limitation, not as evidence about the architecture itself.

### 3.7 Classification evaluation

For each model, internal and external evaluation used the fixed class order Brown rust, Healthy and Yellow rust. Metrics included accuracy, per-class precision, recall and F1-score, macro averages, weighted averages and confusion matrices. Misclassified-image records retained actual class, predicted class and maximum softmax probability.

Wilson 95% confidence intervals were calculated for accuracy to describe binomial uncertainty. They are not presented as corrected estimates of real-world performance. Formal paired significance tests between architectures were not performed; therefore, small differences in external accuracy are described rather than claimed as statistically superior.

### 3.8 Grad-CAM faithfulness experiment

Grad-CAM [8] was evaluated on a deterministic balanced sample of 60 correctly classified internal-validation images: 20 per class. For each image, the model’s original predicted class and confidence were recorded. A Grad-CAM map was generated for that predicted class, resized to the input resolution and used to select the highest-activation 20% of pixels. Selected pixels were replaced using the image mean RGB value, after which confidence in the original predicted class was measured again.

The comparison condition used five spatially coherent random rectangular masks per image, covering approximately the same image fraction. Random masking used seed 42. The principal outcome was:

**Grad-CAM advantage = Grad-CAM confidence drop − mean random-region confidence drop.**

A positive value indicates that removing Grad-CAM-selected regions reduced confidence more than the spatial random baseline. The experiment measures model faithfulness. It does not establish that activated regions correspond to biologically verified lesions.

### 3.9 Reproducibility and evidence controls

The public repository contains source code, confusion matrices, classification reports, training history where available, external misclassification records, overlap audits, the external inventory, Grad-CAM image-level results and model hashes. A validator maps 13 manuscript evidence groups to their machine-readable artifacts. The evidence-generation environment used Python 3.12, TensorFlow 2.18.1, NumPy 2.0.2, pandas 2.3.3, Pillow 11.3.0, scikit-learn 1.9.1 and ImageHash 4.3.2 on Windows 11. Historical training hardware and package versions were not retrospectively inferred and remain a disclosed limitation.

## 4. Results

### 4.1 Leakage-controlled anomaly-detection results

The following findings refer to the leakage-controlled experiment described in Section 3.5, in which model training, early-stopping decisions, threshold calibration and final evaluation used separate data subsets. The prespecified anomaly-detection rule failed. Disease recall and disease F1-score were both 0. The held-out accuracy was 33.65%, and ROC-AUC was 0.027 when higher reconstruction error was interpreted as stronger evidence of disease. Mean reconstruction error was 0.00662 for Healthy images, compared with 0.00154 for Brown rust and 0.00114 for Yellow rust. Thus, disease images were reconstructed with lower error than Healthy images. Reversing the score direction produced a diagnostic AUC of 0.973. However, this diagnostic calculation does not validate the original anomaly-detection hypothesis; it demonstrates that the observed separation occurred in the opposite direction from the prespecified decision rule.

### 4.2 Internal classification

All three supervised classifiers achieved high internal accuracy (Table 1). MobileNetV2 correctly classified 728 of 737 images. EfficientNetB0 correctly classified 730, and ResNet50 correctly classified 734.

**Table 1. Internal and external accuracy. Confidence intervals are Wilson 95% intervals.**

| Architecture | Internal correct/total | Internal accuracy (95% CI) | External correct/total | External accuracy (95% CI) | Absolute drop |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 | 728/737 | 98.78% (97.70–99.36) | 188/489 | 38.45% (34.24–42.83) | 60.33 pp |
| EfficientNetB0 | 730/737 | 99.05% (98.05–99.54) | 202/489 | 41.31% (37.03–45.72) | 57.74 pp |
| ResNet50 | 734/737 | 99.59% (98.81–99.86) | 186/489 | 38.04% (33.84–42.42) | 61.55 pp |

Removing the single detected train/validation near-duplicate changed MobileNetV2 accuracy from 728/737 (98.7788%) to 727/736 (98.7772%). The pair therefore did not materially explain the high internal result.

### 4.3 External classification

No architecture reproduced its internal performance externally. EfficientNetB0 produced the highest observed external accuracy at 41.31%, but its confidence interval overlapped those of MobileNetV2 and ResNet50. The results therefore support a shared generalisation failure, not a strong claim that one architecture solved the problem.

Class-level behaviour was highly uneven (Table 2). Yellow rust was the principal failure mode. MobileNetV2 correctly classified only 15 of 239 external Yellow rust images; EfficientNetB0 classified 31 correctly; and ResNet50 classified one correctly.

**Table 2. External per-class recall.**

| Architecture | Brown rust | Healthy | Yellow rust |
|---|---:|---:|---:|
| MobileNetV2 | 57.81% (74/128) | 81.15% (99/122) | 6.28% (15/239) |
| EfficientNetB0 | 72.66% (93/128) | 63.93% (78/122) | 12.97% (31/239) |
| ResNet50 | 52.34% (67/128) | 96.72% (118/122) | 0.42% (1/239) |

For MobileNetV2, 126 Yellow rust images were predicted as Brown rust and 98 as Healthy. Errors were not confined to uncertain cases. Mean confidence was 83.89% for Yellow rust predicted as Brown rust and 82.68% for Yellow rust predicted as Healthy. These values show that maximum softmax probability was not a reliable indicator of correctness under the external distribution.

### 4.4 Dataset-overlap audit

The cross-dataset audit checked 3,679 original images and 489 external images. It found zero exact SHA-256 matches and zero perceptual-hash matches at Hamming distance ≤ 5. No corrupt images were reported in either inventory. The external-performance reduction therefore cannot be attributed to detected direct image overlap. Conversely, the absence of matches does not identify the cause of the shift.

### 4.5 Grad-CAM faithfulness

Across 60 internal images, mean confidence fell by 0.2091 after Grad-CAM masking and by 0.0386 after coherent random-region masking. The mean Grad-CAM advantage was +0.1705, and the Grad-CAM drop exceeded the random baseline for 68.3% of images (Table 3).

**Table 3. Grad-CAM perturbation results.**

| Class | Images | Mean Grad-CAM drop | Mean random drop | Mean advantage | Grad-CAM > random |
|---|---:|---:|---:|---:|---:|
| Brown rust | 20 | 0.1847 | 0.0766 | +0.1080 | 70% |
| Healthy | 20 | −0.0079 | 0.0075 | −0.0155 | 35% |
| Yellow rust | 20 | 0.4506 | 0.0316 | +0.4190 | 100% |
| **Overall** | **60** | **0.2091** | **0.0386** | **+0.1705** | **68.3%** |

The effect was class-dependent. Yellow rust showed the strongest response, Brown rust showed a positive but variable response, and Healthy images showed no positive mean advantage. Therefore, the aggregate result should not be interpreted as uniform explanation quality.

## 5. Discussion

### 5.1 Principal finding

The principal finding is the scale and consistency of the internal-to-external gap. Internal accuracy ranged from 98.78% to 99.59%, whereas external accuracy ranged from 38.04% to 41.31%. The pattern persisted across a lightweight mobile architecture, a compound-scaled architecture and a residual network. This supports the interpretation that the problem was not resolved by substituting a different established CNN backbone.

The result is consistent with earlier warnings about controlled plant-disease datasets [1,2] and cross-condition evaluation [3]. A validation split can accurately estimate performance on its own source distribution while remaining a poor guide to another source. The correct interpretation of the internal results is therefore narrow: the models separated the three classes extremely well within the supplied validation distribution. They did not demonstrate equivalent recognition under changed acquisition and curation conditions.

### 5.2 Class-specific domain shift

External Yellow rust recall collapsed for every architecture. This may reflect changes in symptom presentation, severity, leaf framing, background, illumination, image quality, annotation practice or correlations specific to the original source. The present experiment cannot isolate which factor caused the collapse. It does show that reporting only aggregate internal accuracy would have concealed the central failure mode.

EfficientNetB0 achieved the highest observed external accuracy, but the difference was modest and descriptive intervals overlapped. ResNet50 illustrates the risk of selecting models solely on internal accuracy: it achieved the best internal result yet the lowest external accuracy and only one correct Yellow rust prediction. Architecture capacity therefore did not correspond to cross-source robustness in this experiment.

### 5.3 Negative anomaly result

The autoencoder result is scientifically informative. The healthy-only model did not assign larger reconstruction errors to diseased leaves. Instead, Healthy images had greater reconstruction error on average. Possible explanations include greater diversity within the Healthy class, differences in backgrounds and compression, or the autoencoder learning low-level image statistics rather than pathology. Because the threshold was calibrated without using the held-out validation set, the failure cannot be repaired by post-hoc threshold tuning without changing the experimental question. Reporting the inverted diagnostic AUC clarifies the direction of separation while preserving the conclusion that the prespecified anomaly detector failed.

### 5.4 Interpretation of Grad-CAM

The perturbation experiment strengthens the explainability analysis beyond visual examples. Overall, masking Grad-CAM-selected pixels affected confidence more than masking coherent random regions. Yet the class-level pattern matters: the Healthy result was weak, and faithfulness on internal images does not establish validity under external shift.

Grad-CAM can be faithful to an incorrect decision. It identifies regions that influence the network, not regions confirmed by a plant pathologist. This distinction is especially important because the classifier produced high-confidence external errors. Expert lesion masks or bounding boxes would be needed to test biological localisation, while insertion/deletion curves, additional attribution methods and model-randomisation checks could provide broader technical evaluation [9].

### 5.5 Implications for evaluation practice

Three practical implications follow. First, an independent-source dataset should be treated as a core evaluation component rather than optional future work. Second, class-level recall and error direction are essential when class prevalence and failure costs are asymmetric. Third, confidence and saliency outputs should be tested rather than accepted at face value. The public evidence manifest, inventories and hashes also demonstrate how reproducibility can be strengthened without redistributing source datasets.

## 6. Limitations and Threats to Validity

This study has several limitations.

1. Only one original and one external dataset were evaluated. Results cannot be generalised to all wheat-growing regions, cultivars, cameras or disease stages.
2. External evaluation was closed-set and limited to three shared classes. The system was not tested for rejection of Mildew, Septoria, other diseases or non-wheat images.
3. Dataset provenance was audited at file and perceptual-hash level, but complete field-, plant- and photographer-level independence could not be established.
4. Architecture comparison was not a fully tuned benchmark. Hyperparameters were not independently optimised for each model, and ResNet50 training was interrupted by a memory error after four completed epochs.
5. Confidence intervals describe binomial uncertainty for the evaluated samples; they do not correct for clustering, source bias or non-random dataset construction.
6. The Grad-CAM experiment used 60 internal images, one attribution method, one masking fraction and an artificial mean-colour perturbation.
7. No expert lesion annotations were available, so biological localisation accuracy was not measured.
8. Historical training hardware and software versions were not fully preserved. Current model hashes, evidence-generation dependencies and repository commit identifiers are available, but missing historical details were not reconstructed speculatively.
9. The work has not yet undergone peer review or prospective agricultural validation.

## 7. Future Work

Future experiments should evaluate repeated cross-dataset splits, additional independently collected field datasets and explicit open-set recognition. Domain adaptation or domain generalisation methods should be assessed without using the final external test set for selection. Calibration should be evaluated using reliability diagrams, expected calibration error and temperature scaling fitted on a dedicated calibration set [10]. Explainability evaluation should incorporate expert lesion annotations, multiple attribution methods, insertion/deletion curves and external-image perturbation tests. Finally, prospective evaluation with agronomists is required before any diagnostic or treatment-related use.

## 8. Conclusion

Three transfer-learning classifiers achieved approximately 99% internal accuracy yet no more than 41.31% accuracy on an independent image source. Yellow rust recall fell as low as 0.42%, and some external errors were highly confident. A healthy-only autoencoder also failed under its intended anomaly rule. Quantitative Grad-CAM testing showed class-dependent evidence of faithfulness but did not establish biological correctness or external robustness.

The study’s main contribution is therefore not another high benchmark score. It is a reproducible account of why that score was insufficient. For image-based wheat-disease research, external validation, transparent negative results, class-level error analysis and evidence-backed explainability are necessary foundations for credible claims.

## Data and Code Availability

Source code and non-image evidence artifacts are available at: https://github.com/devanandha/wheat-rust-ai-detection

Archived project record: https://doi.org/10.5281/zenodo.22726590

The external dataset is available through its cited Zenodo record [4]. Original dataset images are not redistributed. The repository includes relative-file inventories and cryptographic/perceptual hashes to support auditability without republishing image content.

## Ethics Statement

This study used existing image datasets and did not involve human participants, personal data or animal experimentation. Dataset reuse remains subject to the terms supplied by the respective dataset publishers.

## Author Contributions

Devanandha Vellaramkuzhiyil Shaji: conceptualisation, methodology, software, investigation, data curation, validation, visualisation, writing—original draft, and writing—review and editing.

## Competing Interests

The author declares no competing interests.

## Funding

No external funding was received for this study.

## Acknowledgements

The work originated from an MSc Artificial Intelligence dissertation completed at Ulster University under the supervision of Dr Omar Nibouche. Acknowledgement does not imply manuscript authorship or endorsement of the extended results.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the author used ChatGPT by OpenAI to assist with language editing, manuscript organisation and the presentation of author-generated experimental results. After using this tool, the author reviewed and edited the content as needed, verified the reported results against the underlying evidence artifacts and takes full responsibility for the content of the published article.

## References

1. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science, 7*, 1419. https://doi.org/10.3389/fpls.2016.01419
2. Barbedo, J. G. A. (2018). Factors influencing the use of deep learning for plant disease recognition. *Biosystems Engineering, 172*, 84–91. https://doi.org/10.1016/j.biosystemseng.2018.05.013
3. Ahmad, A., El Gamal, A., & Saraswat, D. (2023). Toward generalization of deep learning-based plant disease identification under controlled and field conditions. *IEEE Access, 11*, 9042–9057. https://doi.org/10.1109/ACCESS.2023.3240100
4. Long, M. C., & Brown, J. K. M. (2022). *Wheat Disease Dataset – Small* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7307816
5. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520). https://doi.org/10.1109/CVPR.2018.00474
6. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 6105–6114). https://proceedings.mlr.press/v97/tan19a.html
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770–778). https://doi.org/10.1109/CVPR.2016.90
8. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626). https://doi.org/10.1109/ICCV.2017.74
9. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. In *Advances in Neural Information Processing Systems, 31*.
10. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330). https://proceedings.mlr.press/v70/guo17a.html

## Manuscript Status

This is a pre-submission working manuscript. It has not been peer reviewed. Journal-specific formatting, reference-style conversion, figure placement and independent technical review remain outstanding.
