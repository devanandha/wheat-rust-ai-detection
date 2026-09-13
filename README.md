# Wheat Rust AI Detection

[![DOI](https://zenodo.org/badge/1351419076.svg)](https://doi.org/10.5281/zenodo.22726590)

An applied computer-vision research project for classifying wheat-leaf images as **Healthy**, **Brown Rust**, or **Yellow Rust**, with a focus on **generalisation, failure analysis, model comparison, and explainable AI**.

## Live demonstration

[Launch the Wheat Rust AI Detection app](https://devanandha-wheat-rust-ai.streamlit.app/)

This repository is a reproducible extension of my 2024 MSc Artificial Intelligence dissertation at Ulster University.

The original research explored wheat-rust classification, anomaly detection, transfer learning, image preprocessing, and object-detection concepts. The 2026 extension investigates whether strong internal validation performance transfers to an external image source, how different CNN architectures behave under distribution shift, whether train/validation similarity influences reported performance, and whether Grad-CAM explanations are quantitatively faithful to model decisions.

> This is a research prototype, not an agronomic or diagnostic tool. Predictions should not be used as the sole basis for crop-treatment decisions.

## Research focus

The current project is guided by the following research question:

> **How well do deep-learning models for wheat-rust classification generalise beyond their training distribution, and can explainability and failure-case analysis reveal the visual factors underlying model errors?**

The project therefore treats high internal validation accuracy as a starting point rather than the final result.

## Key findings

### Internal vs external performance

The MobileNetV2 classifier achieved:

- **98.78% internal validation accuracy**
- **38.45% external-source accuracy**

This represents a substantial cross-dataset generalisation gap.

The external evaluation used a separate 489-image subset containing:

- 128 Brown rust images
- 122 Healthy images
- 239 Yellow rust images

The result demonstrates that strong performance on the original validation split did not transfer reliably to the external image source.

### Architecture comparison

Three ImageNet-pretrained CNN architectures were evaluated:

| Architecture | Internal Accuracy | External Accuracy | Accuracy Drop |
|---|---:|---:|---:|
| MobileNetV2 | 98.78% | 38.45% | 60.33 pp |
| EfficientNetB0 | 99.05% | 41.31% | 57.74 pp |
| ResNet50 | 99.59% | 38.04% | 61.55 pp |

EfficientNetB0 achieved the strongest external result, but all three architectures showed a substantial cross-dataset performance reduction.

Higher internal validation accuracy therefore did not correspond to stronger external generalisation in these experiments.

### Leakage and near-duplicate analysis

The original training and validation splits were examined for possible leakage.

- **0 exact SHA-256 duplicates** were detected across the training and validation sets.
- A perceptual-hash analysis identified **1 near-duplicate pair** at the selected threshold.
- Removing the affected validation image changed accuracy from **98.7788% to 98.7772%**.

The detected near-duplicate therefore did not materially explain the high internal validation result.

These checks do not establish complete source-level or field-level independence.

### Failure analysis

The MobileNetV2 classifier correctly classified:

**728 of 737 internal validation images**

The remaining **9 errors** were extracted and analysed individually.

The project also investigates external high-confidence errors, where the classifier can produce a highly confident prediction that is nevertheless incorrect.

This reinforces the importance of examining failure behaviour rather than relying only on aggregate accuracy.

### Quantitative Grad-CAM faithfulness

Grad-CAM was evaluated using a perturbation experiment rather than relying only on qualitative heatmap examples.

A deterministic balanced sample of 60 internal validation images was used:

- 20 Brown rust
- 20 Healthy
- 20 Yellow rust

For each image, the highest-activation 20% of Grad-CAM pixels were masked and compared with spatially coherent random-region masking.

| Metric | Result |
|---|---:|
| Mean Grad-CAM confidence drop | 0.2091 |
| Mean random-region confidence drop | 0.0386 |
| Mean Grad-CAM advantage | +0.1705 |
| Grad-CAM drop greater than random | 68.3% |

Class-level behaviour varied substantially:

| Class | Mean Grad-CAM Drop | Mean Random Drop | Mean Advantage | Grad-CAM > Random |
|---|---:|---:|---:|---:|
| Brown rust | 0.1847 | 0.0766 | +0.1080 | 70% |
| Healthy | -0.0079 | 0.0075 | -0.0155 | 35% |
| Yellow rust | 0.4506 | 0.0316 | +0.4190 | 100% |

Under this perturbation protocol, Grad-CAM-highlighted regions were more influential to model confidence than randomly positioned spatial regions overall, although the effect was strongly class-dependent.

These results evaluate **model faithfulness**, not biological lesion localisation.

Expert-confirmed spatial annotations would be required to determine whether highlighted regions correspond to biologically verified wheat-rust symptoms.

## Project highlights

- MobileNetV2, EfficientNetB0, and ResNet50 classifier comparison
- Independent-source external validation
- Train/validation exact-duplicate and perceptual-similarity analysis
- Detailed failure-case analysis
- Grad-CAM explainability
- Quantitative Grad-CAM perturbation evaluation
- Reproducible training and evaluation workflows
- Precision, recall, F1-score, confusion matrices, and classification reports
- Healthy-only convolutional autoencoder experiment
- Streamlit image-upload demonstration
- Transparent reporting of positive, negative, and mixed results
- Clear separation between the original MSc dissertation and subsequent research extensions

## Internal validation results

The MobileNetV2 classifier was trained for 10 epochs and evaluated on the supplied validation split of 737 images.

| Class | Precision | Recall | F1-score | Images |
|---|---:|---:|---:|---:|
| Brown Rust | 98.65% | 97.35% | 98.00% | 226 |
| Healthy | 98.93% | 99.64% | 99.29% | 279 |
| Yellow Rust | 98.71% | 99.14% | 98.92% | 232 |
| **Overall / weighted** | **98.78%** | **98.78%** | **98.78%** | **737** |

The classifier correctly predicted **728 of 737** internal validation images.

These figures describe performance on the original dataset's validation split and should not be interpreted as real-world diagnostic accuracy.

![Confusion matrix](outputs/classifier/confusion_matrix.png)

## External validation results

The MobileNetV2 classifier was evaluated on a separate 489-image external-source dataset using the same three compatible classes.

External accuracy:

**38.45%**

External confusion matrix:

| Actual / Predicted | Brown rust | Healthy | Yellow rust |
|---|---:|---:|---:|
| Brown rust | 74 | 53 | 1 |
| Healthy | 23 | 99 | 0 |
| Yellow rust | 126 | 98 | 15 |

The substantial difference between internal and external performance is one of the central findings of the extended project.

The external evaluation is a closed-set three-class experiment. It does not test the model's ability to recognise every possible wheat disease.

See [EXTERNAL_VALIDATION.md](docs/EXTERNAL_VALIDATION.md).

## Anomaly-detection result

A complementary convolutional autoencoder was trained only on healthy images and evaluated as an anomaly detector.

The experiment did not successfully separate disease images:

- Disease recall: **0**
- Disease F1-score: **0**
- ROC-AUC: **0.086**

This negative result is intentionally retained because it informed the later model-selection decision rather than being removed from the project.

See [RESULTS.md](docs/RESULTS.md).

## Further analysis

Detailed documentation is available in:

- [Failure and leakage analysis](docs/FAILURE_ANALYSIS.md)
- [External validation](docs/EXTERNAL_VALIDATION.md)
- [Baseline architecture comparison](docs/BASELINE_COMPARISON.md)
- [Grad-CAM faithfulness evaluation](docs/GRADCAM_FAITHFULNESS.md)
- [Implementation of external technical feedback](docs/EXTERNAL_REVIEW_IMPLEMENTATION.md)

## Dataset

### Original training and validation dataset

The original project uses the **Wheat Disease Detection / Wheat Rust Classification Dataset** published on Kaggle by `sinadunk23`:

https://www.kaggle.com/datasets/sinadunk23/behzad-safari-jalal

The complete dataset contains 3,679 images:

| Class | Train | Validation | Total |
|---|---:|---:|---:|
| Brown Rust | 902 | 226 | 1,128 |
| Healthy | 1,116 | 279 | 1,395 |
| Yellow Rust | 924 | 232 | 1,156 |
| **Total** | **2,942** | **737** | **3,679** |

The dataset is not redistributed in this repository. Download it from the source and confirm its current licence and terms before use.

Expected structure:

```text
data/
├── train/
│   ├── Brown_rust/
│   ├── Healthy/
│   └── Yellow_rust/
└── val/
    ├── Brown_rust/
    ├── Healthy/
    └── Yellow_rust/
```

### External validation dataset

External evaluation uses a compatible three-class subset of the **Wheat Disease Dataset - Small** released by researchers associated with the John Innes Centre.

The primary dataset record is:

**DOI: 10.5281/zenodo.7307816**

The external images are not redistributed in this repository.

Only Brown rust, Healthy, and Yellow rust images are used because these correspond to the classes supported by the current classifier.

No exact SHA-256 image overlap or perceptual-hash match at the selected threshold was detected between the original and external evaluation datasets.

This supports the absence of detected direct image overlap under these checks but does not establish complete statistical or source-level independence.

## Quick start

Create and activate a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Audit the original dataset:

```bash
python -m src.audit_dataset --data-dir data
```

Train the default MobileNetV2 classifier:

```bash
python -m src.train_classifier --data-dir data --epochs 10
```

Alternative architectures can be selected using:

```bash
python -m src.train_classifier --data-dir data --epochs 10 --architecture efficientnetb0
```

or:

```bash
python -m src.train_classifier --data-dir data --epochs 10 --architecture resnet50
```

Evaluate a saved classifier:

```bash
python -m src.evaluate --data-dir data --model models/wheat_rust_mobilenetv2.keras
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

Train the anomaly detector separately:

```bash
python -m src.train_anomaly --data-dir data --epochs 15
```

Additional experimental workflows are documented in the corresponding files under `docs/`.

## Methodology

### Classification

The classification pipeline supports three ImageNet-pretrained convolutional neural-network architectures:

- MobileNetV2
- EfficientNetB0
- ResNet50

Each architecture uses its architecture-appropriate preprocessing pipeline followed by a frozen pretrained backbone, global-average pooling, dropout, and a three-class softmax classification head.

The architecture comparison investigates whether changing the CNN architecture improves cross-dataset generalisation rather than simply maximising internal validation accuracy.

### External validation

The trained classifiers are evaluated on both the original internal validation split and a separate external-source wheat-disease dataset containing the same three compatible classes.

The external dataset is not used for training.

This allows internal validation performance to be compared with performance following a change in image source and distribution.

### Leakage and similarity analysis

Training and validation images were examined using:

- SHA-256 hashing for exact duplicate detection
- perceptual hashing for visually similar image detection

No exact cross-split duplicates were detected.

One near-duplicate train/validation pair was identified at the selected perceptual-hash threshold. Re-evaluation after excluding the affected validation image produced a negligible change in validation accuracy.

These checks reduce one potential source of evaluation bias but do not establish complete source-level or field-level independence.

### Failure analysis

Misclassified validation images are retained and analysed rather than reporting only aggregate performance metrics.

The analysis includes predicted classes, confidence values, class probabilities, and representative Grad-CAM visualisations.

External high-confidence errors are also examined because confident incorrect predictions can reveal weaknesses that overall accuracy alone does not show.

### Model explainability with Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to investigate image regions influencing MobileNetV2 predictions.

Gradients for the selected prediction are propagated to the MobileNetV2 `block_12_add` feature layer, producing a **14×14 activation map** that is resized and overlaid on the input image.

The visualisation indicates regions influencing the model's selected prediction.

Grad-CAM does **not** establish that highlighted regions correspond to biologically confirmed wheat-rust lesions.

### Quantitative Grad-CAM faithfulness

Grad-CAM is additionally evaluated using a perturbation-based faithfulness experiment.

For a deterministic balanced sample of validation images:

1. Grad-CAM is generated for the predicted class.
2. The highest-activation 20% of Grad-CAM pixels are masked.
3. Prediction confidence is recalculated.
4. Spatially coherent random regions covering approximately the same image fraction are independently masked.
5. The resulting confidence reductions are compared.

Under this protocol, Grad-CAM-guided masking produced a larger confidence reduction than the random-region baseline for **68.3%** of evaluated images.

The effect was strongly class-dependent and weakest for Healthy images.

This evaluates model faithfulness rather than biological localisation accuracy.

### Anomaly detection

The convolutional autoencoder is trained only on healthy training images.

A reconstruction-error threshold is calibrated using held-out healthy validation images. Brown Rust and Yellow Rust images are then treated as anomalous examples.

The experiment performed poorly at separating disease images and is retained as a documented negative result.

## Evaluation integrity

- Training and validation folders are kept separate.
- SHA-256 analysis detected no exact duplicate images across the original training and validation splits.
- Perceptual-hash analysis detected one near-duplicate pair at the selected threshold.
- Removing the affected validation image produced a negligible change in internal validation accuracy.
- No exact or perceptual-hash match at the selected threshold was detected between the original and external evaluation datasets.
- The external dataset is not used to train the evaluated classifiers.
- Internal and external results are reported separately rather than combining them.
- Baseline architectures are evaluated on the same internal and external evaluation tasks.
- Negative and mixed experimental findings are retained.
- The 98.78% result describes the rebuilt MobileNetV2 pipeline on the original internal validation split.
- The 96.88% result reported in the 2024 dissertation remains a separate historical result.
- Grad-CAM faithfulness does not establish biological localisation accuracy.
- Object localisation is not claimed as a trained capability because verified spatial annotations are not available.

## Repository layout

```text
app.py
README.md
requirements.txt

src/
├── analyze_failures.py
├── audit_dataset.py
├── data.py
├── evaluate.py
├── evaluate_external.py
├── evaluate_gradcam_faithfulness.py
├── gradcam.py
├── model.py
├── train_anomaly.py
└── train_classifier.py

docs/
├── BASELINE_COMPARISON.md
├── EXTERNAL_REVIEW_IMPLEMENTATION.md
├── EXTERNAL_VALIDATION.md
├── FAILURE_ANALYSIS.md
├── GRADCAM_FAITHFULNESS.md
├── RESULTS.md
└── WHEAT_RUST_AI_TECHNICAL_ARTICLE.pdf

models/
outputs/
```

Generated models and experimental outputs may be maintained locally rather than distributed with every repository version.

## Research limitations

- The original internal validation split comes from the same overall dataset as the training data.
- Dataset provenance and field conditions may affect generalisation.
- Image-level train/validation separation does not establish source-level or field-level independence.
- Exact-file and perceptual-hash checks cannot identify every possible form of visual or source relationship.
- External validation demonstrates a substantial cross-dataset performance gap.
- The external experiment is limited to three classes supported by the current classifier.
- The classifier may produce highly confident incorrect predictions.
- Architecture changes alone did not resolve the observed external generalisation problem.
- Grad-CAM indicates model-influential regions rather than biologically confirmed disease lesions.
- The current Grad-CAM faithfulness experiment uses 60 internal validation images and one masking fraction.
- The random-region control and mean-pixel replacement are artificial perturbations.
- Expert lesion annotations were not available for quantitative biological localisation evaluation.
- Reliable field deployment would require broader external datasets, expert assessment, calibration, prospective testing, and evaluation under realistic agricultural conditions.

## Research development

The project originated as an MSc Artificial Intelligence dissertation and has subsequently been extended through public development and external technical feedback.

The later research work includes:

- corrected evaluation;
- public reproducible code;
- interactive deployment;
- Grad-CAM explainability;
- failure-case investigation;
- leakage and near-duplicate analysis;
- independent-source external validation;
- comparison with additional CNN architectures;
- quantitative explainability evaluation; and
- a research focus on generalisation and robustness.

See [EXTERNAL_REVIEW_IMPLEMENTATION.md](docs/EXTERNAL_REVIEW_IMPLEMENTATION.md) for a transparent record of how external technical feedback informed subsequent experiments.

## Technical article and resources

- [Read the Wheat Rust AI technical article](docs/WHEAT_RUST_AI_TECHNICAL_ARTICLE.pdf)
- [Try the live Streamlit demonstration](https://devanandha-wheat-rust-ai.streamlit.app/)
- [Read the published LinkedIn technical article](https://www.linkedin.com/pulse/when-anomaly-detection-failed-lessons-from-building-wheat-vs-iotze/)
- [View the archived software record on Zenodo](https://doi.org/10.5281/zenodo.22726590)

## Responsible use

This software is intended for education and research.

Consult a qualified crop specialist before acting on a prediction. Do not use the system as the sole basis for crop-treatment decisions or pesticide recommendations.

## Citation

The software is archived through Zenodo.

Concept DOI:

**10.5281/zenodo.22726590**

The DOI above resolves to the software record and should be used when referring to the evolving project unless a specific archived version is required.

## Author

**Devanandha Vellaramkuzhiyil Shaji**  
MSc Artificial Intelligence, Ulster University

LinkedIn: https://www.linkedin.com/in/devanandhavs21

## Licence

This project is released under the MIT License. See [LICENSE](LICENSE).