# Classifier Baseline Comparison

## Objective

Following independent technical feedback, additional classifier architectures were evaluated to determine whether the strong internal validation performance of the original MobileNetV2 model generalises across architectures and, more importantly, to an external-source dataset.

The comparison focuses on three ImageNet-pretrained convolutional neural network architectures:

- MobileNetV2
- EfficientNetB0
- ResNet50

The models use the same three-class classification task:

- Brown rust
- Healthy
- Yellow rust

The external dataset was kept separate from model training and was used only for evaluation.

---

## Internal Validation Results

The internal validation set contains 737 images.

| Architecture | Internal Accuracy |
|---|---:|
| MobileNetV2 | 98.78% |
| EfficientNetB0 | 99.05% |
| ResNet50 | 99.59% |

All three architectures achieved very high performance on the internal validation set.

ResNet50 achieved the highest internal accuracy at 99.59%, followed by EfficientNetB0 at 99.05% and MobileNetV2 at 98.78%.

---

## External Validation Results

The external evaluation subset contains 489 images:

- Brown rust: 128
- Healthy: 122
- Yellow rust: 239

| Architecture | External Accuracy |
|---|---:|
| MobileNetV2 | 38.45% |
| EfficientNetB0 | 41.31% |
| ResNet50 | 38.04% |

Despite all three models achieving approximately 99% accuracy internally, none reproduced this performance on the external dataset.

EfficientNetB0 achieved the highest external accuracy at 41.31%, although this remained substantially below its internal performance.

---

## Internal vs External Performance

| Architecture | Internal Accuracy | External Accuracy | Accuracy Drop |
|---|---:|---:|---:|
| MobileNetV2 | 98.78% | 38.45% | 60.33 percentage points |
| EfficientNetB0 | 99.05% | 41.31% | 57.74 percentage points |
| ResNet50 | 99.59% | 38.04% | 61.55 percentage points |

The results demonstrate that improving performance on the original validation distribution did not necessarily improve cross-dataset generalisation.

ResNet50 provides the clearest example. It achieved the highest internal accuracy (99.59%) but the lowest external accuracy (38.04%) of the three architectures.

---

## External Class-Level Behaviour

### MobileNetV2

External confusion matrix:

| Actual \ Predicted | Brown rust | Healthy | Yellow rust |
|---|---:|---:|---:|
| Brown rust | 74 | 53 | 1 |
| Healthy | 23 | 99 | 0 |
| Yellow rust | 126 | 98 | 15 |

Yellow rust recall was approximately 6.28% (15/239).

### EfficientNetB0

External confusion matrix:

| Actual \ Predicted | Brown rust | Healthy | Yellow rust |
|---|---:|---:|---:|
| Brown rust | 93 | 35 | 0 |
| Healthy | 41 | 78 | 3 |
| Yellow rust | 137 | 71 | 31 |

Yellow rust recall improved to approximately 12.97% (31/239), but remained low.

### ResNet50

External confusion matrix:

| Actual \ Predicted | Brown rust | Healthy | Yellow rust |
|---|---:|---:|---:|
| Brown rust | 67 | 61 | 0 |
| Healthy | 2 | 118 | 2 |
| Yellow rust | 105 | 133 | 1 |

Yellow rust recall fell to approximately 0.42% (1/239).

---

## Interpretation

The baseline comparison suggests that the external generalisation problem is not resolved simply by replacing MobileNetV2 with another established CNN architecture.

EfficientNetB0 produced a modest improvement in external accuracy, but a substantial internal-to-external performance gap remained across all three models.

Most importantly, higher internal validation accuracy was not associated with better external performance. ResNet50 achieved the strongest internal result while performing slightly worse externally than MobileNetV2.

These findings motivate further investigation of dataset and domain differences, class-specific behaviour, robustness, and model attention rather than relying solely on internal validation accuracy or architecture changes.

The results do not, by themselves, establish the exact cause of the performance degradation.

---

## ResNet50 Training Note

ResNet50 training was performed with a batch size of 8 because of local CPU/memory constraints.

Training completed four full epochs successfully. During epoch 5, TensorFlow terminated with a memory-allocation error.

The best checkpoint from the completed epochs had already been saved through model checkpointing and was used for subsequent internal and external evaluation.

The evaluated checkpoint achieved:

- 99.59% internal validation accuracy
- 38.04% external validation accuracy

This resource limitation is documented for reproducibility and should not be interpreted as evidence about the comparative generalisation performance of the architecture itself.

---

## Research Implication

The results strengthen the project's focus from maximising validation accuracy toward investigating generalisation and robustness.

A central research question emerging from these experiments is:

> How well do deep-learning models for wheat-rust classification generalise beyond their training distribution, and can explainability and failure-case analysis reveal the visual factors underlying model errors?

The baseline comparison provides evidence that strong performance within the original validation distribution is insufficient on its own to demonstrate robust performance on images from a different source.

---

## Limitations

- Only three CNN architectures were evaluated.
- Hyperparameter optimisation was not performed independently for every architecture.
- The external evaluation is a closed-set three-class comparison and does not evaluate recognition of other wheat diseases.
- Differences between datasets may include acquisition conditions, backgrounds, disease presentation, image composition, and other factors that have not yet been isolated experimentally.
- External evaluation performance should therefore be interpreted as evidence of a cross-dataset generalisation gap rather than proof of a specific causal mechanism.