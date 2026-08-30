# Experimental results

## Dataset audit

- 2,942 training images and 737 validation images
- Three classes: Brown Rust, Healthy, and Yellow Rust
- No corrupt files found
- No exact duplicate files found within or across the supplied splits
- 3,627 unique image dimensions, indicating substantial variation in source resolution

## MobileNetV2 classifier

The classifier completed 10 epochs. Training accuracy increased from 84.50% to 98.98%, while validation accuracy increased from 96.47% to 98.78%. Final validation loss was 0.0410.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Brown Rust | 0.9865 | 0.9735 | 0.9800 | 226 |
| Healthy | 0.9893 | 0.9964 | 0.9929 | 279 |
| Yellow Rust | 0.9871 | 0.9914 | 0.9892 | 232 |

The confusion matrix contains nine errors:

- Three Brown Rust images classified as Healthy
- Three Brown Rust images classified as Yellow Rust
- One Healthy image classified as Brown Rust
- Two Yellow Rust images classified as Brown Rust

![Classifier confusion matrix](../outputs/classifier/confusion_matrix.png)

## Autoencoder anomaly experiment

The healthy-only convolutional autoencoder reduced healthy validation reconstruction loss to 0.0113, but reconstruction error did not distinguish disease images successfully.

| Metric | Result |
|---|---:|
| Accuracy | 0.3596 |
| Disease precision | 0.0000 |
| Disease recall | 0.0000 |
| Disease F1-score | 0.0000 |
| ROC-AUC | 0.0863 |

The disease images were generally reconstructed with lower rather than higher error. Therefore, the fixed high-error anomaly rule failed. The experiment is reported as a negative result and the autoencoder is not used in the application.

## Interpretation

For this labelled dataset, supervised transfer learning was markedly more suitable than pixel-reconstruction anomaly detection. The result does not establish real-world diagnostic accuracy. External field images, source-grouped splitting, expert review, and prospective validation are required before practical use.
