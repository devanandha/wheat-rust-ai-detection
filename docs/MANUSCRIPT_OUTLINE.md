# Manuscript Outline

## Working title

**High Internal Accuracy, Weak External Generalisation: A Reproducible Evaluation of Deep Learning for Wheat-Rust Classification**

## Central research question

How well do deep-learning models for wheat-rust classification generalise beyond their training distribution, and can explainability and failure-case analysis reveal the visual factors underlying model errors?

## Planned structure

1. Abstract
2. Introduction
3. Related work
4. Materials and methods
   - Original and external datasets
   - Preprocessing and class mapping
   - Autoencoder anomaly-detection experiment
   - MobileNetV2, EfficientNetB0 and ResNet50 classifiers
   - Internal and external evaluation
   - Duplicate and near-duplicate checks
   - Failure analysis
   - Grad-CAM faithfulness protocol
5. Results
   - Anomaly-detection result
   - Internal validation
   - External generalisation
   - Architecture comparison
   - Leakage audit
   - Quantitative Grad-CAM faithfulness
6. Discussion
7. Limitations and threats to validity
8. Conclusion
9. Data and code availability
10. Ethics, competing interests and author contributions

## Paper claim boundary

The paper may claim evidence of a cross-dataset generalisation gap and class-dependent Grad-CAM faithfulness under the stated protocol. It must not claim field-ready diagnostic performance, biological lesion localisation, clinical/agronomic validation, or broad real-world generalisation.
