# Grad-CAM Faithfulness Evaluation

## Objective

Following independent technical feedback recommending stronger evaluation of model explainability, a quantitative perturbation experiment was performed to assess whether Grad-CAM-highlighted regions meaningfully influence the Wheat Rust AI classifier's predictions.

This experiment evaluates **model faithfulness**, not biological localisation accuracy.

It asks:

> Does masking image regions identified as important by Grad-CAM reduce the model's confidence more than masking a comparable randomly located region?

If removing Grad-CAM-highlighted regions produces a larger confidence reduction than removing random regions, this provides evidence that the highlighted regions are influential to the model's prediction.

It does not establish that those regions correspond to biologically verified wheat-rust symptoms.

---

## Experimental Setup

The original MobileNetV2 wheat-rust classifier was evaluated using a deterministic balanced sample from the internal validation dataset.

The experiment used:

- 60 images in total
- 20 Brown rust images
- 20 Healthy images
- 20 Yellow rust images
- 20% of image pixels masked
- 5 random-region repetitions per image
- Random seed: 42

For each image:

1. The model's predicted class and original confidence were recorded.
2. Grad-CAM was generated for the predicted class.
3. The highest-activation 20% of Grad-CAM pixels were masked using the image's mean RGB value.
4. The model's confidence in the original predicted class was measured again.
5. Five spatially coherent random rectangular regions covering approximately the same image fraction were independently masked.
6. The mean confidence reduction from the random-region perturbations was calculated.
7. The Grad-CAM confidence reduction was compared with the random-region baseline.

The principal comparison is:

**Grad-CAM advantage = Grad-CAM confidence drop - mean random-region confidence drop**

A positive value means masking the Grad-CAM-selected region reduced prediction confidence more than the random-region baseline.

---

## Why a Spatial Random Baseline Was Used

An initial exploratory experiment used randomly scattered individual pixels as the comparison baseline.

That perturbation produced disproportionately large confidence changes, particularly for Yellow rust images, and was not considered a fair spatial comparison with Grad-CAM regions.

The primary experiment therefore uses spatially coherent random rectangular regions.

This methodological change is documented transparently rather than selecting results based solely on which experiment produced a more favourable outcome.

---

## Overall Results

| Metric | Result |
|---|---:|
| Images evaluated | 60 |
| Mean Grad-CAM confidence drop | 0.2091 |
| Median Grad-CAM confidence drop | 0.0288 |
| Mean random-region confidence drop | 0.0386 |
| Median random-region confidence drop | 0.0036 |
| Mean Grad-CAM advantage | +0.1705 |
| Grad-CAM drop greater than random baseline | 68.3% |

Across the complete balanced sample, masking Grad-CAM-highlighted regions reduced confidence by approximately 0.209 on average, compared with approximately 0.039 for spatially coherent random regions.

Grad-CAM produced a larger confidence reduction than the random-region baseline for approximately 68.3% of evaluated images.

---

## Class-Level Results

| Class | Mean Grad-CAM Drop | Mean Random Drop | Mean Advantage | Grad-CAM > Random |
|---|---:|---:|---:|---:|
| Brown rust | 0.1847 | 0.0766 | +0.1080 | 70% |
| Healthy | -0.0079 | 0.0075 | -0.0155 | 35% |
| Yellow rust | 0.4506 | 0.0316 | +0.4190 | 100% |

### Brown Rust

For Brown rust, Grad-CAM-guided masking produced a larger confidence reduction than random-region masking for 70% of sampled images.

The mean Grad-CAM confidence reduction was approximately 0.185 compared with approximately 0.077 for the random baseline.

This represents a positive mean Grad-CAM advantage of approximately 0.108.

### Healthy

Healthy images showed a substantially weaker pattern.

The mean Grad-CAM confidence change was approximately -0.008, while the mean random-region confidence change was approximately 0.008.

Grad-CAM produced a larger confidence reduction than the random baseline for only 35% of Healthy images.

This indicates that the current Grad-CAM perturbation result is not consistently informative for the Healthy class.

### Yellow Rust

Yellow rust showed the strongest perturbation response.

The mean confidence reduction after Grad-CAM masking was approximately 0.451 compared with approximately 0.032 for random-region masking.

Grad-CAM produced a larger confidence reduction than the random baseline for all 20 sampled Yellow rust images.

The mean Grad-CAM advantage was approximately +0.419.

---

## Interpretation

Under this perturbation protocol, the results provide evidence that Grad-CAM-highlighted regions are more influential to the model's predictions than randomly positioned spatial regions overall.

However, the effect is strongly class-dependent.

The result is strongest for Yellow rust, positive but more variable for Brown rust, and weak for Healthy images.

These differences reinforce the importance of evaluating explainability at class level rather than relying only on selected visual examples or a single aggregate metric.

---

## Relationship to External Validation

The faithfulness result should not be interpreted as evidence that the classifier generalises reliably.

Separate external validation demonstrated a substantial cross-dataset performance gap.

A model explanation can be faithful to the model's internal decision process while the underlying prediction itself is incorrect or poorly generalised.

Therefore:

- **classification performance** asks whether the prediction is correct;
- **external validation** asks whether performance transfers to a different data source;
- **Grad-CAM faithfulness** asks whether highlighted regions influence the model's decision;
- **biological localisation validation** would ask whether those highlighted regions correspond to expert-confirmed disease symptoms.

These are different evaluation questions.

---

## Limitations

This experiment has several important limitations:

- Only 60 internal validation images were evaluated.
- The evaluation uses a single 20% masking fraction.
- The random baseline uses rectangular regions, while Grad-CAM-selected pixels may have irregular spatial structure.
- Replacing pixels with the image mean introduces an artificial perturbation.
- Confidence reduction is only one measure of explanation faithfulness.
- The experiment evaluates the original MobileNetV2 classifier only.
- The experiment does not use expert lesion annotations.
- The experiment does not establish biological correctness of the highlighted regions.
- Results from the internal validation distribution should not be assumed to transfer to external datasets.

---

## Conclusion

The quantitative perturbation experiment provides initial evidence that Grad-CAM identifies model-influential regions for the Wheat Rust AI classifier, particularly for Yellow rust and, to a lesser extent, Brown rust.

The Healthy class does not show the same pattern and remains an important limitation.

These results strengthen the explainability analysis beyond qualitative visual inspection while also demonstrating that model faithfulness and biological correctness are separate questions.

Expert-annotated disease regions would be required for a stronger evaluation of whether Grad-CAM highlights biologically meaningful wheat-rust symptoms.

---

## Future Work

Future explainability evaluation could include:

- expert-annotated disease regions;
- lesion masks or bounding-box annotations;
- multiple masking fractions;
- insertion and deletion curves;
- additional attribution methods;
- evaluation on external-source images;
- comparison of explanation behaviour across classifier architectures.

These extensions would help distinguish model faithfulness from biologically meaningful localisation and provide a stronger assessment of explainability under dataset shift.