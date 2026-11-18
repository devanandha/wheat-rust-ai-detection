# Study S1 — Controlled Test of Acquisition-Related Shortcut Learning

**Draft experimental specification · 18 November 2026**  
**Status:** Prospective plan for new model training and independent field evaluation. No S1 model has been trained, no new field test has been run, and this is **not** a preregistered study. Decisions below must be reviewed and frozen before the independent field test is unsealed.

## Exact scientific question

**H1:** Does training with a class-independent resolution/interpolation variation policy reduce false-Healthy rust classifications on *independent, expert-labeled field data*, compared to otherwise identical MobileNetV2 training?

This tests a **proposed learned acquisition-related shortcut**, not whether metadata width/height is explicitly passed to the model. The retrospective evidence is observational and strongly confounded by source/class/image curation; H1 is **not established**.

## Historical evidence, clearly marked exploratory

- Saved baseline MobileNetV2: original validation 728/737 (98.78%); previously inspected JIC 188/489 (38.45%), with Yellow Rust 15/239 and 151 diseased images called Healthy.
- Original training has class-associated native image-size distributions: at least one side <224 pixels in 72.6% Brown Rust, 1.2% Healthy, 94.9% Yellow Rust. JIC: 0% across compatible classes.
- A single frozen-model 224→112→224 perturbation changed 29/737 original and 154/489 JIC predictions, with opposite aggregate accuracy movements. This tests input sensitivity, **not causality** or improvement through training.
- At JIC score ≥0.99, 55/83 predictions were wrong; confidence-only filtering failed as a safety gate.
- The original, JIC, IARI and Ethiopia datasets have been repeatedly examined; JIC is **not a blind test**, and IARI/Ethiopia have source–class confounding. Preserve their role as development/retrospective research evidence only.

## Intervention arms (preliminary; final decisions to be frozen)

| Arm | Data | Architecture and supervision | Only intended difference |
|---|---|---|---|
| A — matched control | Same licensed, fixed original training data and any permitted field-development data | ImageNet-pretrained frozen MobileNetV2, trainable 3-class head | Reproduce standard 224×224 processing and existing augmentation |
| B — acquisition variation | Exactly same images, class labels, train/val groups and batch/epoch budget as A | Same frozen MobileNetV2 and classifier head | During training only, apply a **class-blind** spatial-resolution/interpolation transformation from a **fixed** predeclared distribution; no metadata-based class conditional transforms |

**Illustrative B policy to evaluate and freeze before running:** With probability 0.5 per training image, downsample RGB after 224×224 standard loading to a size uniformly drawn from {112, 160, 192}, then bilinearly upsample to 224×224; otherwise leave unchanged. Apply regardless of disease class. Keep pretrained MobileNetV2 `preprocess_input` only once and inside the saved model. This policy will likely blur symptoms; prior to definitive training, have a qualified reviewer examine representative transformed **development images** to confirm the disease cue remains interpretable. If it destroys diagnostic information, redesign using development data and transparently amend/version the plan **before** opening final test; do not quietly tune parameters on JIC.

This first study intentionally excludes additional colour augmentation or background cropping: introducing multiple simultaneous changes would obscure the mechanism. A separate subsequent hypothesis may examine saturation using otherwise matched arms.

## Equal-treatment controls and reproducibility

- **Seeds:** 42, 43, 44 for both A and B; pair outcomes by seed. Three repeats are an **initial variance check**, not a large-sample significance analysis.
- **Same model:** MobileNetV2 frozen ImageNet base, 224×224 0–255 input and built-in preprocessing, 3-class head, same optimiser, learning rate schedule, initialisation procedure, 10-epoch maximum, early-stopping validation and batch size for both arms. Select one common batch size **before execution**, and archive exact configurations. No change to class distribution, augmentation except specified intervention, test data or scoring thresholds.
- **Training groups:** Use image group IDs where available; if the historical original source lacks plant/field identifiers, explain its residual leakage risk. Do not claim plant independence for original `train/val` merely from SHA-256 tests.
- **Outcome reports for each seed:** test counts per class; macro F1, balanced accuracy, per-class recall/precision, three-class confusion matrix; especially Yellow Rust recall and diseased→Healthy wrong predictions. Do not report only overall accuracy.
- **Comparison:** paired deltas B−A for each seed and group, means/dispersion, and class-level negative effects. If enough independent fields are available, use field-cluster bootstrap confidence intervals; otherwise mark sample too small for such inference.
- **Reproducibility capsule:** model/data hashes where redistribution is permitted, image/label manifests, random seeds, library versions, exact transforms, training histories, outputs and failure cases. No copyrighted raw data uploaded to GitHub without permission.

## Target primary and secondary evaluation

**Primary outcome, locked independent field test:** *false-Healthy rate among true Brown Rust + Yellow Rust samples*, counting any true rust image assigned Healthy as an error. The metric has a denominator of all independently labeled rust images, **not only accepted images**. For any triage policy, report the number deferred separately; an image deferred to expert review is never silently counted as a correct diagnosis.

**Key secondary outcomes:** Yellow Rust recall, Brown Rust recall, balanced accuracy/macro F1, class-wise confusion, accuracy, rates of unrecognised/mixed/other-image misclassification, and study-specific expert referral burden. Record per-site and per-device outcomes and the spread across random seeds.

**No success threshold yet:** A clinically/agronomically meaningful minimum recall, maximum false reassurance, and sample-size/power plan require agreement with a wheat plant-pathology collaborator. Do not retrofit thresholds after seeing the field outcomes.

## Independent site/field protocol (must precede opening final test)

- Acquire lawful access to images and expert-labelled disease cases; keep original files unmodified and store consent/provenance. In addition to known classes include lookalikes, other fungal diseases and mixed symptoms for separately reported out-of-scope/rejection evaluation.
- Assign stable `site_id`, `field_id`, `plant_id`, `session_id`, `device_id` and, where available, year/crop stage. Keep all images of one plant/field/session or one sequence on the same side of every split; ideally hold out entire fields, sites and season.
- Separate **development**, **calibration** and **final independent test** images by groups. Record decisions, file hashes and split manifest before any model choice; never fit thresholds or transformations against the final test.
- Have a qualified wheat-pathology reviewer set reference-label conventions, disagreements/adjudication and exclusions. Inconclusive reference cases must not be relabelled as confidently Healthy.
- Freeze training arm definitions, thresholds, primary outcome, analysis code and final-test site selection before final evaluation. If independent field samples cannot be obtained, report all training and JIC analyses as **exploratory** and do not claim field-level improvement.

## Falsification, interpretation gates and stop conditions

- **Support for intervention utility:** reproducible B-vs-A improvements with a pre-specified meaningful reduction of the primary failure metric on a genuinely independent field test **without unacceptable declines in any rust class**, reported with uncertainty.
- **Evidence about mechanism:** different outcomes between controlled resolution-specific interventions and matched controls, ideally supported by representative image/pixel-level analysis and source metadata. An accuracy increase alone does not prove the historical model learned a resolution shortcut.
- **Null/negative results:** if B does not improve held-out-site performance, makes Brown Rust worse, or worsens false-Healthy errors, preserve and report the finding; do not cherry-pick seed or data source.
- **Safety stop:** until separately validated with qualified domain reviewers, the public and research apps must not provide autonomous crop diagnoses, pesticide suggestions, or rely on score ≥0.99 for safety.

## Immediate next actions

1. Obtain external critique of this draft before claiming novelty; refine the study hypothesis against existing 2024–2026 wheat-specific shortcut-learning and field-image literature.
2. Seek a wheat-pathology collaborator to advise on label ontology, field capture, expert adjudication, ethical access, and measurable safety targets.
3. Inventory existing dataset terms and group metadata **without** using any potentially untouched test to optimise.
4. Build code for **A/B with deterministic seeds and fixed image manifests**, then test on synthetic images and permitted development data. Independent test is off limits during model selection.
5. Freeze version S1.0 and execute; record every run, including failures.

**No source-image license, newly collected data, external expert agreement, model improvement, field readiness, or scientific novelty is established by this proposal.**
