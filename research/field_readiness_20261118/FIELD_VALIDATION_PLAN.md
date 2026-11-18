# Wheat Rust AI — Field-Readiness Research and Pilot Protocol

**Created:** 18 November 2026  
**Status:** Proposed field-validation protocol. **No field pilot has begun, no field images have been newly collected, and the existing model is not field validated.**  
**Application:** Decision-support *research assistant*, NOT autonomous crop diagnosis or fungicide guidance.

## Executive decision

The existing MobileNetV2 model is **NOT ready for farmer-facing autonomous diagnosis**. It achieved 728/737 correct (98.78%) on the original internal validation split, but only 188/489 (38.45%) on the already-inspected JIC external images. Its JIC Yellow Rust recall was 15/239 (6.3%). Controlled inference-time downsample/upsample perturbed model predictions and raised exploratory JIC accuracy to 233/489 (47.65%), but did not establish a robust or causal remedy. **Never promote the 47.65% to validated deployed performance.**

**Proposal:** Build a mobile-friendly, expert-in-the-loop *shadow-mode* field study: collect photos under real conditions; record provenance and expert reference labels; evaluate model errors, abstention and failure modes; then—if independent validation is satisfactory—consider a closely supervised pilot. UI implementation by itself is not a research contribution or evidence of readiness.

## Research question and pre-registered hypotheses

**Primary question:** Under realistic wheat-field acquisition changes, can a pre-specified image-quality / reliability-gating workflow lower the **rate of unsafe wrong diagnoses on accepted images**, at a transparently reported review/abstention cost, compared with always predicting one of three classes?

Hypotheses to verify (not claimed results):

- **H1 — acquisition:** The source-resolution/colour differences found in the original dataset remain associated with prediction errors when assessed under prespecified, controlled perturbations and/or newly collected field imagery.
- **H2 — operational reliability:** A separately calibrated confidence/uncertainty or quality-based review policy may reduce errors among answered predictions. Evaluate **risk versus coverage**; do not presume maximum-softmax confidence is well calibrated or that all high-confidence mistakes can be filtered.
- **H3 — generalisation:** Training-time policies defined **before touching the new final test**—e.g. controlled scale/interpolation augmentation, photometric augmentation, device/source-balanced sampling—could improve held-out-site field performance without unacceptable degradation for any disease class.

The project must explicitly test null and adverse outcomes. The field pilot should quantify a limitation even if none of the tested mitigations works.

## Allowed and disallowed use

**Supported label vocabulary for the old classifier:** `Healthy`, `Brown_rust` (leaf rust), `Yellow_rust` (stripe rust). Existing model is *closed-set*. It cannot diagnose stem rust, septoria, powdery mildew, nutrient stress, insect damage, mixed infections, non-wheat leaves, or stages of severity.

**Pilot UI:** “Research analysis only — not a diagnosis.” Expert confirmation and appropriate agronomic channels are required. If in doubt, show “**Needs expert review / unable to assess**”, not “healthy”. **Do not give fungicide names, doses, or treatment recommendations based on the model.** Avoid converting confidence 0.9 into “90% likelihood of being correct”.

**Shadow mode:** The expert/farmer decision is made without reliance on the AI output. Model results are logged for retrospective comparison, not delivered as actionable advice. Switching to supervised feedback requires an independent readiness review.

## Work packages

### WP1 — Scope, labeling and stakeholder validation

- Engage an appropriately qualified plant-pathology expert or agricultural extension partner before collecting/labeling field samples.
- Agree a reference-label protocol; for uncertain cases use `uncertain_unconfirmed`, and for non-target/mixed diseases use `other_or_mixed` rather than forcing them into the model's three classes.
- Define the desired **decision**: triage to expert review vs autonomous prediction. Our first deployment goal is **triage/research monitoring**, not a fully automatic diagnosis.
- Secure participant/site permission, consent for image storage/sharing and explicit licensing. Do not assume farmer-submitted photos are public data.

### WP2 — Field image acquisition and provenance

**Suggested initial feasibility pilot (planning target, not a statistical claim):** approximately 60–120 photos spanning several **distinct fields and devices**, recorded as multiple capture sessions. This is a feasibility pilot, **not** a sufficiently powered final safety or accuracy evaluation. Seek wheat-region collaborators in Punjab/Haryana/Uttar Pradesh or other suitable wheat-growing regions rather than assuming proximity to Kerala gives an appropriate wheat field population.

At acquisition:
1. Record `site_id`, `field_id`, `plant_id` (if feasible), `session_id`, `capture_date`, `device_id`, capture distance/lighting, stage and variety when known.
2. Take a clear close-up of symptoms **and** a context photo, keeping related images in the **same evaluation group**. Do not treat two photos of one plant as independent replicates.
3. Capture real complexity: natural backgrounds, shadows, varied phones, healthy plants, mild and severe symptoms, multiple leaves, plausible lookalike conditions.
4. Record condition/provenance and retain original photos **unchanged**. Generate resized inputs in code. Calculate SHA-256 file hashes and image pHash for duplicate audits.
5. De-identify before external release (faces, phone metadata, precise farm coordinates) and retain only location detail needed for scientific evaluation and participant permissions.

**Labeling:** at least one qualified reviewer; for ambiguous or important cases request a second independent reviewer/adjudication; record whether microscopy or other confirmation exists. A model-predicted class is never the reference label.

### WP3 — Data partitions and novelty protection

- Freeze a **final independent test set before model development** with entire fields/farms/sites/dates/capture groups excluded from training. Prefer held-out geographic sites and crop seasons for stronger transfer claims.
- Use separate development, calibration and final test partitions, with all photos from a plant/session/field in the same split; do not select an image-level random split when this causes group leakage.
- The **original 737 validation** and **JIC 489** datasets have already been inspected many times. Label them **exploratory**; they must not be described as new blind evaluations.
- Keep `original_train` and candidate field-development images outside final test, and do not select thresholds/interventions based on final-test performance.
- Record a dataset provenance table, crop stage, expert label uncertainty, duplicates, class coverage, device/site counts and excluded examples. Public releases require license/consent.

### WP4 — Model and reliability experiments

**Benchmark fixed reference first.** The original frozen MobileNetV2 checkpoint and its 224 x 224 loader/embedded preprocessing are the reference. Record SHA-256, TensorFlow version, class-order mapping, preprocessing, per-image predictions and metrics.

Predefine a small experiment matrix (suggested for initial exploratory development only):

| Arm | Exact change relative to reference | What it tests | Interpretation caveat |
|---|---|---|---|
| A | Original frozen checkpoint | Baseline reliability, rejection behavior, class collapse | No change; currently fails JIC |
| B | Predefined scale/interpolation and colour augmentation **during training** | Whether diversified acquisition features change transfer performance | Must match architecture, seeds, splits, data and compute |
| C | Consistent crop/leaf-region detection before classifying (only if segmentation labels / method available) | Effect of context reduction | Masks can remove diagnostic features |
| D | Reference or best *prespecified* trained model with calibrated review/abstention policy | Accuracy versus coverage and unsafe false reassurance | Confidence-based abstention can remain overconfident under shift |

**Do not use an inference-time 112 px perturbation as a deployed “fix” simply because it raised JIC accuracy.** It is a single exploratory probe and worsened original accuracy and JIC Brown Rust recall.

Training arms should report repeated seeds, fixed data budgets and class-wise uncertainty intervals. No indiscriminate grid-search of JIC to maximise reported scores.

### WP5 — Deployment-oriented metrics and decision gates

Report, for every untouched site/test group:
- Confusion matrix and **per-class precision/recall**, particularly **Yellow Rust recall** and diseased → “Healthy” misclassifications (false reassurance).
- Macro F1 and balanced accuracy; class-prevalence/label-definition changes.
- Reliability diagram, expected calibration error (with bin definition), Brier score; score distribution of confident wrong predictions.
- Risk-versus-coverage curves for any review/abstention policy; proportion routed to expert; errors among accepted predictions.
- Unknown/non-target image rejection performance, including mixed/other conditions, and invalid/non-wheat/photo-quality failure rate.
- Operational measurements: capture failure rate, inference latency on real target phone, bandwidth/offline behavior, user time to complete, cost and privacy compliance.
- Uncertainty intervals and field/site breakdown; comparisons without an adequate final sample are **preliminary**.

**Provisional pilot gates requiring expert agreement, not universal research standards:** No false autonomous diagnosis to farmers; safe fallback for unsupported images, reviewable audit trail, verified stable image-processing/inference pipeline, adequate annotation quality, and pre-registered outcome thresholds plus sample-size plan *before* the final evaluation. If field recall or false reassurance is unacceptable, preserve the negative result and continue **shadow mode**.

### WP6 — Prototype implementation

**Phase A, research prototype:** Extend the existing Streamlit demo on a **separate experimental branch**. UI: upload/capture image; quality flag (“too blurry / cannot assess”); optional structured capture metadata; prediction visible only to approved researchers; label as uncalibrated; “expert review required”; capture expert ground-truth feedback in a separately secured datastore; researcher export with consent. Do not change the live public-facing demo into a “field-ready” diagnosis tool without evaluation.

**Phase B, optional Android/offline app after field-validation:** Evaluate converting the verified Keras model to TensorFlow Lite/LiteRT and compare numerical outputs with the original Keras model **on the same images** including class order and 0–255 input scale. Benchmark actual model size, speed, battery, offline usability, and quantized-model accuracy per class. A conversion is not evidence of field reliability.

**Phase C, prospective supervised pilot:** Allow experts to review and intervene; record decisions and feedback, monitor source/device drift, maintain versioned datasets, models and protocols.

### Suggested recording schema

Use the companion [`FIELD_IMAGE_LOG_TEMPLATE.csv`](FIELD_IMAGE_LOG_TEMPLATE.csv) as a **blank header-only template**. Each photo has a unique image ID and associated cluster identifiers; avoid public personally identifying farm locations. Keep image path, original camera image and reviewer provenance in a controlled repository, not public GitHub.

For each image record `label_status` (`unreviewed`, `confirmed`, `uncertain`), reference `expert_label` and `reviewer_code`, plus optional `review_method`. Do **not** put uncertain labels into model-performance denominators as known ground truth. Unknown classes are used to test abstention/rejection, not forced into the closed-set classifier's three known classes.

## Proposed milestones and deliverables

| Gate | Deliverable | Acceptance evidence |
|---|---|---|
| G0 | Defined clinical/agronomic scope and independent collaboration plan | Documented target users, risks, reviewers, consent, dataset permissions |
| G1 | Provenance-aware acquisition pilot | Expert-labeled photos and metadata across sites/devices, no unconsented publication |
| G2 | Locked group-aware split and fixed scoring code | Split manifest, duplicate audit, model hash, baseline reproduction |
| G3 | Controlled training and abstention comparisons | Repeated seeds, per-class results, risk–coverage curves, failures reported |
| G4 | Sealed external site evaluation | Independent field-specific accuracy/recall with uncertainty; no tuning after unsealing |
| G5 | Research app demonstration | Non-diagnostic shadow-mode tool, audit logs and review workflow |
| G6 | Field pilot decision | Joint technical/agronomic safety review before any farmer-facing recommendation |

A G5 demo **can be developed before G4**, but it must remain clearly marked research-only and should not claim field diagnostic reliability.

## Prior-art check: what is not novel

- A **May 2025 Frontiers in Plant Science paper** developed a wheat-rust severity Android application with photo upload and server inference: https://doi.org/10.3389/fpls.2025.1540642
- A **May 2026 Frontiers in Plant Science paper** studied cross-domain plant-disease reliability using acquisition statistics, calibration, selective prediction, OOD rejection and domain adaptation. Our earlier descriptive saturation, context and reliability ideas overlap significantly; do not claim these methods are new: https://doi.org/10.3389/fpls.2026.1826962
- A **2026 wheat-disease study** describes expert-informed field labeling and grouping image collection batches to reduce overlap: https://doi.org/10.3389/fpls.2026.1838344
- For mobile conversion, the TensorFlow Lite/LiteRT model converter is documented at https://developers.google.com/edge/litert/conversion/tensorflow/convert_tf
- Wheat cultivation in India is concentrated in major northern/central growing states: https://new1.isro.gov.in/Wheat_Crop_Area_and_Production_Estimation_in_India.html

**Potential distinctive contribution (conditional):** A transparent, wheat-specific, expert-labeled, group-independent evaluation of **when three-class disease recognition should be deferred**, including per-class false reassurance and acquisition effects, with complete negative results and mobile workflow measures. Novelty is **not established** until the precise proposed experiment is compared with these and other papers and independent findings are obtained.

## Immediate next actions

1. Identify a qualified wheat pathology collaborator and ask for feasible field samples plus labeling/adjudication support; **do not yet send any emails or claim a collaboration**.
2. Select a pilot scope: research staff only; wheat leaves; brown/stripe rust plus healthy and unknown/other.
3. Prepare the consent/permission and reference labeling form; do not publish farmer images or coordinates.
4. On a research branch, build the **non-diagnostic intake + expert review workflow**, separate from the live demo.
5. Prepare a frozen evaluation harness that prevents group leakage and reports error/coverage and false reassurance; determine final sample size with domain experts.
6. Freeze hypotheses and final test selection before examining prospective field test labels or tuning a model.

**Bottom line:** The next contribution should be **field evidence and documented reliability**, not a better benchmark number or merely an app screen.
