# Wheat Rust AI: From Shortcut Diagnosis to Trustworthy Field Assessment

**Research direction and execution gates — 18 November 2026**  
**Status:** Proposed research agenda. Not a preregistered study, new model, field-validation result, peer-reviewed paper or claim of scientific novelty. No training, evaluation, data downloads, or existing model files are changed by this document.

## Proposed research title

**Beyond Accuracy: Investigating Acquisition-Related Shortcuts and Safe Expert Triage in Wheat Rust Recognition**

**Primary scientific question:** *Do source-image acquisition and preprocessing cues contribute causally to the cross-source class-prediction collapse observed in Wheat Rust AI, and can prespecified training/data interventions reduce these errors on field-independent images without unsafe false reassurance?*

**Secondary operational question:** *When presented with unseen field images, can a pre-specified expert-review policy lower wrong predictions, especially Brown/Yellow Rust called Healthy, at a measurable referral cost?*

The goal is reproducible evidence **including negative findings**, not a particular score, commercial readiness, proof of biological causation from saliency alone, or an assurance of publication.

## What is already established (retrospective; reused images)

- Frozen MobileNetV2: **728/737 (98.78%) original validation** versus **188/489 (38.45%) JIC**, exact reproduction with the known saved checkpoint.
- JIC Yellow Rust: **15/239** correct, and **98/239** assigned Healthy. Across both JIC rust classes **151 diseased images were classified Healthy**.
- 4,168-image acquisition-statistics audit showed original training/validation have strong class-associated source-image size and saturation characteristics; **no direct evidence yet that MobileNetV2 relies on the source-size metadata**.
- The 224→112→224 artificial downsample/upsample caused 29/737 original and 154/489 JIC prediction changes; JIC correct count rose 188→233, original declined 728→707. Not a remedy or causal identification of acquisition shortcuts.
- JIC confidence audit: **135/233** accepted predictions at maximum score ≥0.90 were wrong; **55/83** at ≥0.99 were wrong. Therefore do **not** use “high score => trustworthy” as a deployment rule.
- JIC and original `val` sets have been repeatedly inspected; **no new blind evaluation** should be claimed from any further analysis on them.

See [acquisition findings](../acquisition_shift_20261118/EXPLORATORY_FINDINGS.md), [resolution results](../acquisition_shift_20261118/RESOLUTION_SENSITIVITY_RESULTS.md) and [reliability results](CONFIDENCE_RELIABILITY_RESULTS.md).

## What would constitute a strong *new* contribution?

A scientifically compelling result would consist of all of:

1. **Mechanistic test**, not correlation: create controlled variations of *the same* image, altering only predefined scaling/interpolation or colour/context factors; compare paired predictions and localize class-specific effects. Keep leaf symptoms visible and check that transformations do not invalidate the reference label. Artificial transformations test inference sensitivity; proving **learned shortcut reliance** requires controlled retraining/ablation on fixed data.
2. **Controlled training intervention:** evaluate the reference architecture under the same class balance, split, sample budget, seeds and evaluation code with a prespecified acquisition-invariance policy (e.g. source-size-balanced crop/resample pipeline and/or scale/interpolation augmentation). Include a no-change baseline and multiple seeds. Predefine which comparison addresses which hypothesis; avoid trying many changes and reporting the best JIC result.
3. **Provenance-aware grouping:** keep images of the same leaf, plant, field, session, device batch or acquisition sequence together. If the original source does not contain these IDs, acknowledge unresolved independence; use a newly collected or appropriately documented dataset for the definitive evaluation.
4. **True independent field evaluation:** expert-agreed label conventions, multiple sites/devices, known and unknown/mixed conditions, a sealed final set from held-out fields/sites/seasons, and all model/threshold decisions locked before its opening.
5. **Operational failure characterization:** diseased→Healthy errors (false reassurance), Yellow Rust recall, balanced accuracy/macro F1, confusion matrices, calibration, risk–coverage and percentage routed to review, with confidence intervals and cluster-level analyses.
6. **External critique and reproducibility:** full protocol, transparent negative results, dataset and model cards, runnable scripts, hashes, environment/version lock, independent expert label review, and a paper/report submitted only when evidence supports its claims.

**Null result is valid:** If experiments show apparent shortcut mitigation does not generalize, document that limitation. Strong research can reveal why intuitive fixes fail.

## Work sequence and decision gates

| Stage | Next deliverable | Must demonstrate before moving on |
|---|---|---|
| **A — Problem definition** | One-page hypothesis sheet, label ontology (including unknown/mixed/uncertain), predeclared primary outcome and comparison table | A clear claim not already established in literature; all existing JIC results labelled exploratory |
| **B — Data and collaborators** | Dataset provenance inventory; qualified plant-pathology contact; image-permission plan; group identifiers | Legal access, realistic label mapping and a credible independent field-evaluation route |
| **C — Mechanism test** | Paired perturbation + controlled retraining ablation, fixed seeds and sample budgets | Class-wise outcomes with uncertainty, evidence on possible shortcut reliance beyond image-size correlations |
| **D — Independent test** | One locked field/site/season test evaluation with expert labels | No prior model/threshold tuning on test; uncertainty intervals and per-class failure analysis reported |
| **E — Triage design** | Evidence-based research-only review workflow | Risk versus coverage, false-Healthy rates, unknown-image rejection, and review cost compared to no-gating baseline |
| **F — Communication** | Research paper + reproducibility capsule + short visual case study + responsible local app demo | Claims supported by actual results, code and domain-expert review; failures visible rather than hidden |

**Do not start by redesigning the app.** Maintain the separate local Streamlit demo as an evidence collection / expert-review interface; the public demo remains research-only.

## Candidate field image resources: investigate before use

These are **leads, not newly tested datasets or guaranteed licenses**:

1. **Wheat Fungi Diseases Dataset (WFD)** — https://wfd.sysbio.ru/index.html — 2,414 expert-marked images; Healthy, leaf rust, yellow rust, septoria, mildew and stem rust; **multi-label**: samples may carry several disease tags. Download involves terms and a form. Review licensing, overlapping sources and label mapping; don't silently map multi-label images into exclusive Brown/Healthy/Yellow labels.
2. **ETH high-resolution field leaf sequences (2026)** — https://www.research-collection.ethz.ch/items/225e7a74-5ff6-4527-82f4-58b46322cdd3 — source paper reports 12,520 RGB wheat-leaf images in 1,032 sequences, plus temporal/geometric metadata, symptom masks and source grouping. Evaluate whether conditions/labels actually align with our narrow diagnosis task and whether download and downstream use are permitted. **Split by entire leaf sequence, not by image**, and preferably held-out field/season.
3. **NWRD field stripe-rust segmentation resource** — https://zenodo.org/records/21672928 — high-resolution, complex-background multi-leaf images annotated for stripe rust; likely more appropriate for *localization/robustness* stress tests than direct three-class classification. Check label mapping and terms.
4. **New expert-collected wheat-leaf data** — prioritise a legitimate research relationship with a wheat-pathology or extension partner. Capture site, field, plant, session, device, lighting, wheat variety/stage and disease expert reference; obtain permissions and retain originals unchanged. Include known rusts, healthy, uncertain, lookalikes and mixed symptoms.

**No one of these datasets becomes an untouched test merely because it is downloaded.** Define intended use, label compatibility, split and freeze the evaluation protocol before model selection or visual inspection of its evaluation examples. Preserve some entirely new field/site data for prospective testing whenever feasible.

## Literature positioning — prevent accidental duplication

- Xiang, Shi and Zhu (May 2026), *Quantifying the reliability gap in cross-domain plant disease classification*, Frontiers in Plant Science, https://doi.org/10.3389/fpls.2026.1826962. **Already compares** image statistics, Grad-CAM, calibration, selective prediction, adaptation and multiple backbones under domain shift. Our general observation is not new.
- *StripeRust-Pocket* (2024), https://doi.org/10.34133/plantphenomics.0201, developed a mobile wheat stripe-rust severity tool with field image assessments. **A mobile interface alone is not novel.**
- 2026 *High-Resolution Leaf Image Sequences with Geometric Alignment for Dynamic Phenotyping of Foliar Diseases*, https://doi.org/10.1038/s41597-026-06567-y — relevant for sequence-aware field independence and spatial/temporal analysis, but not automatically a disease-classification holdout.
- WFD original publication: Genaev et al. (2021), https://doi.org/10.3390/plants10081500.

Before drafting a submission, expand this related-work review to include *wheat-specific source-generalisation and shortcut-learning intervention studies*, not just broad plant-disease accuracy papers. **Do not claim established novelty yet.**

## First actionable tasks (in order)

1. Write the **one-page experiment specification**: hypotheses, intervention arms, primary outcome (false-Healthy risk + Yellow Rust recall on independent data), secondary outcomes, grouping and statistical plan. Get external scientific criticism before running the definitive experiment.
2. Build a **dataset provenance and licensing inventory** for original/JIC/Ethiopia/IARI and each proposed external collection. Avoid opening a prospective final test just to see if performance improves.
3. Contact a **qualified wheat-pathology collaborator**, asking for feasibility and reference-label guidance rather than an endorsement. Do not imply anyone has already agreed to collaborate.
4. Implement **a deterministic grouped-split manifest and data audit**, with tests preventing cross-split plant/sequence leakage. No model retraining on JIC to choose settings.
5. Conduct the preregistered pilot on development sets; only after model and review policy are frozen, evaluate an appropriate independent field test.
6. Produce a **figure-led scientific report**: domain-wise confusion matrices, causal ablations, group-level confidence intervals, calibration/risk–coverage, examples of corrected and unresolved failures, and a transparent limitations page.

## Presentation / recognition criterion

An impressive research narrative consists of a difficult, well-supported finding, a clear causal design, independent expert validation, a demonstrated difference from prior literature, and an honest account of unresolved cases. GitHub stars, polished screens, repeated conference talks, or a high score on an already examined benchmark cannot replace these. Citation, independent use, adoption, outside critique and peer-reviewed evaluation must be earned rather than assumed.

**Current truthful description:** "Wheat Rust AI has documented severe cross-source class collapse and high-confidence false-Healthy errors; acquisition-related shortcut learning is under investigation, and the field-readiness research protocol is being developed." 
