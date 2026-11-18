# Acquisition-shift diagnosis: protocol and evidence limits

**Status, 18 November 2026:** New exploratory diagnostic protocol and synthetic-tested image audit tool. A descriptive audit of **4,168 images** (original training and validation plus previously inspected JIC) and an exploratory image-dimensions-only decision-tree diagnostic are documented in [EXPLORATORY_FINDINGS.md](EXPLORATORY_FINDINGS.md). No new blind dataset was tested; no causal explanation or neural-network model improvement has been demonstrated.

## Narrow research question

Do image-acquisition characteristics such as dimensions, sRGB brightness, saturation and image-edge statistics differ by source **within the same disease class**? Could these differences motivate a controlled explanation for the classifier's cross-source prediction collapse?

The audit describes data distributions; it cannot identify the model's learned features or prove the cause of prediction errors.

## What is already established

- Original-dataset internal validation: MobileNetV2 correctly classified 728/737 images (98.78%). The backbone is ImageNet-pretrained, and the classification head is trained on the original wheat-rust dataset.
- JIC external-source evaluation without training on JIC images: 188/489 correct (38.45%). Repeated inspection means the JIC benchmark is **not** an untouched blind test for future research.
- Original train/validation leakage checks reported zero SHA-256 exact duplicates and one pHash near-duplicate pair at the stated threshold. Removing that validation image barely changed accuracy. **Neither hash check establishes plant, field or session independence.**
- The IARI and Ethiopia source-exclusion studies have missing disease classes in individual sources and holdouts, so source and class are partly confounded. Strong within-source, image-held-out accuracy is not automatically unseen-domain generalisation.

These are historical results, not reruns performed by the new tool.

## Step 1 — Run the descriptive audit on licensed local images

The new script is at tools/audit_acquisition_shift.py. It expects each source folder to contain class subfolders named Brown_rust, Healthy and/or Yellow_rust. For example, in a terminal from the repository root:

    python tools/audit_acquisition_shift.py --source original=/path/to/original/data/val --source jic=/path/to/jic/val --output-dir outputs/acquisition_shift_audit

For Windows Command Prompt, put the command on one line and use the local Windows paths. Do not upload raw datasets to GitHub. If a source has different folder labels, verify the original taxonomy before creating a labelled local working copy.

Outputs:

- per_image_metrics.csv — image ID, source, label, SHA-256, dimensions/aspect ratio, normalized brightness and saturation proxies, and outer-image-edge metrics.
- source_class_summary.csv — sample counts and descriptive means/SD, grouped separately by source and class; missing classes are explicit.
- audit_report.json — classes shared between sources, unreadable inputs, exact-file hash overlaps and warnings.

The **outer 12.5% image-edge** statistic is not a verified background segmentation mask. Normalized brightness is an sRGB-coded pixel proxy, not physically calibrated illumination. Exact-file hash overlap is not a complete leakage audit.

Verify the script without research data:

    python -m unittest discover -s tests -p test_acquisition_shift.py -v

These tests use synthetic images; passing them does not validate any wheat-dataset finding.

## Step 2 — Evaluate one falsifiable mechanism, not dozens of configurations

1. First assess provenance, label definitions, image overlap and available plant/field/session identifiers. Name the absent metadata explicitly.
2. Compare only like-for-like disease classes across sources; report source/class counts and full confusion matrices so absent-class predictions are visible.
3. Prespecify one hypothesis to test. For example: contextual image-edge/background cues help explain Brown-rust prediction collapse after an acquisition change. This is a hypothesis, not a finding.
4. Design a matched controlled intervention after inspecting original source data, ideally keeping model architecture, seed schedule and training images comparable. Any cropping, colour adjustment or background standardization must be checked for preservation of disease symptoms.
5. Use JIC/IARI/Ethiopia results already inspected for exploratory development only. **Do not select an intervention based on an evaluation set later advertised as untouched.** Seek a genuinely independent, compatible three-class source for final evaluation; if unavailable, report exploratory findings only.
6. Report per-class recall, three-class confusion matrices, class support, macro-F1 with correct class definitions, seed variation and all negative results. Where valid, report uncertainty intervals.
7. Review prior literature to identify what new controlled evidence, if any, is added. A large accuracy drop between datasets alone is not a novel discovery.

## Prior work and novelty boundary

This proposed diagnostic is **not a novel technique by itself**. Existing plant-disease research documents image-source shift, background/illumination differences, performance collapse and image-statistic/Grad-CAM comparisons. Relevant starting points include:

- [Plant disease recognition datasets in the age of deep learning: challenges and opportunities (2024)](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1452551/full) — dataset provenance, repeated observations, background and cross-source evaluation.
- [Quantifying the reliability gap in cross-domain plant disease classification (2026)](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1826962/full) — existing image-statistic and Grad-CAM analysis, so those tools alone should not be represented as a novel contribution.

A possible specific research contribution would require a **new, well-controlled explanation** of the unusually strong class-prediction collapse observed in this wheat-rust evaluation, with compatible labels, documented provenance and independently evaluated interventions. This remains a research objective, not a demonstrated contribution.


## Frozen-model paired sensitivity probe (added 18 November 2026)

The exploratory acquisition audit found strong class-associated differences in **source image dimensions** in both original train and validation. A metadata-only classifier transferred poorly. This is not evidence that the actual CNN learned those metadata cues.

To test the saved model's **input-resolution sensitivity** without retraining, run the [paired resolution probe](../../tools/probe_resolution_sensitivity.py). It uses the existing `src.data.load_split` loader, which supplies 224 x 224 RGB values on the original 0–255 scale; **MobileNetV2 preprocessing is inside the saved model**. The probe compares inference-mode predictions with and without a predetermined bilinear downsample to 112 x 112 followed by bilinear resizing back to 224 x 224. Model weights and source files remain unchanged.

From repository root on Windows (replace paths for other machines):

```cmd
python -m tools.probe_resolution_sensitivity --model "F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\models\wheat_rust_mobilenetv2.keras" --original-val "F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data\val" --jic-val "F:\Github\external-wheat-validation\val" --output-dir "outputs\resolution_sensitivity"
```

Outputs include `original_val_paired_predictions.csv`, `jic_paired_predictions.csv` and `paired_sensitivity_summary.json`. The latter includes a SHA-256 identifier of the exact saved model and flags whether unperturbed correct counts reproduce the historical original 728/737 and JIC 188/489. **If either baseline does not match, diagnose preprocessing, model-file and dataset-version differences before interpreting perturbation effects.**

This artificial blur/downsample is not a physical recreation of original collection conditions, and input-scale sensitivity is not a direct causal demonstration of resolution-shortcut learning. Pre-specify follow-up interventions and seek independent evaluation data rather than selecting hyperparameters based on JIC.

For a lightweight unit test of metric bookkeeping (no TensorFlow required):

```cmd
python -m unittest discover -s tests -p test_resolution_sensitivity.py -v
```

## Interpretation gate

Do not claim that brightness, preprocessing or background differences **cause** poor generalisation without a controlled demonstration. If descriptive metrics show little separation, or if a properly controlled intervention does not improve performance on independent evaluation data, preserve that negative result.

## Repository evidence to read alongside this protocol

- [External-source evaluation](../../docs/EXTERNAL_VALIDATION.md)
- [Original-split leakage and failure analysis](../../docs/FAILURE_ANALYSIS.md)
- [Dataset provenance and source/class confounding](../dataset_generalisation/dataset_provenance.md)
- [Source-exclusion evidence and limits](../source_exclusion_20261023/README.md)
