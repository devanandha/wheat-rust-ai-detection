# Phase 2 — External generalisation and failure analysis

**Project:** Wheat Rust AI Detection
**Status:** Exploratory, retrospective diagnostic analyses; not a prospective independent validation study.
**Reproducibility check:** Five experiment outputs were rerun from a local clone on 19 November 2026 and compared with archived CSV files. Exact agreement was observed for the compared rows, labels, and numerical values in that environment. Independent reproduction on a different machine has **not** been established.

## Purpose

The classifier achieved high accuracy on its internal validation split but performed poorly on external images. Phase 2 investigates sensitivity to image framing, brightness, and foreground/background manipulations, and documents failure on a further external dataset. These interventions are **diagnostic** and do not establish a causal explanation for domain shift.

The frozen classifier is a three-class MobileNetV2-based model with class order `Brown_rust`, `Healthy`, `Yellow_rust`. The model weights are not updated in any experiment described here. The saved model handles its MobileNetV2 preprocessing internally: supply RGB images at 224 × 224 in the 0–255 scale, without applying a second external MobileNetV2 normalization.

## Main results

| Evaluation or intervention | Scope | Observed result |
|---|---|---|
| Original internal validation | 737 images | 728/737 correct (98.78%) |
| JIC external baseline | 489 images | 188/489 correct (38.45%); Yellow rust 15/239 correct (6.28%) |
| JIC 80% central crop | 489 images | 42.33% accuracy; Yellow rust 12.55% recall |
| JIC 60% central crop | 489 images | 48.67% accuracy; Yellow rust 22.18% recall |
| JIC 0.8× brightness | 489 images | 37.63% accuracy |
| JIC 1.2× brightness | 489 images | 38.65% accuracy |
| JIC manually masked background pilot | Nine selected Yellow rust images | Three class predictions changed; zero wrong-to-right and two right-to-wrong |
| NUST frozen-model evaluation | 100 images | Zero Yellow rust predictions; 80 Healthy and 20 Brown rust predictions |
| NUST lesion-centred cropping | Same 100 images | Zero Yellow rust predictions in full, lesion-centred, and control conditions |

**NUST label caveat:** Interpreting all 100 NUST images as positive Yellow/stripe rust *depends on the assumption* that their supplied lesion masks annotate that condition. This assumption must be checked against the dataset's source documentation before making a definitive disease-recall claim. The model was not trained on these NUST images. The source dataset's `train`/`test` split names describe its own partition, not this classifier's training and test partitions. The 0/100 result is a conditional, one-class observation, **not** a three-class accuracy estimate.

## Scientific limitations

- JIC was inspected repeatedly during development and should not be described as an untouched or independent final test set for the intervention choices.
- Central cropping changes framing, apparent lesion size, and scene composition simultaneously; its accuracy increase does not identify a single causal mechanism.
- The nine-image background experiment used outcome-selected images, manually drawn masks, and an artificial constant-grey replacement. It is an exploratory pilot, not evidence of background causality or a representative performance estimate.
- NUST lesion-centred crop selection used *ground-truth masks* to locate the crop and therefore cannot be presented as an inference-time intervention available without segmentation labels. The crop comparison is diagnostic.
- One NUST image (`101.jpg`) had a mask/image size discrepancy and required nearest-neighbour mask resizing. An overlay was visually reviewed, but pixel-perfect registration was not established.
- Synthetic resolution and Grad-CAM perturbation analyses are separate diagnostics and do not establish robust interpretability or clinical/agricultural readiness.
- These results do not establish reliability under new agricultural field conditions or readiness for deployment.

## Repository layout

```text
research/phase2_generalisation/
├── .gitignore
├── README.md
├── scripts/
│   ├── phase2_config.py
│   ├── phase2_nust_frozen_evaluation.py
│   ├── phase2_nust_lesion_crop_experiment_v2.py
│   ├── phase2_cropping_experiment.py
│   ├── phase2_illumination_experiment.py
│   ├── phase2_background_pilot.py
│   └── ... additional audit/legacy scripts
├── results/                   # archived numerical outputs
├── masks/                     # local only; ignored pending rights review
├── figures/                   # local only; ignored pending rights review
└── reproduced_outputs/        # generated locally; ignored by Git
```

Historical and diagnostic scripts are retained for transparency. The corrected NUST lesion-crop implementation is `phase2_nust_lesion_crop_experiment_v2.py`; the earlier similarly named version is retained as a historical attempt and **should not** be used to reproduce the reported lesion-crop results.

## Reproducing the five tested experiments

Run these commands **from the repository root** in Windows Command Prompt. Provide datasets separately from their original authorised sources; they are **not** bundled with this repository.

### Requirements

Use the project's compatible Python environment with TensorFlow, NumPy, Pandas, Pillow, and the dependencies used by `src.data`. The local verification used Python 3.11.2 and TensorFlow 2.18.1; package versions and platform-dependent numerical differences may affect replication. Verify the model checksum separately if exact artifact identity is important.

### Configure source locations

```cmd
set "PHASE2_JIC_DIR=D:\datasets\JIC\val"
set "PHASE2_NUST_DIR=D:\datasets\NUST\NWRD"
```

Replace these illustrative local paths with the paths where **you** have legally obtained the datasets. Optional variables are `PHASE2_MODEL`, `PHASE2_JIC_REFERENCE`, `PHASE2_NUST_REFERENCE`, and `PHASE2_OUTPUT_DIR`. The default model path is `models/wheat_rust_mobilenetv2.keras`, and the default output path is `research/phase2_generalisation/reproduced_outputs/`.

### Run evaluations

```cmd
python research\phase2_generalisation\scripts\phase2_nust_frozen_evaluation.py
python research\phase2_generalisation\scripts\phase2_nust_lesion_crop_experiment_v2.py
python research\phase2_generalisation\scripts\phase2_cropping_experiment.py
python research\phase2_generalisation\scripts\phase2_illumination_experiment.py
python research\phase2_generalisation\scripts\phase2_background_pilot.py evaluate
```

The background evaluation requires the nine manually annotated PNG masks in `research/phase2_generalisation/reproduced_outputs/phase2_background_pilot/masks/`, with filenames matching those expected by the script. These masks are not currently published pending a redistribution-rights review; the archived numerical outputs and code can still be examined, but the background experiment is **not self-contained for external reruns**.

Do not invoke `phase2_background_pilot.py annotate` when only reproducing the published evaluation; annotation is a separate interactive workflow.

## Local numerical verification — 19 November 2026

| Test | Comparison of archived and rerun CSV |
|---|---|
| NUST frozen-model baseline | 100 rows; image IDs, labels, and all three class probabilities matched; maximum difference 0.0 |
| NUST lesion-crop evaluation | 100 rows; image IDs, labels, and nine class probabilities matched; maximum difference 0.0 |
| JIC central crops | 978 rows (489 × 2); entire CSV matched; maximum numerical difference 0.0 |
| JIC illumination | 978 rows (489 × 2); entire CSV matched; maximum numerical difference 0.0 |
| JIC background pilot | Nine rows; entire CSV matched; maximum numerical difference 0.0 |

These checks demonstrate **same-environment rerun consistency**, not cross-machine reproducibility. A direct rerun of the JIC cropping script was also completed after the import-path repair with `PYTHONPATH` cleared, reproducing its reported summary metrics.

## Data provenance, redistribution, and archived references

The source images and the original datasets are not included. Before publishing or distributing any masks, image-based figures, or derived materials, check the JIC and NUST datasets' licence terms and any attribution requirements. The local `.gitignore` intentionally excludes `masks/`, `figures/`, and `reproduced_outputs/`.

The published NUST baseline reference is `results/phase2_nust_evaluation_nust_frozen_predictions.csv`. It contains portable relative paths and was verified against the original 100 image records, image hashes, predicted labels, and probability values. The historical `nust_frozen_predictions_original.csv` remains local and excluded from Git because it contains absolute Windows paths. Its SHA-256 differs from the portable export, although the verified prediction values match. The Experiment 7 script now uses the portable reference by default. Dataset images and masks must be obtained separately from their original sources.

Review `results/` and individual legacy scripts for local filesystem paths, source-image-derived data, licensing issues, and reproducibility limitations before staging or pushing to GitHub.

## Interpretation

A near-99% internal-validation score did not transfer reliably to the JIC and NUST external sources. Some central crops improved JIC predictions, but simple brightness changes did not materially improve accuracy, and mask-guided NUST crops did not recover Yellow rust predictions. These observations support further investigation of dataset composition, framing, annotation alignment, image acquisition, and model generalisation. They do not uniquely identify why transfer failed.

## Publication status

As of 19 November 2026, the five selected experiments have passed local prediction-level comparisons. Public release remains subject to a final code/data-path audit, source-dataset documentation and licence verification, documentation review, and an explicit Git commit/push. No independent third-party reproduction is claimed here.
