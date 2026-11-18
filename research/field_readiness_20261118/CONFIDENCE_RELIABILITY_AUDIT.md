# Exploratory confidence and false-reassurance audit — protocol

**Date:** 18 November 2026  
**Status:** Completed by the user on the previously evaluated 737 original-validation and 489 JIC images. [View the exploratory reliability results](CONFIDENCE_RELIABILITY_RESULTS.md): JIC had 151 diseased → Healthy predictions, and 55 of 83 accepted score-≥0.99 predictions were wrong. These are not independent field-validation results.  
**Scope:** Existing frozen MobileNetV2 model, original `val` and previously examined JIC `val`. This is a **reused benchmark exploratory analysis, not a new field study or a novel external test**.

## Motivation

The original MobileNetV2 model got **728/737** original validation images correct (98.78%), but only **188/489** from JIC (38.45%). JIC Yellow Rust recall was **15/239**. The field prototype can run predictions locally; before expanding its features, study whether the model's **uncalibrated maximum softmax score** distinguishes correct and incorrect outputs under this acquisition shift.

**Research question:** Does referring low-score predictions to experts meaningfully reduce errors and **diseased-as-Healthy false reassurance** without rejecting nearly all difficult disease cases? Does the original source relationship transfer to the already inspected JIC data?

**Do not assume** that a softmax score of 0.99 corresponds to a 99% probability that the diagnosis is correct. The existing model is **closed-set** and cannot tell all unknown plant diseases from its three trained labels.

## Fixed analysis plan (before rerunning)

The [`tools/audit_prediction_reliability.py`](../../tools/audit_prediction_reliability.py) program uses the same original `src.data.load_split` (224 × 224 RGB on 0–255; preprocessing is built into model) and the unchanged saved `.keras` weights. Predictions are made once per image without training or modifying source files.

- Assess **original validation** (737) and **previously inspected JIC** (489) separately; report all classes and the source difference explicitly.
- Calculate confusion matrix, overall/macro metrics, class precision/recall/F1 and `diseased_predicted_healthy` (**Brown/Yellow truth predicted Healthy**).
- Calculate 10-bin expected calibration error (ECE) for maximum class score, multiclass Brier score, negative log-likelihood, mean max score. These are **descriptive**, and calibration metrics can depend on data composition.
- Prespecify acceptance thresholds **0.0, 0.5, 0.7, 0.9, 0.95, 0.99**. Accepted means maximum model score >= threshold; remaining predictions are marked for expert review, *not assumed safe*. For each: `coverage`, count referred for review, accepted error rate, count of accepted diseased images predicted Healthy, and Yellow Rust recall **among accepted Yellow Rust images**. Cases with no accepted images get `null` rates, **not 0%**.
- Count wrong predictions with score >=0.9 and >=0.99, and retain per-image labels/scores as local CSV for case review.
- Verify unperturbed baseline reproduction of **728** and **188**. If either baseline mismatch occurs, **stop scientific interpretation** and investigate saved model, label mapping, preprocessing and dataset version.

**Important:** Thresholds are fixed exploratory descriptive values, **not optimised or deployed review thresholds**. Even if an apparently successful threshold is found here, it cannot be described as field-validated or calibrated. The JIC data has been examined repeatedly in preceding experiments.

## Windows setup

From the existing project folder `C:\WheatAI\wheat-rust-ai-detection-main` (not the TensorFlow Streamlit server terminal), download the two new files from `main`:

```cmd
curl.exe -L "https://raw.githubusercontent.com/devanandha/wheat-rust-ai-detection/main/tools/audit_prediction_reliability.py" -o "tools\audit_prediction_reliability.py"
curl.exe -L "https://raw.githubusercontent.com/devanandha/wheat-rust-ai-detection/main/tests/test_prediction_reliability.py" -o "tests\test_prediction_reliability.py"
```

Run the lightweight synthetic unit tests (do not require TensorFlow):

```cmd
python -m unittest discover -s tests -p "test_prediction_reliability.py" -v
```

Then, in the normal command prompt, run the frozen-model audit on the **two previously inspected collections**:

```cmd
python -m tools.audit_prediction_reliability --model "F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\models\wheat_rust_mobilenetv2.keras" --original-val "F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\Data\val" --jic-val "F:\Github\external-wheat-validation\val" --output-dir "outputs\reliability_audit"
```

If TensorFlow runs out of memory, stop the local Streamlit app with Ctrl+C, close other heavy apps and retry in Command Prompt; do not delete the existing model or datasets.

Outputs:
- `reliability_report.json` — full source-wise summary, prespecified fixed-threshold table, calibration bins, model SHA, baseline check and caveats.
- `original_val_predictions.csv`, `jic_predictions.csv` — every image's true class, class prediction, correctness, all three uncalibrated scores and highest score.
- `original_val_review_tradeoff.csv`, `jic_review_tradeoff.csv` — coverage versus residual risk, Yellow Rust cases and false-Healthy reassurances.
- `original_val_confidence_bins.csv`, `jic_confidence_bins.csv` — descriptive 10-bin confidence versus observed accuracy.

For file location after success:

```cmd
explorer outputs\reliability_audit
```

These outputs need not be committed to GitHub; source-image identifiers and field metadata should be handled carefully.

## Interpretation and next gating study

Do not claim a reliability gate is effective solely because high-score predictions are often correct on original validation. The operational question is how it behaves under **independent device/site/season shift**, including unseen or mixed diseases.

After this retrospective audit:
1. Decide which failure mechanisms remain after considering high-confidence wrong cases, especially diseased → Healthy. Do not alter thresholds based on JIC results and then re-report JIC as a fresh test.
2. Define an abstention/review policy and calibration training split **separate from a locked independent field test**. If the calibration data are unlike the field test, confidence scores may still fail.
3. Obtain an external qualified wheat-pathology reviewer and prospective field images with recorded capture conditions, locations grouped at least by field/session/plant, and independent reference labels. Include healthy, brown, yellow, lookalikes, mixed and unknown cases.
4. Pre-register safety and failure gates with the domain reviewer: per-class recall, disease-as-Healthy rates, risk–coverage, percentage referred for expert review, calibration by site/device and latency.
5. Only after prospective testing consider allowing the app to reveal conditional automated suggestions. The present prototype must remain research/demo only.

**Success for this stage:** reproducible, transparent **failure characterization** and a defensible prospective experiment—not a higher headline accuracy or an unsupported “field-ready” claim.
