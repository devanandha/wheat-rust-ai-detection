# Field Research App — local prototype (v0.1)

**Branch:** [`field-research-prototype-20261118`](https://github.com/devanandha/wheat-rust-ai-detection/tree/field-research-prototype-20261118)  
**Entry point:** [`field_research_app.py`](../../field_research_app.py)  
**Status:** Implementation available for local smoke testing; **no test run on the user's machine has been observed**, no independent field samples have been collected, no user-access controls are implemented and no real-world performance has been validated.

## Two clearly separated workflows (updated 18 November 2026)

- **Demo Mode (default):** Upload an image you have the right to process, confirm permission to process it locally, and click **Run AI demo**. The isolated worker performs one frozen-model prediction. **No reviewer code, expert label or permission-reference field is requested, and no JSONL research record is created.** Uploaded bytes remain in app memory for processing, not stored as files. The model scores are uncalibrated, not a diagnosis.
- **Research Mode:** Only for studies with genuine permission and an independently assessed reference label. Reviewer code, permission/consent reference and two attestations are required. Explicit submission appends one metadata-only JSONL record. **Do not enter fake reviewer or permission identifiers for smoke testing.**

Choose the mode at the top of the local interface. Both modes include **unvalidated descriptive image checks** (brightness, dimensions, edge proxy). Neither mode provides field-validated crop decisions. The software does not substitute for a qualified plant pathologist.

## Purpose

This prototype is a **two-workflow local-only research interface**, not a wheat disease diagnosis app. The default **Demo Mode** runs the frozen MobileNetV2 model on a permitted photo without logging. **Research Mode** permits a researcher to upload/capture a photo, examine **unvalidated** image-quality signals, enter an independently assessed label plus site/device identifiers, then run a frozen-model comparison and append a local study record. In Research Mode, the reviewer records the label **before seeing the model score** to reduce anchoring.

To reduce startup memory pressure on Windows, Streamlit no longer imports TensorFlow at startup. A separate Python worker [\`src/field_inference_worker.py\`](../../src/field_inference_worker.py) imports TensorFlow for one model inference when a reviewer submits a permitted image, then exits. The worker reads photo bytes from stdin without writing image files. If TensorFlow still raises \`MemoryError\` in the worker, the computer needs more available RAM/Windows virtual memory; this software separation cannot fix a system-wide commit limit.

The prototype does not recommend fungicides or provide agronomic treatment instructions. It never declares the image “safe” based on a confidence threshold. The model's three supported labels are Healthy, Brown Rust and Yellow Rust; other diseases and mixed presentations are outside that scope.

## Security and ethics

**DO NOT publish or deploy this branch to Streamlit Cloud, Railway, a VPS, or an external network.** There is no authentication, protected data store, consent manager, expert verification, audit access log, encryption-at-rest, or retention policy. It is suitable only for **a trusted person's laptop at 127.0.0.1**. Even hashed images, source identifiers and expert records may be sensitive.

- Uploaded original image bytes remain in memory during the session; the prototype does not write photos to disk.
- Only after explicit permission acknowledgment and reviewer-entered label does the app append a **local JSONL study record** to `outputs/field_research/reviews.jsonl`.
- The log contains timestamp, SHA-256 image fingerprint, photo dimensions/quality proxies, field/session/device codes, permission reference, reviewer label, model hash and three uncalibrated scores. Do not enter farmer names, farm coordinates or other personally identifying data in notes/code fields.
- Git excludes `outputs/field_research/` and a `research/field_readiness_20261118/private/` directory on this branch. This does not encrypt those files or prevent manual accidental release.
- `reference_status = reviewer_entered` is *not* proof of independent expert confirmation; an agricultural partner must establish and audit reference-label procedures before performance claims.
- Do not use an image merely because it is available online; verify its license and the appropriate consent for local processing/metadata storage.
- Any experiment on the old JIC images is exploratory. The saved model is not field-ready (JIC 188/489; Yellow Rust 15/239).

## Existing Windows working copy: run without switching Git branches

The user currently has a working local ZIP extraction at `C:\WheatAI\wheat-rust-ai-detection-main` and a previously trained Keras checkpoint on F:. To try the prototype **without overwriting `app.py` or `src/model.py`**, open Windows Command Prompt in the existing directory and run, one command per line:

```cmd
curl.exe -L "https://raw.githubusercontent.com/devanandha/wheat-rust-ai-detection/field-research-prototype-20261118/field_research_app.py" -o "field_research_app.py"
curl.exe -L "https://raw.githubusercontent.com/devanandha/wheat-rust-ai-detection/field-research-prototype-20261118/src/field_research.py" -o "src\field_research.py"
curl.exe -L "https://raw.githubusercontent.com/devanandha/wheat-rust-ai-detection/field-research-prototype-20261118/src/field_inference_worker.py" -o "src\field_inference_worker.py"
curl.exe -L "https://raw.githubusercontent.com/devanandha/wheat-rust-ai-detection/field-research-prototype-20261118/tests/test_field_research.py" -o "tests\test_field_research.py"
python -m pip install "streamlit>=1.45,<2"
python -m unittest discover -s tests -p "test_field_research.py" -v
set "WHEAT_FIELD_MODEL=F:\Github\wheat-rust-ai-github-ready\wheat-rust-ai\models\wheat_rust_mobilenetv2.keras"
python -m streamlit run field_research_app.py --server.address 127.0.0.1 --server.port 8502 --server.fileWatcherType none
```

Then open **http://127.0.0.1:8502** on the *same computer*. For a software test, select **Demo Mode**, upload an image you are allowed to process, confirm local processing permission, and click **Run AI demo**; no reviewer code, expert label or research log is required. For a genuine, permissioned expert study, select **Research Mode**; obtain a real consent reference and an independent reviewer label before AI inference. Do not treat test data or existing benchmark images as new field evidence.

Exit with **Ctrl+C** in Command Prompt.

**TensorFlow dependencies:** the user's Windows Python 3.11.2 has already verified TensorFlow 2.18.1. The app also needs NumPy, Pillow and Streamlit. The test file is designed to run without TensorFlow.

## Basic workflow

1. Upload an image or use the camera (localhost).
2. Review original dimensions and descriptive brightness/contrast/edge proxies (not certified image-quality or wheat-content checks).
3. Enter a pseudonymous reviewer code, site/field/plant/session/device IDs and observation method; label *independently* as `Healthy`, `Brown_rust`, `Yellow_rust`, `Other_or_mixed`, or `Uncertain_unconfirmed`.
4. Provide an actual study permission reference and confirm the explicit consent/independent-label checkboxes.
5. Submit to run local model inference, show the per-class **uncalibrated scores**, reveal disagreement and save the metadata-only record.
6. If you choose to export the JSONL log, protect it appropriately. The source images are not included. Keep matching photos securely and separately under their valid permissions if scientific traceability is required.

## Planned next engineering/research improvements

- Real participant consent recording and access-controlled, encrypted storage **before multi-user or field deployment**.
- Plant/leaf image-quality annotations and calibrated tests, rather than arbitrary brightness/edge thresholds.
- Expert-adjudicated labels, support for unknown/mixed diseases, record history and reviewer-blinded model results.
- Append-safe storage and export of grouped, anonymized study metadata, validation of date/session identifiers, and duplicate-warning logic.
- Locked group-aware train/development/test manifests, site/season/device audits and an external independent test collected before model selection.
- Risk–coverage analysis and per-class disease-to-Healthy errors. Only after external validation and agronomist review should the tool expose automated field suggestions.
- Evaluate mobile/Android conversion offline on actual devices only after the model and data pipeline are validated.

For the full research design, see [FIELD_VALIDATION_PLAN.md](FIELD_VALIDATION_PLAN.md). This MVP is **software infrastructure**, not evidence that the research hypothesis is proved.
