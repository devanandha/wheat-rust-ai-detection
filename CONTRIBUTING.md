# Contributing to Wheat Rust AI Detection

Thank you for considering a contribution. This project is an **experimental computer-vision research prototype** focused on source-shift evaluation, wheat-rust classification, failure analysis, and explainability. It is **not validated for field diagnosis or crop-treatment decisions**.

We welcome reproducible corrections, critical technical feedback, and small improvements, including findings that challenge our assumptions.

## Helpful contributions

- Reproduce a documented result and report differences, including versions, seeds, hardware and data selection.
- Fix evaluation scripts, data checks, tests, documentation, or installation instructions.
- Improve class-wise error reporting, dataset-provenance checks, calibration analysis or leakage diagnostics.
- Identify limitations in interpretation, especially missing disease classes, repeated examination of test sets or source/acquisition confounding.
- Propose new experiments with a clearly defined question and an evaluation split that has **not** already informed development.

## Opening an issue

Before submitting, check existing issues and the [research documentation](docs/). Please include:

1. **Question or problem:** what did you expect, and what happened?
2. **Reproduction details:** operating system, Python and library versions, relevant command, model configuration, random seed and dataset source.
3. **Evidence:** a concise traceback, aggregate numbers or a small synthetic example.
4. **Limitations:** whether the issue depends on data or checkpoint files that cannot be publicly distributed.

Please **do not** upload third-party dataset images, identifiable personal information, confidential materials, credentials, or model weights whose sharing terms you cannot verify. Prefer links to official dataset records and the authors' usage conditions.

## Opening a pull request

- Keep changes focused and describe the scientific or engineering reason for each change.
- Include a simple test or reproducible command where feasible.
- Explain any effect on reported metrics, generated figures or research conclusions.
- **Preserve original experimental results.** Never overwrite or silently reinterpret previous outcomes to make performance look better.
- Discuss major architectural changes or new study designs in an issue first.
- Clearly distinguish exploratory tests from independent, untouched final evaluations.

## Research integrity

- Strong internal validation performance does not establish real-world field performance.
- No duplicate match under a chosen SHA-256/pHash protocol establishes only that the tested comparisons found no flagged pairs; it does not establish independent plants, fields or collection sessions.
- Existing JIC and source-specific evaluation sets have been inspected and used in analysis. New hypotheses should not treat them as untouched final tests.
- Grad-CAM perturbation results measure influence on the model under a specified intervention, **not** biologically verified lesion localisation.
- Dataset licences and permissions are distinct from this repository's software licence.

## Licence and citation

The source code is under the [MIT licence](LICENSE). Third-party images, data and weights may be subject to separate conditions.

If this code helps your research, please use the GitHub **Cite this repository** feature (backed by [CITATION.cff](CITATION.cff)) or the Zenodo DOI linked in the README. A citation is appreciated; it is not a substitute for compliance with licence terms.

Public issues and pull requests can be read by anyone. Please avoid private or sensitive material. An independent contributor's issue, review or pull request is not automatically evidence of product adoption, endorsement or scientific validation.

Thanks for helping make the project more transparent and reproducible.
