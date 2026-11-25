# Phase 3 — Cohort Preparation Rules
Date: 2026-11-20
Status: Design decisions recorded before Phase 3 cohort generation

## Experimental design
- Per-class sizes: 150, 300
- Additional-source fractions: 0, 0.25, 0.50
- Random seeds: 42, 43, 44
- Planned training runs: 18
- Architecture: Frozen ImageNet MobileNetV2

## Candidate selection
- Primary source: original training split only.
- Additional sources: 764 records in the historical staging_manifest.json.
- Verify original paths and SHA-256 hashes before sampling.
- Preserve historical exclusions.
- Protect original validation, JIC benchmark, Ethiopia holdout, and IARI validation/test.
- Exclude exact SHA-256 overlaps and pHash distances <= 5.
- Screen selected training images for cross-source duplicates.
- Record all exclusions and remaining eligible counts.
- Abort if any condition lacks sufficient eligible images.

## Sampling
- 150 images/class, 25% mix: 38 additional + 112 primary.
- 300 images/class, 25% mix: 75 additional + 225 primary.
- 150 images/class, 50% mix: 75 additional + 75 primary.
- 300 images/class, 50% mix: 150 additional + 150 primary.
- Zero-mix conditions use primary-source images only.
- Sample reproducibly using the specified seeds.

## Evaluation and reporting
- Original validation: early stopping and checkpoint selection.
- JIC: previously examined external benchmark; never select models using JIC results.
- Report all 18 conditions, including failures and negative findings.
- Disclose suspected original train-validation near-duplicates.
- Disclose source/class confounding and lack of field-level independence.
- Do not claim the October protocol was formally preregistered.
- Record code hashes, environment, image manifests, and experiment outputs.
- Preserve the original model and historical results.
