# Phase 3 — Dataset Integrity Audit

Audit date: 2026-11-24

## Scope

This audit examined the original eligibility manifest, all 18
cohort manifests, recorded source composition, selected-image
eligibility, and the staged training image contents.

## Results

| Check | Result |
|---|---|
| Eligibility manifest SHA-256 | PASS |
| Cohort manifest SHA-256 | 18/18 PASS |
| Cohort membership and composition | 18/18 PASS |
| Duplicate SHA-256 within cohorts | None detected |
| Selected records marked eligible | All passed |
| Recorded protected-set matches selected | None |
| Staged training image SHA-256 | 12,150/12,150 PASS |
| Missing or unexpected staged image files | None detected |

The 18 training cohorts comprise nine 150-images-per-class runs
and nine 300-images-per-class runs.

The 12,150 checks represent image occurrences across cohorts,
not necessarily 12,150 unique images.

## Eligibility screening

Candidate records: 3,706
Eligible records: 3,705
Excluded records: 1

The excluded candidate was a primary-source Healthy image
with a perceptual-hash distance of 2 from an original
validation image.

The screening configuration used exact SHA-256 comparison
and perceptual-hash distance <= 5.

## Source composition

Additional-source images were recorded under the source labels
`iari` and `ethiopia`, not a generic `additional` label.

Source identity was associated with disease class. For example,
the n150_mix25_seed42 cohort used IARI Brown Rust images,
Ethiopia Yellow Rust images, and both sources for Healthy images.

This represents potential source-class confounding.

## Interpretation

The preserved manifests match the hashes recorded in the
original study configuration.

All recorded cohort membership and composition checks passed.

The staged training image contents match their recorded
SHA-256 hashes.

These checks support dataset traceability and integrity.
They do not establish complete biological or field-level
independence.

## Remaining limitations

- A previously identified Brown Rust training/validation pair
  had perceptual-hash distance 8, above the exclusion threshold.
- Original validation and JIC had been evaluated previously.
- The JIC benchmark is not an untouched independent test.
- Source-class confounding remains a possible explanation
  for changes in class-specific recall.
- Plant-level and field-level independence are not established.
- Independent external validation remains necessary.

## Preservation

No original cohort manifests, datasets, experiment metadata,
model checkpoints, or historical results were modified.

The original eligibility and cohort manifests remain local
and are not included in this public report.
