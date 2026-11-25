# Phase 3 — Per-Class Generalisation Analysis

## Research question

How does increasing training-source diversity affect class-specific generalisation in wheat-rust classification?

## Experimental setup

- Model: MobileNetV2 with a frozen pretrained backbone
- Training sizes: 150 and 300 images per class
- Source mixing: 0%, 25%, 50%
- Seeds: 42, 43, 44
- External benchmark: JIC, 489 images
- Class order: Brown_rust, Healthy, Yellow_rust

The external benchmark contains 128 Brown Rust, 122 Healthy and 239 Yellow Rust images.

All 18 confusion matrices passed the class-support audit.

## Mean external recall

| Images/class | Source mix | Brown Rust | Healthy | Yellow Rust |
|---|---|---|---|---|
| 150 | 0% | 62.24% | 80.05% | 4.18% |
| 150 | 25% | 29.17% | 93.99% | 40.03% |
| 150 | 50% | 21.35% | 89.34% | 47.56% |
| 300 | 0% | 54.69% | 81.97% | 3.63% |
| 300 | 25% | 26.04% | 88.25% | 46.44% |
| 300 | 50% | 14.84% | 93.17% | 53.42% |

Values are means across three random seeds.

## Principal finding

Increasing source mixing improved overall external benchmark accuracy, but the improvement was not uniform across disease classes.

At 300 images per class, increasing source mixing from 0% to 50%:

- Improved external accuracy from 36.54% to 53.24%.
- Increased Yellow Rust recall from 3.63% to 53.42%.
- Increased Healthy recall from 81.97% to 93.17%.
- Reduced Brown Rust recall from 54.69% to 14.84%.

These changes correspond to +16.70, +49.79, +11.20 and -39.85 percentage points, respectively.

## Error analysis

For the 300-images-per-class experiments, the average number of Brown Rust images predicted as Yellow Rust increased from 0.0 to 58.0 as source mixing increased from 0% to 50%.

Correctly classified Brown Rust images decreased from 70.0 to 19.0.

Conversely, correctly classified Yellow Rust images increased from approximately 8.7 to 127.7.

This indicates a substantial shift in class-specific prediction behaviour.

The confusion matrices do not establish the underlying cause of this shift.

## Interpretation

Additional-source training was associated with improved detection of Yellow Rust on the evaluated benchmark, accompanied by deterioration in Brown Rust detection.

Aggregate accuracy alone would conceal this trade-off.

Greater source diversity should therefore not be interpreted as uniformly improving disease classification.

## Limitations

- Only three random seeds were evaluated per condition.
- The JIC benchmark had been evaluated previously and is not an untouched independent confirmation dataset.
- Identical class-support counts do not prove identical image identities or absence of leakage.
- Class-specific performance may depend on source composition, visual characteristics and label consistency.
- The cause of Brown Rust versus Yellow Rust confusion has not been established.
- No claim of field deployment readiness is supported.

## Reproducibility

The analysis is generated from the archived external confusion matrices using:

`analysis_20261123/calculate_per_class_recall.py`

The resulting summary is:

`analysis_20261123/per_class_recall.csv`

No historical model outputs or experiment metadata were modified.

## Next investigation

Evaluate frozen models on a genuinely independent external dataset, verify image-level provenance and overlap, and investigate Brown Rust versus Yellow Rust error patterns.
