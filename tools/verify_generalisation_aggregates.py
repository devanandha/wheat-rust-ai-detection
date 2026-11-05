"""Check published 18-run aggregate statistics using only the Python standard library.

This validates arithmetic and metadata consistency of *archived CSV tables*.
It does NOT reproduce model training, rerun inference, or establish dataset independence.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / "research/dataset_generalisation/per_run_results_18.csv"
DEFAULT_SUMMARY = ROOT / "research/dataset_generalisation/condition_summary.csv"
EXPECTED_GROUPS = {(n, share) for n in (450, 900) for share in (0, 25, 50)}
EXPECTED_SEEDS = {42, 43, 44}
DATASET_COUNTS = {"jic": 489, "iari": 129, "ethiopia": 61}
METRIC_COLUMNS = {
    "jic": ("jic_accuracy", "jic_macro_f1"),
    "iari": ("iari_accuracy", "iari_macro_f1_present_classes"),
    "ethiopia": ("ethiopia_accuracy", "ethiopia_macro_f1_present_classes"),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"No rows found in {path}")
    return result


def verify(runs_path, summary_path):
    runs, summaries = read_csv(runs_path), read_csv(summary_path)
    if len(runs) != 18:
        raise ValueError(f"Expected 18 archived runs; found {len(runs)}")
    if len(summaries) != 6:
        raise ValueError(f"Expected 6 aggregate conditions; found {len(summaries)}")

    by_group = {}
    for row in runs:
        group = (int(row["train_images"]), int(row["additional_source_share_nominal_pct"]))
        seed = int(row["seed"])
        if group not in EXPECTED_GROUPS or seed not in EXPECTED_SEEDS:
            raise ValueError(f"Unexpected training condition or seed: {group}, {seed}")
        expected_condition = f"n{group[0] // 3}_mix{group[1]:03d}_seed{seed}"
        if row["condition"] != expected_condition:
            raise ValueError(f"Unexpected condition label: {row['condition']} vs {expected_condition}")
        for name, count in DATASET_COUNTS.items():
            if int(row[f"{name}_images"]) != count:
                raise ValueError(f"Unexpected {name} sample count in {row['condition']}")
        for columns in METRIC_COLUMNS.values():
            for column in columns:
                value = float(row[column])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {column} in {row['condition']}: {value}")
        by_group.setdefault(group, []).append(row)

    if set(by_group) != EXPECTED_GROUPS:
        raise ValueError("Missing one or more training conditions")
    for group, members in by_group.items():
        if len(members) != 3 or {int(r["seed"]) for r in members} != EXPECTED_SEEDS:
            raise ValueError(f"Incorrect seeds/duplicates in condition {group}")

    seen = set()
    for summary in summaries:
        group = (int(summary["train_images"]), int(summary["additional_source_share_nominal_pct"]))
        if group not in by_group or group in seen:
            raise ValueError(f"Unknown or duplicate summary group: {group}")
        seen.add(group)
        if int(summary["seed_count"]) != 3:
            raise ValueError(f"Incorrect seed_count for {group}")
        for name, columns in METRIC_COLUMNS.items():
            for field_name, run_column in zip(("accuracy", "macro_f1"), columns):
                samples = [float(row[run_column]) for row in by_group[group]]
                for suffix, computed in (("mean", mean(samples)), ("sd", stdev(samples))):
                    label = f"{name}_{field_name}_{suffix}"
                    recorded = float(summary[label])
                    if not math.isclose(computed, recorded, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(
                            f"Mismatch in {label} for {group}: "
                            f"archived={recorded:.12g}, recalculated={computed:.12g}"
                        )
    if seen != EXPECTED_GROUPS:
        raise ValueError("Aggregate table is missing conditions")

    print("PASS: 18 archived runs across 6 groups and 3 seeds.")
    print("PASS: JIC, IARI and Ethiopia reported means and sample SDs match source CSV.")
    print("NOTE: This verifies archived summary arithmetic only. It does NOT re-run")
    print("      models, verify image provenance, or independently reproduce experiments.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    try:
        verify(args.runs, args.summary)
    except (OSError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
