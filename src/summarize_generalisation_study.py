"""Summarise every available run; do not tune models using held-out results."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, stdev


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("experiments/generalisation/study_manifest.json"))
    args = p.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.resolve().parent
    rows = []
    missing = []
    for cond in manifest["conditions"]:
        path = root / cond["id"] / "results" / "run_metadata.json"
        if not path.is_file():
            missing.append(cond["id"])
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["condition"] != cond["id"]:
            raise ValueError("Run condition does not match manifest")
        rows.append({"id": cond["id"], "per_class": cond["per_class"],
                     "mix": cond["mix"], "seed": cond["seed"],
                     "internal_accuracy": r["internal"]["accuracy"],
                     "internal_macro_f1": r["internal"]["macro_f1"],
                     "external_accuracy": r["external"]["accuracy"],
                     "external_macro_f1": r["external"]["macro_f1"],
                     "accuracy_gap": r["internal"]["accuracy"] - r["external"]["accuracy"]})
    if not rows:
        print("No trained runs found. Cohort preparation is not experimental evidence.")
        return
    with (root / "individual_run_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["per_class"], row["mix"])].append(row)
    metrics = ["internal_accuracy", "internal_macro_f1", "external_accuracy", "external_macro_f1", "accuracy_gap"]
    agg = []
    for (n, mix), rs in sorted(grouped.items()):
        d = {"per_class": n, "mix": mix, "runs": len(rs), "seeds": ",".join(str(r["seed"]) for r in rs)}
        for metric in metrics:
            values = [r[metric] for r in rs]
            d[metric + "_mean"] = mean(values)
            d[metric + "_sd"] = stdev(values) if len(values) > 1 else None
        agg.append(d)
    with (root / "aggregated_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0]))
        writer.writeheader(); writer.writerows(agg)
    print(f"Summarised {len(rows)} runs; {len(missing)} missing: " + ", ".join(missing[:10]))
    print("Seed-to-seed variation is descriptive, not a confidence interval; study may be incomplete.")


if __name__ == "__main__":
    main()
