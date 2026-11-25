import csv
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = Path(__file__).resolve().parent / "per_class_recall.csv"

classes = ["Brown_rust", "Healthy", "Yellow_rust"]
expected_supports = (128, 122, 239)
groups = defaultdict(list)

for size in (150, 300):
    for mix in (0, 25, 50):
        for seed in (42, 43, 44):
            condition = f"n{size}_mix{mix:02d}_seed{seed}"
            path = root / condition / "results/external/confusion_matrix.csv"

            with path.open(newline="", encoding="utf-8") as f:
                matrix = [
                    [int(value) for value in row]
                    for row in csv.reader(f) if row
                ]

            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError(f"Invalid matrix: {condition}")

            supports = tuple(sum(row) for row in matrix)

            if supports != expected_supports:
                raise ValueError(
                    f"Unexpected class supports for {condition}: {supports}"
                )

            for i, class_name in enumerate(classes):
                recall = matrix[i][i] / supports[i] * 100
                groups[(size, mix, class_name)].append(recall)

rows = []

for size in (150, 300):
    for mix in (0, 25, 50):
        for class_name in classes:
            values = groups[(size, mix, class_name)]

            if len(values) != 3:
                raise ValueError("Expected exactly three seeds")

            rows.append([
                size,
                mix,
                class_name,
                round(statistics.mean(values), 4),
                round(min(values), 4),
                round(max(values), 4),
                len(values)
            ])

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "images_per_class",
        "source_mix_percent",
        "class",
        "mean_recall_percent",
        "min_recall_percent",
        "max_recall_percent",
        "seeds"
    ])
    writer.writerows(rows)

print("PER-CLASS ANALYSIS VERIFIED")
print("Conditions:", 6)
print("Runs:", 18)
print("Output rows:", len(rows))
print("Output:", output)
