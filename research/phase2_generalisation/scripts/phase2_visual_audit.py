import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from phase2_config import JIC_DIR, JIC_REFERENCE, OUT as OUTBASE

# Paths
predictions_path = str(JIC_REFERENCE)
image_root = str(JIC_DIR)
output_dir = str(OUTBASE / 'phase2_visual_audit')
os.makedirs(output_dir, exist_ok=True)

# Load predictions
df = pd.read_csv(predictions_path)

# Selected images
selected_images = [
    "Yellow_rust/YellowRust2280.png",
    "Yellow_rust/YellowRust2292.png",
    "Yellow_rust/YellowRust2324.png",
    "Yellow_rust/YellowRust2405.png",
    "Yellow_rust/YellowRust3037.png",
    "Yellow_rust/YellowRust3422.png",
    "Yellow_rust/YellowRust40.png",
    "Yellow_rust/YellowRust626.png",
    "Yellow_rust/YellowRust3856.png",
]

subset = df[df["image_id"].isin(selected_images)].copy()

# Keep the same order as selected_images
subset["image_id"] = pd.Categorical(subset["image_id"], categories=selected_images, ordered=True)
subset = subset.sort_values("image_id")

# Save selected prediction details
csv_out = os.path.join(output_dir, "selected_yellow_rust_cases.csv")
subset.to_csv(csv_out, index=False)

# Create figure
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

for ax, (_, row) in zip(axes, subset.iterrows()):
    rel_path = row["image_id"]
    img_path = os.path.join(image_root, *rel_path.split("/"))

    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"Image not found:\n{img_path}", ha="center", va="center")

    title = (
        f"{os.path.basename(rel_path)}\n"
        f"Actual: {row['actual']}\n"
        f"Pred: {row['predicted']} ({row['max_score']:.4f})\n"
        f"B:{row['score_Brown_rust']:.3f}  "
        f"H:{row['score_Healthy']:.3f}  "
        f"Y:{row['score_Yellow_rust']:.3f}"
    )
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.93,
    bottom=0.05,
    hspace=0.65,
    wspace=0.25
)
fig_out = os.path.join(output_dir, "yellow_rust_visual_audit.png")
plt.savefig(fig_out, dpi=200, bbox_inches="tight")
plt.show()

print("Saved figure to:", fig_out)
print("Saved CSV to:", csv_out)