"""Quality-control contact sheet for nine manually annotated Wheat Rust AI masks.

Run from repository root: python phase2_review_masks.py
No inference, training, or mask modification.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from phase2_config import JIC_DIR, OUT as OUTBASE
IMAGE_ROOT = JIC_DIR / 'Yellow_rust'
MASK_ROOT = OUTBASE / 'phase2_background_pilot' / 'masks'
OUT = OUTBASE / 'phase2_background_pilot' / 'mask_quality_review.png'
FILENAMES = [
    "YellowRust2280.png", "YellowRust2292.png", "YellowRust2324.png",
    "YellowRust2405.png", "YellowRust3037.png", "YellowRust3422.png",
    "YellowRust40.png", "YellowRust626.png", "YellowRust3856.png",
]
CELL_W, CELL_H = 340, 310
DISPLAY_W, DISPLAY_H = 315, 245


def fit_on_canvas(image):
    image = image.copy()
    image.thumbnail((DISPLAY_W, DISPLAY_H), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL_W, DISPLAY_H), "white")
    canvas.paste(image, ((CELL_W-image.width)//2, (DISPLAY_H-image.height)//2))
    return canvas


def main():
    sheet = Image.new("RGB", (3 * CELL_W, len(FILENAMES) * CELL_H + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for j, heading in enumerate(["Original", "Mask overlay", "Grey background"]):
        draw.text((j*CELL_W+10, 8), heading, fill="black")
    for i, filename in enumerate(FILENAMES):
        original_path = IMAGE_ROOT / filename
        mask_path = MASK_ROOT / filename.replace(".png", "_mask.png")
        if not original_path.is_file():
            raise FileNotFoundError(f"Missing original: {original_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {mask_path}")
        orig = Image.open(original_path).convert("RGB")
        raw_mask = Image.open(mask_path).convert("L")
        if raw_mask.size != orig.size:
            print(f"Size differs for {filename}: original {orig.size}, mask {raw_mask.size}; resizing mask NEAREST for QC display")
        mask = raw_mask.resize(orig.size, Image.Resampling.NEAREST)
        m = np.asarray(mask) >= 128
        x = np.asarray(orig, dtype=np.uint8)
        # Red overlay over retained foreground; original remains visible beneath.
        overlay = x.astype(np.float32).copy()
        overlay[m] = 0.65 * overlay[m] + 0.35 * np.array([255, 30, 30])
        overlay = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))
        grey = x.copy()
        grey[~m] = (127, 127, 127)
        variants = [orig, overlay, Image.fromarray(grey)]
        y = 30 + i * CELL_H
        pct = 100 * float(m.mean())
        for j, variant in enumerate(variants):
            sheet.paste(fit_on_canvas(variant), (j*CELL_W, y+32))
        draw.text((10, y+8), f"{filename}  |  retained foreground: {pct:.1f}%", fill="black")
        print(f"{filename}: foreground {pct:.1f}%")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(f"Saved mask-quality sheet: {OUT}")


if __name__ == "__main__":
    main()
