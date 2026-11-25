"""Wheat Rust AI Phase 2 — manually checked background intervention (9-image PILOT).

Run from repository root:
  python phase2_background_pilot.py annotate
  python phase2_background_pilot.py evaluate

Annotate each image: left-click points around visible leaf/plant foreground.
N closes the current polygon; multiple polygons are allowed (e.g. separate leaves).
S saves the mask and proceeds; R resets; ESC exits without saving current mask.

Scientific limits: masks are manually drawn; fixed grey replacement introduces unnatural
edges and colours; nine images were selected based on prior outcomes. This is a
mechanism-screening pilot, not independent evidence of a causal mechanism.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from PIL import Image, ImageDraw

from phase2_config import JIC_DIR, MODEL as CONFIG_MODEL, JIC_REFERENCE, OUT as OUTBASE

ROOT = JIC_DIR
MODEL = CONFIG_MODEL
REF = JIC_REFERENCE
OUT = OUTBASE / 'phase2_background_pilot'
MASKS = OUT / "masks"
CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
SELECTED = [
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
FILL_RGB = (128, 128, 128)  # single prespecified, not optimized


def image_path(rel):
    return ROOT.joinpath(*rel.split("/"))


def mask_path(rel):
    return MASKS / (Path(rel).stem + "_mask.png")


def annotate():
    import cv2  # pip install opencv-python

    MASKS.mkdir(parents=True, exist_ok=True)
    for number, rel in enumerate(SELECTED, 1):
        image = image_path(rel)
        if not image.is_file():
            raise FileNotFoundError(image)
        rgb = np.asarray(Image.open(image).convert("RGB"))
        h, w = rgb.shape[:2]
        scale = min(1., 1000. / w, 700. / h)
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        original = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        polygons = []
        current = []
        win = f"Leaf outlines {number}/9: {Path(rel).name}"

        def on_mouse(event, x, y, flags, userdata):
            if event == cv2.EVENT_LBUTTONDOWN:
                current.append((min(w-1, int(x / scale)), min(h-1, int(y / scale))))

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            frame = original.copy()
            for poly in polygons:
                pts = np.asarray([(round(x*scale),round(y*scale)) for x,y in poly], np.int32)
                cv2.polylines(frame, [pts], True, (0,255,255), 2)
            if current:
                pts = np.asarray([(round(x*scale),round(y*scale)) for x,y in current], np.int32)
                cv2.polylines(frame, [pts], False, (255,100,0), 2)
                for px, py in pts:
                    cv2.circle(frame, (int(px),int(py)), 3, (255,100,0), -1)
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(display, "CLICK outline | N close polygon | S save | R reset | ESC exit", (8, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 2)
            cv2.putText(display, "CLICK outline | N close polygon | S save | R reset | ESC exit", (8, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (0,0,0), 1)
            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('n'):
                if len(current) >= 3:
                    polygons.append(current.copy())
                    current.clear()
                else:
                    print("Need at least 3 clicked points to close a polygon")
            elif key == ord('r'):
                polygons.clear()
                current.clear()
            elif key == ord('s'):
                if len(current) >= 3:
                    polygons.append(current.copy())
                    current.clear()
                if not polygons:
                    print("No valid polygon; please outline at least one leaf")
                    continue
                mask = Image.new("L", (w,h), 0)
                draw = ImageDraw.Draw(mask)
                for poly in polygons:
                    draw.polygon(poly, fill=255)
                fraction = float(np.asarray(mask).mean() / 255.)
                if fraction < 0.01 or fraction > 0.95:
                    print(f"Foreground coverage {fraction:.1%} unusual; press R to retry or S again to save")
                    # Treat as warning, not automatic acceptance. User must press s again.
                    if not getattr(on_mouse, 'accept_warning', False):
                        on_mouse.accept_warning = True
                        continue
                mask.save(mask_path(rel))
                print(f"Saved {mask_path(rel)} | foreground {fraction:.1%}")
                break
            elif key == 27:
                cv2.destroyAllWindows()
                print("Stopped. Already saved masks remain available.")
                return
        cv2.destroyWindow(win)
    cv2.destroyAllWindows()
    print("All nine masks saved. Review masks visually before running evaluate.")


def evaluate():
    import tensorflow as tf
    from src.data import EXPECTED_CLASSES, IMAGE_SIZE, load_split

    if tuple(EXPECTED_CLASSES) != CLASSES or tuple(IMAGE_SIZE) != (224, 224):
        raise RuntimeError("Model data assumptions differ from audit")
    for path in (MODEL, REF, ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    if any(not mask_path(r).is_file() for r in SELECTED):
        missing = [r for r in SELECTED if not mask_path(r).is_file()]
        raise RuntimeError(f"Please annotate all masks first. Missing: {missing}")

    import pandas as pd
    reference = pd.read_csv(REF).set_index('image_id')
    if reference.index.has_duplicates:
        raise RuntimeError("Duplicated IDs in reference predictions")
    model = tf.keras.models.load_model(MODEL, compile=False)
    ds, paths = load_split(ROOT, shuffle=False, batch_size=2, return_paths=True)
    target = set(SELECTED)
    rows = []
    seen = 0
    for imgs, labels in ds:
        for k in range(int(imgs.shape[0])):
            rel = Path(paths[seen + k]).relative_to(ROOT).as_posix()
            if rel not in target:
                continue
            x = tf.cast(imgs[k:k+1], tf.float32)
            base = np.asarray(model(x, training=False), dtype=float)[0]
            if rel not in reference.index:
                raise RuntimeError(f"No stored prediction for {rel}")
            refrow = reference.loc[rel]
            if CLASSES[int(np.argmax(np.asarray(labels[k])))] != refrow.actual:
                raise RuntimeError(f"Label mismatch for {rel}")
            old = np.array([float(refrow['score_'+c]) for c in CLASSES])
            if not np.allclose(base, old, atol=1e-3, rtol=0):
                raise RuntimeError(f"Baseline mismatch {rel}: max delta {np.max(np.abs(base-old)):.6f}")

            # Drawn at original resolution; nearest-neighbour resizing preserves binary mask.
            original_size = Image.open(image_path(rel)).size
            mask_img = Image.open(mask_path(rel)).convert('L')
            if mask_img.size != original_size:
                raise RuntimeError(f"Mask dimension mismatch {rel}")
            mask224 = mask_img.resize((224,224), resample=Image.Resampling.NEAREST)
            mask_np = np.asarray(mask224, dtype=np.uint8) >= 128
            coverage = float(mask_np.mean())
            if not (0.01 <= coverage <= 0.95):
                raise RuntimeError(f"Suspicious foreground fraction for {rel}: {coverage:.1%}")
            mask_tf = tf.convert_to_tensor(mask_np[None,:,:,None])
            masked = tf.where(mask_tf, x, tf.constant(FILL_RGB, dtype=tf.float32)[None,None,None,:])
            changed = np.asarray(model(masked, training=False), dtype=float)[0]
            row = {'image_id':rel, 'actual':refrow.actual, 'foreground_fraction':coverage,
                   'baseline_predicted':CLASSES[int(base.argmax())],
                   'background_replaced_predicted':CLASSES[int(changed.argmax())],
                   'baseline_correct':int(CLASSES[int(base.argmax())] == refrow.actual),
                   'background_replaced_correct':int(CLASSES[int(changed.argmax())] == refrow.actual)}
            for j,c in enumerate(CLASSES):
                row['baseline_'+c] = float(base[j])
                row['background_replaced_'+c] = float(changed[j])
            rows.append(row)
            print(f"{len(rows)}/9 {rel}: {row['baseline_predicted']} -> {row['background_replaced_predicted']}", flush=True)
        seen += int(imgs.shape[0])
    if len(rows) != len(SELECTED):
        raise RuntimeError(f"Expected 9 selected images but evaluated {len(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    result = pd.DataFrame(rows).set_index('image_id').loc[SELECTED].reset_index()
    result.to_csv(OUT / 'paired_background_pilot.csv', index=False)
    report = {'status':'exploratory_selected_nine_image_pilot_not_independent',
              'model_unchanged':True,'background_fill_rgb':list(FILL_RGB),
              'n':len(result),'changed_predictions':int((result.baseline_predicted != result.background_replaced_predicted).sum()),
              'wrong_to_right':int(((result.baseline_correct == 0)&(result.background_replaced_correct == 1)).sum()),
              'right_to_wrong':int(((result.baseline_correct == 1)&(result.background_replaced_correct == 0)).sum()),
              'caveat':'Foreground manually annotated; masks can be imperfect, grey backgrounds introduce artifacts; outcomes selected before annotation. Not proof of background causation.'}
    (OUT/'pilot_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', OUT/'paired_background_pilot.csv')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('annotate','evaluate'))
    args=parser.parse_args()
    if args.mode=='annotate': annotate()
    else: evaluate()
