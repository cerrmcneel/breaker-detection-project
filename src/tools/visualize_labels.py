#!/usr/bin/env python3
"""
visualize_labels.py — render GROUND-TRUTH YOLO labels onto dataset images so you can
VISUALLY verify that each box has the correct class.

Why this exists: structural checks (scratch_check_dataset.py) validate that class IDs are
in range and coords are in bounds — but they CANNOT catch a wrong class on the right box,
which is exactly how the classes.txt permutation slipped through. The only reliable check
for that is drawing the labels and looking.

Writes annotated copies to data/label_check/<split>/ (open them in any image viewer and
flip through). It never modifies the dataset.

Usage:
    python -m src.tools.visualize_labels                      # val split, all images
    python -m src.tools.visualize_labels --split val --real   # only human-labeled real images
    python -m src.tools.visualize_labels --split train --limit 40
    python -m src.tools.visualize_labels --images-dir X --labels-dir Y --out-dir Z
"""
import argparse
import glob
import os
import pathlib
from collections import Counter

import cv2

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Fallback if data.yaml can't be read (same convention as test_homography_warp.py).
DEFAULT_NAMES = {0: 'MCB', 1: 'RCD', 2: 'RCD_SI', 3: 'MAINBREAKER', 4: 'OVERSURGE', 5: 'OTHER'}
CLASS_COLORS = {  # BGR; just needs to be distinct per class
    0: (0, 242, 254), 1: (0, 0, 255), 2: (255, 0, 255),
    3: (0, 255, 0), 4: (255, 255, 0), 5: (255, 255, 255),
}


def load_names(yaml_path):
    """Read class names from data.yaml (single source of truth); None on failure."""
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        names = d.get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
    except Exception:
        pass
    return None


def parse_yolo(label_path):
    items = []
    if not os.path.exists(label_path):
        return items
    for line in open(label_path):
        p = line.split()
        if len(p) >= 5:
            items.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return items


def main():
    ap = argparse.ArgumentParser(description="Render GT YOLO labels onto images for visual verification")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="annotate at most N images (0 = all)")
    ap.add_argument("--real", action="store_true", help="only human-labeled real images (exclude synth_*)")
    args = ap.parse_args()

    images_dir = args.images_dir or str(PROJECT_ROOT / "data" / "dataset" / args.split / "images")
    labels_dir = args.labels_dir or str(PROJECT_ROOT / "data" / "dataset" / args.split / "labels")
    out_dir = args.out_dir or str(PROJECT_ROOT / "data" / "label_check" / args.split)
    os.makedirs(out_dir, exist_ok=True)

    names = load_names(str(PROJECT_ROOT / "data.yaml")) or DEFAULT_NAMES

    imgs = [p for p in sorted(glob.glob(os.path.join(images_dir, "*")))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    if args.real:
        imgs = [p for p in imgs if not os.path.basename(p).startswith("synth")]
    if args.limit > 0:
        imgs = imgs[:args.limit]

    if not imgs:
        print(f"No images found in {images_dir} (real={args.real}).")
        return

    total = Counter()
    n_done = n_missing = 0
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            print(f"  [skip corrupt] {os.path.basename(p)}")
            continue
        H, W = img.shape[:2]
        thick = max(2, round(min(H, W) / 500))
        fs = max(0.5, min(H, W) / 1400)
        stem = os.path.splitext(os.path.basename(p))[0]
        boxes = parse_yolo(os.path.join(labels_dir, stem + ".txt"))
        if not boxes:
            n_missing += 1
        for cid, cx, cy, w, h in boxes:
            x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
            x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
            color = CLASS_COLORS.get(cid, (200, 200, 200))
            cname = names.get(cid, f"?{cid}")
            total[cname] += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            label = f"{cid}:{cname}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), img)
        n_done += 1

    print(f"\nAnnotated {n_done} images -> {out_dir}")
    if n_missing:
        print(f"  ({n_missing} image(s) had no label file)")
    print("Class instances drawn (sanity-check the distribution):")
    for k, v in sorted(total.items(), key=lambda x: -x[1]):
        print(f"  {k:12s}: {v}")
    print(f"\nOpen {out_dir} and flip through — confirm each colored box's class label matches the real device.")


if __name__ == "__main__":
    main()
