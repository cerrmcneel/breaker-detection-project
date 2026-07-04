#!/usr/bin/env python3
"""
Phase 0 of the RCD test-button detector plan (directives/button_detector_plan.md).

Assembles a binary (button / no-button) dataset:

- Training / val crops come from the pre-labeled seed library (data/seeds/*), mapped to
  button presence by their component class:
      positives (button):    RCD, RCD_SI
      negatives (no button): MCB, MAINBREAKER, OVERSURGE, OTHER
- A REAL held-out test set is carved from the real (non-synthetic) validation photos by
  cropping their labeled MCB / RCD / RCD_SI boxes. Seeds -> train, real photo crops -> test,
  so the detector is never evaluated on its own training data (the Gate A de-risk set).

Outputs (under --out-dir, default data/button_dataset/):
    manifest_train.csv, manifest_val.csv      (reference seed crop paths, no copy)
    real_test/button/*, real_test/no_button/* (cropped real-photo regions)
    manifest_real_test.csv
    summary.json   (class counts + suggested inverse-frequency class weights)
"""
import os
import re
import csv
import json
import random
import argparse
import pathlib
from PIL import Image

POSITIVE_BASES = {"RCD", "RCD_SI"}
NEGATIVE_BASES = {"MCB", "MAINBREAKER", "OVERSURGE", "OTHER"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# YOLO class index -> name (matches data.yaml / evaluate_pipeline.CLASS_MAP)
CLASS_MAP = ["MCB", "RCD", "RCD_SI", "MAINBREAKER", "OVERSURGE", "OTHER"]
# Real-test is carved only from the confusable axis the override actually touches.
REAL_TEST_CLASSES = {"MCB", "RCD", "RCD_SI"}
MIN_CROP_PX = 8
# Real-image detection excludes by the synthetic prefix rather than allowlisting
# real-image prefixes -- an allowlist (previously "SPAIN_"/"FRANCE_" only)
# silently drifts out of date every time a new real naming convention is added
# (e.g. scraped_approved_*).
SYNTH_PREFIX = "synth_panel_"


def base_class_from_folder(folder_name):
    """RCD_SI_2 -> RCD_SI, MCB_1 -> MCB, OVERSURGE -> OVERSURGE."""
    m = re.match(r"^(.*)_\d+$", folder_name)
    return m.group(1) if m else folder_name


def label_for_base(base):
    if base in POSITIVE_BASES:
        return 1
    if base in NEGATIVE_BASES:
        return 0
    return None  # unknown folder -> skip


def collect_seed_crops(seeds_dir):
    """Returns (rows, skipped) where rows = [(abs_path, label, source_class), ...]."""
    rows = []
    skipped = 0
    if not os.path.isdir(seeds_dir):
        return rows, skipped
    for folder in sorted(os.listdir(seeds_dir)):
        fpath = os.path.join(seeds_dir, folder)
        if not os.path.isdir(fpath):
            continue
        base = base_class_from_folder(folder)
        label = label_for_base(base)
        if label is None:
            continue
        for fn in os.listdir(fpath):
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            p = os.path.join(fpath, fn)
            try:
                with Image.open(p) as im:
                    im.verify()  # cheap integrity check (handles corrupt / unreadable)
            except Exception:
                skipped += 1
                continue
            rows.append((os.path.abspath(p), label, base))
    return rows, skipped


def parse_yolo_label(label_path):
    items = []
    if not os.path.exists(label_path):
        return items
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                items.append((cid, cx, cy, w, h))
    return items


def carve_real_test(val_images, val_labels, out_dir):
    """Crop MCB/RCD/RCD_SI boxes out of real (non-synthetic) val photos into a real test set."""
    rows = []
    pos_dir = os.path.join(out_dir, "real_test", "button")
    neg_dir = os.path.join(out_dir, "real_test", "no_button")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    if not os.path.isdir(val_images):
        return rows

    for fn in sorted(os.listdir(val_images)):
        if fn.startswith(SYNTH_PREFIX):
            continue
        if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
            continue
        img_path = os.path.join(val_images, fn)
        stem = os.path.splitext(fn)[0]
        label_path = os.path.join(val_labels, stem + ".txt")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        for i, (cid, cx, cy, w, h) in enumerate(parse_yolo_label(label_path)):
            cname = CLASS_MAP[cid] if cid < len(CLASS_MAP) else "UNKNOWN"
            if cname not in REAL_TEST_CLASSES:
                continue
            x1 = int((cx - w / 2) * W); y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W); y2 = int((cy + h / 2) * H)
            x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W));     y2 = max(0, min(y2, H))
            if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
                continue
            label = 1 if cname in POSITIVE_BASES else 0
            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{stem}_{i}_{cname}.jpg"
            out_path = os.path.join(pos_dir if label else neg_dir, out_name)
            crop.save(out_path, quality=95)
            rows.append((os.path.abspath(out_path), label, cname, fn))
    return rows


def write_manifest(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def counts(rows, idx=1):
    pos = sum(1 for r in rows if r[idx] == 1)
    neg = sum(1 for r in rows if r[idx] == 0)
    return pos, neg


def main():
    project_root = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Phase 0: assemble button-detector dataset")
    ap.add_argument("--seeds-dir", default=str(project_root / "data" / "seeds"))
    ap.add_argument("--val-images", default=str(project_root / "data" / "dataset" / "val" / "images"))
    ap.add_argument("--val-labels", default=str(project_root / "data" / "dataset" / "val" / "labels"))
    ap.add_argument("--out-dir", default=str(project_root / "data" / "button_dataset"))
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    # ---- Seed crops -> train / val (stratified by label) ----
    seed_rows, skipped = collect_seed_crops(args.seeds_dir)
    pos_rows = [r for r in seed_rows if r[1] == 1]
    neg_rows = [r for r in seed_rows if r[1] == 0]
    random.shuffle(pos_rows)
    random.shuffle(neg_rows)

    def split(rows):
        n_val = int(len(rows) * args.val_split)
        return rows[n_val:], rows[:n_val]  # train, val

    pos_train, pos_val = split(pos_rows)
    neg_train, neg_val = split(neg_rows)
    train_rows = pos_train + neg_train
    val_rows = pos_val + neg_val
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    write_manifest(os.path.join(args.out_dir, "manifest_train.csv"),
                   [(p, l, c, "train") for (p, l, c) in train_rows],
                   ["path", "label", "source_class", "split"])
    write_manifest(os.path.join(args.out_dir, "manifest_val.csv"),
                   [(p, l, c, "val") for (p, l, c) in val_rows],
                   ["path", "label", "source_class", "split"])

    # ---- Real held-out test ----
    real_rows = carve_real_test(args.val_images, args.val_labels, args.out_dir)
    write_manifest(os.path.join(args.out_dir, "manifest_real_test.csv"),
                   real_rows, ["path", "label", "source_class", "origin_image"])

    # ---- Summary + inverse-frequency class weights (for Phase 1 BCE) ----
    tr_pos, tr_neg = counts(train_rows)
    va_pos, va_neg = counts(val_rows)
    rt_pos, rt_neg = counts(real_rows)
    total = tr_pos + tr_neg
    w_pos = round(total / (2 * tr_pos), 4) if tr_pos else 0.0
    w_neg = round(total / (2 * tr_neg), 4) if tr_neg else 0.0

    summary = {
        "seed_crops_total": len(seed_rows),
        "seed_crops_skipped_unreadable": skipped,
        "train": {"button": tr_pos, "no_button": tr_neg},
        "val": {"button": va_pos, "no_button": va_neg},
        "real_test": {"button": rt_pos, "no_button": rt_neg},
        "suggested_class_weights": {"button": w_pos, "no_button": w_neg},
        "val_split": args.val_split,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nManifests + real_test crops written to: {args.out_dir}")
    if rt_pos == 0 or rt_neg == 0:
        print("WARNING: real-test set is missing a class. Gate A de-risk needs BOTH "
              "button and no_button real crops — check for real (non-synthetic) val labels.")


if __name__ == "__main__":
    main()
