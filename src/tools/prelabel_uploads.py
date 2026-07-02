#!/usr/bin/env python3
"""
prelabel_uploads.py — model-assisted pre-labeling for the human review loop.

Runs the trained YOLO detector over UNLABELED images in data/images/raw_uploads and
writes YOLO-format .txt pre-labels next to each image, so a human only has to *correct*
boxes in LabelImg instead of drawing them from scratch.

READ BEFORE RUNNING:
  * Use the RETRAINED model. The class indices written here come from models/best.pt;
    if you run this with the old (permuted) weights, the pre-labels inherit the same
    MCB/RCD inversion and waste your correction time. The script prints the model's
    modified time so you can confirm it's the fresh one.
  * Class-order integrity guard: the model's class order must match
    raw_uploads/classes.txt (and data.yaml). The script ABORTS on mismatch so it can
    never silently re-introduce the permutation bug.
  * These are PRE-LABELS — model guesses, not ground truth. Review/correct EVERY file in
    LabelImg before running mix_raw_uploads.

Usage:
    python -m src.tools.prelabel_uploads                  # all unlabeled raw_uploads
    python -m src.tools.prelabel_uploads --conf 0.30
    python -m src.tools.prelabel_uploads --overwrite      # re-do already-labeled too
"""
import os
import glob
import time
import argparse
import pathlib
from ultralytics import YOLO

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_class_order(classes_txt):
    if not os.path.exists(classes_txt):
        return None
    with open(classes_txt, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Model-assisted YOLO pre-labeling for human review")
    ap.add_argument("--images-dir", default=str(PROJECT_ROOT / "data" / "images" / "raw_uploads"))
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (lower = more candidate boxes to prune)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out-dir", default=None, help="where to write .txt (default: next to the images, LabelImg-style)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing .txt (default: skip already-labeled images)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N images (0 = all)")
    args = ap.parse_args()

    images_dir = args.images_dir
    out_dir = args.out_dir or images_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}. Retrain first, then copy weights to models/best.pt.")
        return

    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(args.model)))
    print(f"Using model: {args.model}  (modified {mtime})")
    print("  -> confirm this is your RETRAINED model, not the old permuted weights.\n")

    model = YOLO(args.model)
    names = [model.names[i] for i in range(len(model.names))]

    # Integrity guard: the model's class order must match raw_uploads/classes.txt (= data.yaml).
    classes_txt = os.path.join(images_dir, "classes.txt")
    order = load_class_order(classes_txt)
    if order is not None and order != names:
        print("ERROR: class-order mismatch — refusing to write potentially-permuted labels.")
        print(f"  model order      : {names}")
        print(f"  classes.txt order: {order}")
        print("  Align classes.txt with the model / data.yaml before pre-labeling.")
        return

    # Gather images; skip already-labeled unless --overwrite.
    images = []
    for p in sorted(glob.glob(os.path.join(images_dir, "*"))):
        if not p.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        # "Already labeled" means a .txt exists next to the image (the source of truth),
        # regardless of where we write pre-labels. Protects existing/human labels.
        if os.path.exists(os.path.join(images_dir, stem + ".txt")) and not args.overwrite:
            continue
        images.append(p)

    if args.limit > 0:
        images = images[:args.limit]

    if not images:
        print("No unlabeled images to pre-label.")
        return

    print(f"Pre-labeling {len(images)} image(s) (conf>={args.conf}, imgsz={args.imgsz})...")
    total_boxes = 0
    for p in images:
        r = model.predict(p, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        lines = []
        for b in r.boxes:
            cls = int(b.cls[0])
            x, y, w, h = b.xywhn[0].tolist()  # normalized YOLO center-format, straight from the model
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(os.path.join(out_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        total_boxes += len(lines)
        print(f"  {os.path.basename(p):48s} -> {len(lines)} boxes")

    print(f"\nWrote {len(images)} pre-label files ({total_boxes} boxes) to {out_dir}.")
    print("NEXT: review/correct EVERY file in LabelImg, THEN run "
          "`python -m src.data_gen.mix_raw_uploads`. These are model guesses, not ground truth.")


if __name__ == "__main__":
    main()
