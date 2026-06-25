"""
calibrate_hmm_confusion.py
--------------------------
Runs the trained YOLO model over real-world val images and builds an
EMPIRICAL confusion matrix from its actual predictions vs. ground truth.

The result is written into src/model/hmm_config.json as the new
`yolo_confusion` key, replacing the hand-coded approximations inside
hmm_corrector.py so the Viterbi emission function reflects how THIS
specific YOLO26n model actually confuses breaker classes.

Usage (from project root):
    .venv/Scripts/python.exe src/tools/calibrate_hmm_confusion.py
"""

import os
import sys
import json
import pathlib
import collections
import cv2
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
CLASSES          = ["MCB", "RCD", "RCD_SI", "MAINBREAKER", "OVERSURGE", "OTHER"]
MIN_CONF         = 0.20          # same threshold as pipeline_config.json
IOU_MATCH_THRESH = 0.40          # same as evaluate_pipeline.py
LAPLACE_ALPHA    = 0.01          # smoothing so no cell ever reaches true zero
VAL_IMAGES_DIR   = pathlib.Path("data/dataset/val/images")
VAL_LABELS_DIR   = pathlib.Path("data/dataset/val/labels")
HMM_CONFIG_PATH  = pathlib.Path("src/model/hmm_config.json")
YOLO_MODEL_PATH  = "models/best.pt"
# Only evaluate real-world images so calibration reflects field conditions
REAL_PREFIXES    = ("SPAIN_", "FRANCE_")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_ground_truth(label_path: pathlib.Path, img_w: int, img_h: int) -> list:
    """Parse YOLO txt label → list of {class, box:[x1,y1,x2,y2]}."""
    gt = []
    if not label_path.exists():
        return gt
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            gt.append({"class": CLASSES[cls_id], "box": [x1, y1, x2, y2]})
    return gt


def compute_iou(b1: list, b2: list) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(preds: list, gts: list) -> list:
    """
    Greedy IoU-based matching. Returns list of (true_class, pred_class) pairs
    for every matched (gt, pred) pair that clears the IoU threshold.
    """
    matched_gt = set()
    pairs = []
    for pred in preds:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            iou = compute_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx != -1 and best_iou >= IOU_MATCH_THRESH:
            matched_gt.add(best_idx)
            pairs.append((gts[best_idx]["class"], pred["class"]))
    return pairs


def normalize_confusion(counts: dict) -> dict:
    """
    Laplace-smooth and row-normalise the raw counts matrix.
    counts[true_cls][pred_cls] = int
    Returns matrix[true_cls][pred_cls] = float probability.
    """
    matrix = {}
    for true_cls in CLASSES:
        row = counts.get(true_cls, {})
        total = sum(row.get(p, 0) for p in CLASSES)
        matrix[true_cls] = {}
        for pred_cls in CLASSES:
            count = row.get(pred_cls, 0)
            matrix[true_cls][pred_cls] = (count + LAPLACE_ALPHA) / (total + len(CLASSES) * LAPLACE_ALPHA)
    return matrix

# ── Main ──────────────────────────────────────────────────────────────────────

def calibrate():
    if not VAL_IMAGES_DIR.exists():
        print(f"Error: val images dir not found at {VAL_IMAGES_DIR}")
        sys.exit(1)
    if not pathlib.Path(YOLO_MODEL_PATH).exists():
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
        sys.exit(1)

    print(f"Loading YOLO model from {YOLO_MODEL_PATH} ...")
    model = YOLO(YOLO_MODEL_PATH)

    # Real-world images only
    img_files = [
        f for f in VAL_IMAGES_DIR.glob("*")
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        and f.name.startswith(REAL_PREFIXES)
    ]
    if not img_files:
        print("No real-world validation images found (expected SPAIN_* or FRANCE_* prefixes).")
        sys.exit(1)

    print(f"Calibrating on {len(img_files)} real-world validation images...")

    # Raw count matrix: counts[true_cls][pred_cls]
    counts: dict = {c: collections.defaultdict(int) for c in CLASSES}
    total_matched = 0

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        label_path = VAL_LABELS_DIR / f"{img_path.stem}.txt"
        gts = load_ground_truth(label_path, w_img, h_img)
        if not gts:
            continue

        results = model.predict(img_path, conf=MIN_CONF, imgsz=1280, verbose=False)
        preds = []
        for r in results:
            for b in r.boxes:
                raw = b.xyxy[0]
                box = raw.tolist() if hasattr(raw, "tolist") else list(raw)
                preds.append({
                    "class": r.names[int(b.cls[0])],
                    "box":   box,
                    "conf":  float(b.conf[0])
                })

        pairs = match_predictions_to_gt(preds, gts)
        for true_cls, pred_cls in pairs:
            if true_cls in counts and pred_cls in CLASSES:
                counts[true_cls][pred_cls] += 1
                total_matched += 1

    if total_matched == 0:
        print("Error: no matched prediction/GT pairs found. Check paths and model.")
        sys.exit(1)

    print(f"\nTotal matched prediction-GT pairs: {total_matched}")

    # ── Print raw confusion matrix ────────────────────────────────────────────
    print("\nRaw Confusion Matrix (rows=True, cols=Predicted):")
    header = f"{'True \\ Pred':15s} | " + " | ".join(f"{c:10s}" for c in CLASSES)
    print(header)
    print("-" * len(header))
    for true_cls in CLASSES:
        row = " | ".join(f"{counts[true_cls].get(p, 0):10d}" for p in CLASSES)
        print(f"{true_cls:15s} | {row}")

    # ── Normalise ─────────────────────────────────────────────────────────────
    empirical_confusion = normalize_confusion(counts)

    print("\nEmpirical Confusion Matrix (Laplace-smoothed, row-normalised):")
    header2 = f"{'True \\ Pred':15s} | " + " | ".join(f"{c:10s}" for c in CLASSES)
    print(header2)
    print("-" * len(header2))
    for true_cls in CLASSES:
        row = " | ".join(f"{empirical_confusion[true_cls][p]:.3f}     " for p in CLASSES)
        print(f"{true_cls:15s} | {row}")

    # ── Write into hmm_config.json ────────────────────────────────────────────
    if not HMM_CONFIG_PATH.exists():
        print(f"\nError: {HMM_CONFIG_PATH} not found. Run train_hmm.py first.")
        sys.exit(1)

    with open(HMM_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["yolo_confusion"] = empirical_confusion

    with open(HMM_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"\n[OK] Empirical confusion matrix written to {HMM_CONFIG_PATH}")
    print("     Next: update HMMCorrector to load yolo_confusion from config,")
    print("     then re-run evaluate_pipeline.py to measure the improvement.")


if __name__ == "__main__":
    calibrate()
