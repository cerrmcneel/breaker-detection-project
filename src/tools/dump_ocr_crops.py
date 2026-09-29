import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.ocr_reader import OCRReader  # noqa: E402

CLASS_NAMES: Dict[int, str] = {
    0: "MCB",
    1: "RCD",
    2: "RCD_SI",
    3: "MAINBREAKER",
    4: "OVERSURGE",
    5: "OTHER",
}


def parse_label_file(label_path: Path, img_w: int, img_h: int) -> List[dict]:
    """Parses YOLO normalized box label file into absolute pixel coordinates."""
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = int(round((cx - w / 2) * img_w))
            y1 = int(round((cy - h / 2) * img_h))
            x2 = int(round((cx + w / 2) * img_w))
            y2 = int(round((cy + h / 2) * img_h))
            boxes.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"UNKNOWN_{cls_id}"),
                "box": [x1, y1, x2, y2],
            })
    return boxes


def dump_dataset_crops(
    dataset_dir: str,
    output_csv_path: str,
    margin: int = 12,
    device: Optional[str] = None,
    limit: Optional[int] = None,
) -> int:
    """Extracts crops for all real (non-synth_panel_) boxes across train and val splits,

    runs EasyOCR and OCRReader regex cleaning, and records results into a CSV.
    """
    import easyocr

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing EasyOCR on device={device}...")
    easyocr_reader = easyocr.Reader(["en"], gpu=(device == "cuda"))
    cleaner = OCRReader()._clean_ocr_text

    dataset_path = Path(dataset_dir)
    splits = ["train", "val"]
    rows = []
    total_processed = 0

    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        if not img_dir.exists():
            print(f"Warning: {img_dir} does not exist, skipping split '{split}'")
            continue

        img_files = sorted(img_dir.glob("*.*"))
        for img_p in img_files:
            if img_p.name.startswith("synth_panel_"):
                continue  # Rigor rule: strictly exclude synthetic panels by prefix

            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            img = cv2.imread(str(img_p))
            if img is None:
                print(f"Warning: Could not read image {img_p}")
                continue

            h_img, w_img = img.shape[:2]
            boxes = parse_label_file(lbl_p, w_img, h_img)

            for idx, box_info in enumerate(boxes):
                if limit is not None and total_processed >= limit:
                    break

                x1, y1, x2, y2 = box_info["box"]
                # Apply margin padding identical to pipeline.py
                px1 = max(0, x1 - margin)
                py1 = max(0, y1 - margin)
                px2 = min(w_img, x2 + margin)
                py2 = min(h_img, y2 + margin)

                raw_text = ""
                cleaned_verdict = ""
                tokens_info = []

                if px2 > px1 and py2 > py1:
                    crop_cv = img[py1:py2, px1:px2]
                    crop_rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
                    try:
                        ocr_results = easyocr_reader.readtext(crop_rgb)
                        raw_text = " ".join([res[1] for res in ocr_results])
                        cleaned_verdict = cleaner(raw_text)

                        # Capture detailed token geometry relative to crop
                        # res[0] is list of 4 points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        for res in ocr_results:
                            pts = res[0]
                            t_text = str(res[1])
                            t_conf = float(res[2])
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            token_cx = sum(xs) / len(xs)
                            token_cy = sum(ys) / len(ys)
                            # Original unpadded box inside this crop is:
                            # [x1 - px1, y1 - py1, x2 - px1, y2 - py1]
                            orig_x1 = int(x1 - px1)
                            orig_y1 = int(y1 - py1)
                            orig_x2 = int(x2 - px1)
                            orig_y2 = int(y2 - py1)
                            is_inside_orig = bool((orig_x1 <= token_cx <= orig_x2) and (orig_y1 <= token_cy <= orig_y2))

                            tokens_info.append({
                                "text": t_text,
                                "conf": round(t_conf, 4),
                                "center": [round(float(token_cx), 1), round(float(token_cy), 1)],
                                "orig_box_in_crop": [orig_x1, orig_y1, orig_x2, orig_y2],
                                "inside_orig": is_inside_orig,
                            })
                    except Exception as err:
                        print(f"Error on {img_p.name} box {idx}: {err}")

                rows.append({
                    "split": split,
                    "image_name": img_p.name,
                    "box_index": idx,
                    "gt_class_id": box_info["class_id"],
                    "gt_class": box_info["class_name"],
                    "box_x1": x1,
                    "box_y1": y1,
                    "box_x2": x2,
                    "box_y2": y2,
                    "pad_x1": px1,
                    "pad_y1": py1,
                    "pad_x2": px2,
                    "pad_y2": py2,
                    "margin": margin,
                    "raw_text": raw_text,
                    "cleaned_verdict": cleaned_verdict,
                    "tokens_json": json.dumps(tokens_info),
                })
                total_processed += 1

                if total_processed % 100 == 0:
                    print(f"Processed {total_processed} boxes...")

            if limit is not None and total_processed >= limit:
                break

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} records to {output_csv_path}")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Dump OCR crops from real dataset images.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\ironhack\labs\breaker-detection-project\data\dataset",
        help="Path to dataset root containing train/ and val/ splits.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="ocr_crops_dump.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=12,
        help="Padding margin in pixels around box (default: 12).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of crops processed (for testing).",
    )
    args = parser.parse_args()

    count = dump_dataset_crops(
        dataset_dir=args.data_dir,
        output_csv_path=args.output,
        margin=args.margin,
        limit=args.limit,
    )
    print(f"Done. Processed {count} crops.")


if __name__ == "__main__":
    sys.exit(main())
