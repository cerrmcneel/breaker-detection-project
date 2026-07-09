"""
Standalone test of the "position alone identifies MAINBREAKER" hypothesis,
evaluated directly against ground-truth labels (no model/detection noise).

Brainstormed 2026-07-05: since MAINBREAKER is visually near-identical to a
1-module MCB except for position/size, could a simple rule ("leftmost
component on the topmost rail = MAINBREAKER") replace vision-based
classification for this class? This was reportedly tested informally before
but never documented -- this script gives a real, reproducible number.

Tests two versions of the rule per image:
  1. "Leftmost box in the whole image" = MAINBREAKER
  2. "Leftmost box on the topmost row" = MAINBREAKER (rows grouped by Y-tolerance,
     matching SpatialHeuristicEngine.group_into_rows' own logic)
"""
import os
import pathlib

CLASS_MAP = ["MCB", "RCD", "RCD_SI", "MAINBREAKER", "OVERSURGE", "OTHER"]


def load_ground_truth(label_path, img_w, img_h):
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                gt_boxes.append({
                    "class": CLASS_MAP[class_id] if class_id < len(CLASS_MAP) else "UNKNOWN",
                    "box": [x1, y1, x2, y2],
                })
    return gt_boxes


def group_into_rows(boxes, y_tolerance=50):
    if not boxes:
        return []
    sorted_by_y = sorted(boxes, key=lambda b: b["box"][1])
    rows = [[sorted_by_y[0]]]
    for b in sorted_by_y[1:]:
        if abs(b["box"][1] - rows[-1][-1]["box"][1]) <= y_tolerance:
            rows[-1].append(b)
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b["box"][0])
    return rows


def evaluate_rule(gt_by_image, rule_name, picker):
    """picker(boxes) -> the box this rule nominates as MAINBREAKER, or None."""
    true_positives = 0     # rule picked a box, and it really is MAINBREAKER
    false_positives = 0    # rule picked a box, but it is NOT MAINBREAKER
    false_negatives = 0    # a real MAINBREAKER existed but the rule didn't pick it
    total_mainbreakers = 0
    images_with_mainbreaker = 0

    for boxes in gt_by_image:
        actual_mainbreakers = [b for b in boxes if b["class"] == "MAINBREAKER"]
        total_mainbreakers += len(actual_mainbreakers)
        if actual_mainbreakers:
            images_with_mainbreaker += 1

        picked = picker(boxes)
        if picked is None:
            false_negatives += len(actual_mainbreakers)
            continue

        if picked["class"] == "MAINBREAKER":
            true_positives += 1
            # any additional real MAINBREAKERs beyond the one picked are missed
            false_negatives += max(0, len(actual_mainbreakers) - 1)
        else:
            false_positives += 1
            false_negatives += len(actual_mainbreakers)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / total_mainbreakers if total_mainbreakers else 0.0

    print(f"\n--- Rule: {rule_name} ---")
    print(f"Images with >=1 real MAINBREAKER: {images_with_mainbreaker} / {len(gt_by_image)}")
    print(f"True positives:  {true_positives}")
    print(f"False positives: {false_positives}  (rule said MAINBREAKER, ground truth disagreed)")
    print(f"False negatives: {false_negatives}  (a real MAINBREAKER existed, rule missed it)")
    print(f"Precision: {precision:.2%}   Recall: {recall:.2%}")


def main():
    project_root = pathlib.Path(__file__).parent.parent.parent
    val_images_dir = project_root / "data" / "dataset" / "val" / "images"
    val_labels_dir = project_root / "data" / "dataset" / "val" / "labels"

    img_files = [
        f for f in val_images_dir.glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and not f.name.startswith("synth_panel_")
    ]

    # Need real image dimensions to convert normalized YOLO coords -> pixels
    import cv2
    gt_by_image = []
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        label_path = val_labels_dir / f"{img_path.stem}.txt"
        boxes = load_ground_truth(label_path, w_img, h_img)
        if boxes:
            gt_by_image.append(boxes)

    print(f"Evaluated on {len(gt_by_image)} real images with ground-truth labels.")

    def leftmost_in_image(boxes):
        return min(boxes, key=lambda b: b["box"][0]) if boxes else None

    def leftmost_in_top_row(boxes):
        rows = group_into_rows(boxes)
        if not rows:
            return None
        return rows[0][0]

    evaluate_rule(gt_by_image, "Leftmost box in whole image", leftmost_in_image)
    evaluate_rule(gt_by_image, "Leftmost box on topmost row", leftmost_in_top_row)


if __name__ == "__main__":
    main()
