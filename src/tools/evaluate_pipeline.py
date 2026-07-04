import os
import time
import pathlib
import numpy as np
import cv2
from src.model.pipeline import PanelSafePipeline

# YOLO label index mapping for PanelSafe (data.yaml)
CLASS_MAP = ["MCB", "RCD", "RCD_SI", "MAINBREAKER", "OVERSURGE", "OTHER"]

def load_ground_truth(label_path, img_w, img_h):
    """
    Parses a YOLO txt label file and converts normalized coordinates 
    to absolute pixel coordinates [x1, y1, x2, y2].
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                # Convert from normalized center/width/height to absolute x1,y1,x2,y2
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                
                gt_boxes.append({
                    "class": CLASS_MAP[class_id] if class_id < len(CLASS_MAP) else "UNKNOWN",
                    "box": [x1, y1, x2, y2]
                })
    # Sort left-to-right to compare sequence order
    gt_boxes.sort(key=lambda item: item["box"][0])
    return gt_boxes

def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def evaluate_on_real_dataset(pipeline):
    """
    Evaluates the pipeline on the 23 real-world validation images.
    Returns accuracy metrics.
    """
    project_root = pathlib.Path(__file__).parent.parent.parent
    val_images_dir = project_root / "data" / "dataset" / "val" / "images"
    val_labels_dir = project_root / "data" / "dataset" / "val" / "labels"
    
    if not val_images_dir.exists():
        print(f"Error: validation folder not found at {val_images_dir}")
        return None

    # Filter for real-world images. Exclude by the synthetic prefix rather than
    # allowlisting real-image prefixes -- an allowlist (previously "SPAIN_"/
    # "FRANCE_" only) silently drifts out of date every time a new real naming
    # convention is added (e.g. scraped_approved_*), and had been silently
    # excluding all Round-2 real val images from every evaluation run.
    img_files = [
        f for f in val_images_dir.glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and
        not f.name.startswith("synth_panel_")
    ]

    if not img_files:
        # Fallback to any images if no real-world files found (for debugging)
        img_files = list(val_images_dir.glob("*.jpg"))[:10]
        print(f"No real-world files found. Falling back to first {len(img_files)} validation images.")

    total_real_images = len(img_files)
    if total_real_images == 0:
        print("No validation images found.")
        return None

    total_gt_boxes = 0
    correct_localizations = 0
    correct_classifications = 0
    correct_sequences = 0
    total_latency = 0.0

    for img_path in img_files:
        # Load image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]
        
        # Load ground truth
        label_path = val_labels_dir / f"{img_path.stem}.txt"
        gt_items = load_ground_truth(label_path, w_img, h_img)
        total_gt_boxes += len(gt_items)
        
        # Run inference
        t0 = time.time()
        preds = pipeline.run_inference(str(img_path))
        latency = (time.time() - t0) * 1000
        total_latency += latency
        
        # Sort predictions left-to-right
        preds_sorted = sorted(preds, key=lambda item: item["box"][0])
        
        # 1. Bounding Box & Class Accuracy
        matched_gt = set()
        img_correct_localizations = 0
        img_correct_classifications = 0
        
        for p in preds_sorted:
            best_iou = 0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_items):
                if idx in matched_gt:
                    continue
                iou = compute_iou(p["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # Successful localization threshold (IoU > 0.4)
            if best_gt_idx != -1 and best_iou >= 0.4:
                matched_gt.add(best_gt_idx)
                img_correct_localizations += 1
                correct_localizations += 1
                
                # Check classification correctness
                if p["class"] == gt_items[best_gt_idx]["class"]:
                    correct_classifications += 1
                    img_correct_classifications += 1

        # 2. Sequence Accuracy (Whole board matches exactly)
        # Sequence matches if localized and classified perfectly
        if len(gt_items) > 0 and len(preds_sorted) == len(gt_items):
            seq_match = True
            for idx, p in enumerate(preds_sorted):
                # Simple IoU match with corresponding sorted index
                iou = compute_iou(p["box"], gt_items[idx]["box"])
                if iou < 0.4 or p["class"] != gt_items[idx]["class"]:
                    seq_match = False
                    break
            if seq_match:
                correct_sequences += 1

    avg_latency = total_latency / total_real_images if total_real_images > 0 else 0
    loc_acc = correct_localizations / total_gt_boxes if total_gt_boxes > 0 else 0
    cls_acc = correct_classifications / total_gt_boxes if total_gt_boxes > 0 else 0
    seq_acc = correct_sequences / total_real_images if total_real_images > 0 else 0

    return {
        "images_count": total_real_images,
        "localization_acc": loc_acc,
        "classification_acc": cls_acc,
        "sequence_acc": seq_acc,
        "avg_latency_ms": avg_latency
    }

def run_ablation_study():
    """Runs the ablation study systematically across different configuration flags."""
    project_root = pathlib.Path(__file__).parent.parent.parent
    config_path = project_root / "src" / "model" / "pipeline_config.json"
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
        
    print("\n==================================================")
    print("      PANELSAFE PIPELINE ABLATION STUDY RUNNER    ")
    print("==================================================\n")

    # Define ablation scenarios to test
    scenarios = [
        {
            "name": "YOLO Baseline (Single-Stage)",
            "updates": {"classifier_mode": "single_stage", "use_hmm": False}
        },
        {
            "name": "YOLO + HMM Sequence Corrector",
            "updates": {"classifier_mode": "single_stage", "use_hmm": True}
        },
        {
            "name": "Two-Stage Pipeline (Stage 1 Bbox + Stage 2 Classifier)",
            "updates": {"classifier_mode": "two_stage", "use_hmm": False}
        },
        {
            "name": "Two-Stage Pipeline + HMM Corrector",
            "updates": {"classifier_mode": "two_stage", "use_hmm": True}
        }
    ]

    # Backup original configuration
    original_config = config_path.read_text(encoding="utf-8")

    results_table = []
    
    try:
        pipeline = PanelSafePipeline(config_path=str(config_path))
        
        for s in scenarios:
            print(f"Running evaluation scenario: {s['name']}...")
            
            # Apply temporary config update in pipeline object directly
            for k, v in s["updates"].items():
                pipeline.config[k] = v
                
            # If two_stage, make sure weights are loaded/mocked
            if s["updates"].get("classifier_mode") == "two_stage" and not pipeline.crop_classifier:
                pipeline.load_crop_classifier()
            
            # Run
            metrics = evaluate_on_real_dataset(pipeline)
            
            if metrics:
                results_table.append({
                    "name": s["name"],
                    "loc_acc": metrics["localization_acc"],
                    "cls_acc": metrics["classification_acc"],
                    "seq_acc": metrics["sequence_acc"],
                    "latency": metrics["avg_latency_ms"]
                })
    finally:
        # Restore configuration
        config_path.write_text(original_config, encoding="utf-8")

    # Output Results
    print("\n### Ablation Study Performance Matrix (Real-World Images)\n")
    print("| Configuration | Box Localization (Recall) | Component Classification | Sequence Accuracy | Avg Latency (ms) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for r in results_table:
        print(f"| {r['name']} | {r['loc_acc']:.2%} | {r['cls_acc']:.2%} | {r['seq_acc']:.2%} | {r['latency']:.1f}ms |")
    print("")

if __name__ == "__main__":
    run_ablation_study()
