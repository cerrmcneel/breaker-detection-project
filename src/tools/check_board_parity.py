import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def letterbox_image(
    img: np.ndarray,
    target_size: int = 320,
    pad_val: int = 114,
) -> Tuple[np.ndarray, float, float, float]:
    """Resizes and pads image to square target_size x target_size, preserving aspect ratio.

    Returns:
        (blob, scale, pad_x, pad_y) where blob is shape (1, 3, target_size, target_size) float32 RGB [0, 1].
    """
    h0, w0 = img.shape[:2]
    scale = min(target_size / h0, target_size / w0)
    nw = int(round(w0 * scale))
    nh = int(round(h0 * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_size, target_size, 3), pad_val, dtype=np.uint8)

    pad_x = (target_size - nw) / 2.0
    pad_y = (target_size - nh) / 2.0
    dx_int = int(round(pad_x))
    dy_int = int(round(pad_y))

    padded[dy_int : dy_int + nh, dx_int : dx_int + nw] = resized

    # BGR to RGB, normalize [0, 1], HWC to CHW to NCHW
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    return blob, scale, pad_x, pad_y


def run_onnx_inference(
    session,
    img: np.ndarray,
    target_size: int = 320,
    conf_thresh: float = 0.25,
) -> Tuple[List[Dict], float]:
    """Runs ONNX model inference and maps predictions back to original image coordinates."""
    t0 = time.perf_counter()
    blob, scale, pad_x, pad_y = letterbox_image(img, target_size=target_size)

    input_name = session.get_inputs()[0].name
    raw_preds = session.run(None, {input_name: blob})[0][0]  # shape (300, 6)
    t_elapsed = (time.perf_counter() - t0) * 1000.0

    detections = []
    for det in raw_preds:
        score = float(det[4])
        if score < conf_thresh:
            continue
        cls_id = int(det[5])
        # Map letterboxed coordinates back to original image
        x1 = float((det[0] - pad_x) / scale)
        y1 = float((det[1] - pad_y) / scale)
        x2 = float((det[2] - pad_x) / scale)
        y2 = float((det[3] - pad_y) / scale)

        detections.append({
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "score": score,
            "class_id": cls_id,
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections, t_elapsed


def evaluate_parity(
    pt_weights: str,
    onnx_path: str,
    val_dir: str,
    int8_path: Optional[str] = None,
    imgsz: int = 320,
    conf_thresh: float = 0.25,
) -> Dict:
    """Evaluates PyTorch vs ONNX (FP32 and optional INT8) on validation images."""
    import onnxruntime as ort
    from ultralytics import YOLO

    print(f"Loading PyTorch weights: {pt_weights}")
    pt_model = YOLO(pt_weights)

    print(f"Loading ONNX FP32 model: {onnx_path}")
    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    int8_session = None
    if int8_path and os.path.exists(int8_path):
        print(f"Loading ONNX INT8 model: {int8_path}")
        int8_session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    val_images = sorted(
        p for p in Path(val_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    total_imgs = len(val_images)
    print(f"Found {total_imgs} validation images in {val_dir}\n")

    ious_fp32 = []
    score_diffs_fp32 = []
    ious_int8 = []
    score_diffs_int8 = []

    pt_times = []
    fp32_times = []
    int8_times = []

    worst_fp32_case = None
    worst_int8_case = None

    for idx, img_p in enumerate(val_images, 1):
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        # 1. PyTorch inference (rect=False to match fixed 320x320 export)
        t_pt0 = time.perf_counter()
        pt_res = pt_model.predict(
            source=img,
            imgsz=imgsz,
            rect=False,
            device="cpu",
            conf=conf_thresh,
            verbose=False,
        )[0]
        t_pt = (time.perf_counter() - t_pt0) * 1000.0
        pt_times.append(t_pt)

        pt_boxes = pt_res.boxes.xyxy.cpu().numpy()
        pt_scores = pt_res.boxes.conf.cpu().numpy()

        # 2. ONNX FP32 inference
        fp32_dets, t_fp32 = run_onnx_inference(
            onnx_session, img, target_size=imgsz, conf_thresh=conf_thresh
        )
        fp32_times.append(t_fp32)

        # 3. ONNX INT8 inference (if present)
        int8_dets, t_int8 = ([], 0.0)
        if int8_session is not None:
            int8_dets, t_int8 = run_onnx_inference(
                int8_session, img, target_size=imgsz, conf_thresh=conf_thresh
            )
            int8_times.append(t_int8)

        # Compare Top Box: PyTorch vs ONNX FP32
        has_pt = len(pt_boxes) > 0
        has_fp32 = len(fp32_dets) > 0

        if has_pt and has_fp32:
            iou_f = box_iou(pt_boxes[0], fp32_dets[0]["box"])
            diff_f = abs(float(pt_scores[0]) - float(fp32_dets[0]["score"]))
            ious_fp32.append(iou_f)
            score_diffs_fp32.append(diff_f)

            if worst_fp32_case is None or iou_f < worst_fp32_case["iou"]:
                worst_fp32_case = {
                    "image": img_p.name,
                    "iou": iou_f,
                    "pt_box": pt_boxes[0].tolist(),
                    "onnx_box": fp32_dets[0]["box"].tolist(),
                    "pt_score": float(pt_scores[0]),
                    "onnx_score": float(fp32_dets[0]["score"]),
                }
        elif not has_pt and not has_fp32:
            ious_fp32.append(1.0)
            score_diffs_fp32.append(0.0)
        else:
            ious_fp32.append(0.0)
            score_diffs_fp32.append(1.0)
            if worst_fp32_case is None or 0.0 < worst_fp32_case["iou"]:
                worst_fp32_case = {
                    "image": img_p.name,
                    "iou": 0.0,
                    "pt_count": len(pt_boxes),
                    "onnx_count": len(fp32_dets),
                }

        # Compare Top Box: PyTorch vs ONNX INT8
        if int8_session is not None:
            has_int8 = len(int8_dets) > 0
            if has_pt and has_int8:
                iou_8 = box_iou(pt_boxes[0], int8_dets[0]["box"])
                diff_8 = abs(float(pt_scores[0]) - float(int8_dets[0]["score"]))
                ious_int8.append(iou_8)
                score_diffs_int8.append(diff_8)

                if worst_int8_case is None or iou_8 < worst_int8_case["iou"]:
                    worst_int8_case = {
                        "image": img_p.name,
                        "iou": iou_8,
                        "pt_box": pt_boxes[0].tolist(),
                        "int8_box": int8_dets[0]["box"].tolist(),
                        "pt_score": float(pt_scores[0]),
                        "int8_score": float(int8_dets[0]["score"]),
                    }
            elif not has_pt and not has_int8:
                ious_int8.append(1.0)
                score_diffs_int8.append(0.0)
            else:
                ious_int8.append(0.0)
                score_diffs_int8.append(1.0)

        if idx % 25 == 0 or idx == total_imgs:
            print(f"Evaluated {idx}/{total_imgs} images...")

    results = {
        "total_images": total_imgs,
        "fp32_iou_mean": float(np.mean(ious_fp32)),
        "fp32_iou_median": float(np.median(ious_fp32)),
        "fp32_iou_min": float(np.min(ious_fp32)),
        "fp32_score_diff_mean": float(np.mean(score_diffs_fp32)),
        "fp32_score_diff_max": float(np.max(score_diffs_fp32)),
        "worst_fp32_case": worst_fp32_case,
        "pt_time_mean": float(np.mean(pt_times)),
        "pt_time_median": float(np.median(pt_times)),
        "fp32_time_mean": float(np.mean(fp32_times)),
        "fp32_time_median": float(np.median(fp32_times)),
        "fp32_size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
    }

    if int8_session is not None:
        results.update({
            "int8_iou_mean": float(np.mean(ious_int8)),
            "int8_iou_median": float(np.median(ious_int8)),
            "int8_iou_min": float(np.min(ious_int8)),
            "int8_score_diff_mean": float(np.mean(score_diffs_int8)),
            "int8_score_diff_max": float(np.max(score_diffs_int8)),
            "worst_int8_case": worst_int8_case,
            "int8_time_mean": float(np.mean(int8_times)),
            "int8_time_median": float(np.median(int8_times)),
            "int8_size_mb": os.path.getsize(int8_path) / (1024 * 1024),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Check PyTorch vs ONNX parity on board dataset.")
    parser.add_argument(
        "--pt-weights",
        type=str,
        default=r"C:\Users\PC GAMING\breaker-detection-project\runs\detect\model\runs\board-viewfinder-tracked\weights\best.pt",
        help="Path to PyTorch .pt weights.",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        default=r"C:\Users\PC GAMING\breaker-detection-project\runs\detect\model\runs\board-viewfinder-tracked\weights\best.onnx",
        help="Path to ONNX FP32 model.",
    )
    parser.add_argument(
        "--int8-model",
        type=str,
        default=r"C:\Users\PC GAMING\breaker-detection-project\runs\detect\model\runs\board-viewfinder-tracked\weights\best_int8.onnx",
        help="Path to ONNX INT8 model (optional).",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=r"C:\ironhack\labs\breaker-detection-project\data\board_dataset\val\images",
        help="Directory with validation images.",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Image size (default: 320).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    args = parser.parse_args()

    results = evaluate_parity(
        pt_weights=args.pt_weights,
        onnx_path=args.onnx_model,
        val_dir=args.val_dir,
        int8_path=args.int8_model,
        imgsz=args.imgsz,
        conf_thresh=args.conf,
    )

    print("\n" + "=" * 60)
    print("PARITY EVALUATION RESULTS (114 Validation Images)")
    print("=" * 60)
    print(f"FP32 Model File Size:       {results['fp32_size_mb']:.2f} MB")
    print(f"FP32 Top Box Mean IoU:      {results['fp32_iou_mean'] * 100:.3f}%")
    print(f"FP32 Top Box Median IoU:    {results['fp32_iou_median'] * 100:.3f}%")
    print(f"FP32 Top Box Worst IoU:     {results['fp32_iou_min'] * 100:.3f}%")
    print(f"FP32 Mean Conf Score Diff:  {results['fp32_score_diff_mean']:.5f}")
    print(f"FP32 Max Conf Score Diff:   {results['fp32_score_diff_max']:.5f}")
    print(f"PyTorch CPU Inference Time: {results['pt_time_median']:.1f} ms median ({results['pt_time_mean']:.1f} ms mean)")
    print(f"ONNX FP32 CPU Latency:      {results['fp32_time_median']:.1f} ms median ({results['fp32_time_mean']:.1f} ms mean)")

    if "int8_size_mb" in results:
        print("-" * 60)
        print(f"INT8 Model File Size:       {results['int8_size_mb']:.2f} MB ({results['int8_size_mb'] / results['fp32_size_mb'] * 100:.1f}% of FP32)")
        print(f"INT8 Top Box Mean IoU:      {results['int8_iou_mean'] * 100:.3f}%")
        print(f"INT8 Top Box Median IoU:    {results['int8_iou_median'] * 100:.3f}%")
        print(f"INT8 Top Box Worst IoU:     {results['int8_iou_min'] * 100:.3f}%")
        print(f"INT8 Mean Conf Score Diff:  {results['int8_score_diff_mean']:.5f}")
        print(f"INT8 Max Conf Score Diff:   {results['int8_score_diff_max']:.5f}")
        print(f"ONNX INT8 CPU Latency:      {results['int8_time_median']:.1f} ms median ({results['int8_time_mean']:.1f} ms mean)")
    print("=" * 60)


if __name__ == "__main__":
    main()
