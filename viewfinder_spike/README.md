# PanelSafe Viewfinder ONNX Spike (T5)

This directory contains the self-contained static HTML + JS viewfinder spike for testing in-browser inference of the 1-class `BOARD` detector using ONNX Runtime Web (WASM).

## Architecture & Features
- **Model:** YOLO26-Nano trained for 1-class `BOARD` detection, exported at fixed input shape `[1, 3, 320, 320]`.
- **NMS-Free Output:** The model outputs `output0` of shape `[1, 300, 6]`, where each detection contains `[x1, y1, x2, y2, score, class_id]`.
- **Letterboxing:** Preserves aspect ratio with letterbox padding (`114` gray) to 320x320, mapping bounding boxes back to unpadded coordinates.
- **DIN Rail Row Detection & Union:** On multi-row breaker panels, the model detects individual rows. The spike displays all rows above the confidence threshold (green) and computes their union bounding box (amber dashed).
- **Fill Fraction:** Reports the ratio of the board union area to total frame area. *(Note: The hypothesis that fill fraction predicts OCR legibility was falsified with a -0.07 correlation; the value is displayed for telemetry only).*

## How to Export Models
Using the worktree virtual environment (`.venv_t5`):
```powershell
# Export FP32 ONNX
.\.venv_t5\Scripts\python.exe -m src.model.export `
  --weights "C:\Users\PC GAMING\breaker-detection-project\runs\detect\model\runs\board-viewfinder-tracked\weights\best.pt" `
  --imgsz 320 --format onnx --no-dynamic

# Copy to viewfinder_spike
Copy-Item "C:\Users\PC GAMING\breaker-detection-project\runs\detect\model\runs\board-viewfinder-tracked\weights\best.onnx" viewfinder_spike\board_viewfinder.onnx
```

## How to Run Local Server
```powershell
.\.venv_t5\Scripts\python.exe viewfinder_spike/serve.py --port 8092
```
Open `http://127.0.0.1:8092/` in any modern browser (Chrome, Edge, Safari, Firefox).
The server sets COOP (`same-origin`) and COEP (`require-corp`) headers to enable multi-threaded WASM (`SharedArrayBuffer`).
