# PanelSafe Live Camera Viewfinder (T7)

This directory contains the standalone live camera viewfinder demo page running the 1-class `BOARD` detector directly inside the browser using client-side WebAssembly via vendored `onnxruntime-web` (pinned to version 1.20.1).

## Architecture & Features
- **Client-Side Model:** YOLO26-Nano FP32 (`board_viewfinder.onnx`, 9.68 MB), trained on breaker board boundaries at 320x320 input shape. NMS-free output: `[1, 300, 6]`.
- **Vendored Dependencies:** Fully offline and local. Zero CDN or external network dependencies.
  - `vendor/ort.min.js`: 446,284 bytes (v1.20.1)
  - `vendor/ort-wasm-simd-threaded.wasm`: 11,246,032 bytes (v1.20.1)
  - `vendor/ort-wasm-simd-threaded.mjs`: 24,618 bytes (v1.20.1)
- **Live Camera Pipeline:** Connects via `navigator.mediaDevices.getUserMedia` (`facingMode: "environment"`, 1280x720 ideal resolution).
- **Video Fallback Mode (`?src=video`):** If no physical webcam is attached or permission is denied, seamlessly falls back to `test_board_feed.mp4` (stitched from real validation panels in `data/board_dataset/val`).
- **Overlay & Coordinate Mapping:** Accurately maps 320x320 letterbox coordinates back to native video coordinates, and displays them on a synchronized overlay canvas.
- **Stability & Smoothing:**
  - Smoothed via Exponential Moving Average (EMA, $\alpha = 0.5$) across frames.
  - Box is hidden after 3 consecutive missed detections to prevent flickering.
- **DIN Rail Rows & Union:** Faintly draws individual DIN rail rows (`rgba(34, 197, 94, 0.45)`), and strongly highlights the board union bounding box with high-contrast cyan outlines and corner brackets.
- **HUD Telemetry:** Real-time latency, pipeline FPS, confidence %, row count, fill fraction %, WASM thread count, and `crossOriginIsolated` status.
- **Capture Button:** Extracts full-resolution still from the video stream to local download. Does not upload anywhere.

---

## How to Run Locally

### 1. Start the HTTP Server
Run `serve.py` using Python (serves on port 8095 by default with COOP and COEP cross-origin isolation headers):
```powershell
python viewfinder_live/serve.py --port 8095
```

### 2. Open in Browser
- Open `http://localhost:8095/` in Chrome, Edge, Safari, or Firefox.
- `localhost` is treated as a secure context by modern browsers, allowing camera access without HTTPS.
- To run with test video footage instead of a webcam:
  `http://localhost:8095/?src=video`

---

## Automated Verification Benchmark
To measure in-browser inference performance in headless Chrome:
```powershell
python viewfinder_live/serve.py --port 8095
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --user-data-dir="$env:TEMP\chrome_bench" "http://localhost:8095/?benchmark=true"
```
The browser runs 40 inference frames against the test video feed and posts the telemetry payload to `viewfinder_live/benchmark_results.json`.
