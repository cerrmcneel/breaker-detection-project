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

## Mobile Layout & Phone Access (T8)

The viewfinder interface has been specifically optimized for mobile devices (tested against Google Pixel 8 and modern mobile viewports):
- **Dynamic Fullscreen Viewport:** Built with CSS `100dvh` (with standard `100vh` fallback) and `viewport-fit=cover` to eliminate scroll bouncing and accommodate mobile dynamic address bars.
- **Orientation & Resize Handling:** Reacts to `orientationchange` and window `resize` events by executing `recomputeOverlayMapping()`. In both portrait and landscape modes, the overlay canvas stays aligned with the video feed.
- **Touch-Sized Controls:** Action buttons have minimum touch target dimensions of $\ge 48\text{px} \times 48\text{px}$ (meeting WCAG AAA / Android Material Design standards) with `touch-action: manipulation` to eliminate 300 ms tap delays.
- **Safe Area Insets:** Margins dynamically respect `env(safe-area-inset-top)` and `env(safe-area-inset-bottom)` so controls and telemetry are never obscured by camera punch-holes or gesture navigation bars.
- **Zero Horizontal Scroll:** Bounded within viewport width (`width: 100vw; overflow-x: hidden`).

---

## Tailscale Serve Setup (For Owner to Run)

> [!WARNING]
> **DO NOT USE `tailscale funnel`.** Funnel exposes your local machine to the public internet. `tailscale serve` keeps the connection strictly private within your encrypted Tailscale tailnet.

Browsers mandate a **Secure Context** (HTTPS or `localhost`) to grant access to the camera (`navigator.mediaDevices.getUserMedia`). Because this PC (`datainmind`, Tailscale IP `100.102.124.29`) and the owner's phone (`pixel-8`) are on the same Tailnet, Tailscale Serve provides automatic HTTPS with trusted certificates inside the tailnet.

### Syntax Verified Against Installed Tailscale:
- **Tailscale Version:** `1.102.2` (commit `6cac918179d4d673bfebe2fc74f81183ddd73fea`, Go `go1.26.5`)
- **Current Tailnet Status:** Node root `/` on port 443 already proxies `http://127.0.0.1:8000`. Therefore, port `8443` is selected to avoid collision.

### 1. Enable HTTPS Proxy on Port 8443:
Run this on the PC from powershell:
```powershell
tailscale serve --bg --https=8443 8095
```

### 2. Verify Status:
```powershell
tailscale serve status
```
Output should show:
```text
https://datainmind.taila2c133.ts.net:8443 (tailnet only)
|-- / proxy http://127.0.0.1:8095
```

### 3. Disable Proxy After Testing:
```powershell
tailscale serve --https=8443 off
```

### COOP / COEP & Multi-Threading Over Proxy:
`viewfinder_live/serve.py` injects `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Tailscale Serve reverse-proxies these HTTP headers unchanged. 
Over HTTPS, `window.crossOriginIsolated` evaluates to `true`, activating `SharedArrayBuffer` and enabling 4-threaded WebAssembly execution on the phone. If opened over plain HTTP, the HUD indicates `Isolated: false (No SharedArrayBuffer)` and gracefully falls back to single-threaded mode.

---

## Owner's Phone Testing Checklist (Pixel 8)

Follow these steps to test the viewfinder on the Pixel:

1. **Start Local Server:**
   ```powershell
   python viewfinder_live/serve.py --port 8095
   ```
2. **Enable Tailscale Serve:**
   ```powershell
   tailscale serve --bg --https=8443 8095
   ```
3. **Open on Pixel 8:**
   - Ensure the Tailscale app is connected on the Pixel.
   - In Chrome on the phone, open:
     `https://datainmind.taila2c133.ts.net:8443/`
4. **Grant Camera Access:**
   - Tap "Allow" when the browser asks for permission to use the camera.
5. **Frame an Electrical Panel:**
   - Point the camera at a physical breaker panel or a printed photo of a panel.
6. **Check HUD Telemetry:**
   - Verify `Isolated:` shows `true (COOP/COEP)`.
   - Verify `WASM Threads:` shows `4 threads` (or matching CPU cores).
   - Check `Inference:` latency (expected: 25–40 ms).
   - Check `Pipeline FPS:` (expected: 30+ FPS).
   - Confirm cyan union bounding box and green DIN rail row boxes appear around the panel.
7. **Rotate Phone:**
   - Rotate between portrait and landscape mode.
   - Confirm bounding boxes remain aligned with the physical panel and no horizontal scrolling occurs.
8. **Test Still Capture:**
   - Tap "Capture Frame". The phone will download a full-resolution JPEG still without uploading anything.
9. **Test Offline "Basement" Mode (T9):**
   - Confirm the HUD displays `Offline: READY (CACHED)` indicating all 9 assets (`index.html`, vendored runtime, `.wasm`, model, and test feeds) are precached in Cache Storage (`panelsafe-viewfinder-v1.0.0`).
   - Turn on **Airplane Mode** (or disable Wi-Fi / cellular) on the Pixel 8.
   - Refresh / reload the page in Chrome while completely disconnected.
   - Verify the page loads immediately with zero connection.
   - Verify HUD indicates `Offline: OFFLINE ACTIVE` and `Isolated: true (COOP/COEP)`.
   - Point the camera at a breaker panel: verify real-time inference, EMA smoothing, and box tracking continue smoothly offline at ~30+ FPS!
   - Turn Airplane Mode back off.
10. **Clean Up:**
   - On the PC, turn off the proxy:
     ```powershell
     tailscale serve --https=8443 off
     ```

