ARCHITECTURAL_PRINCIPLES.md
🏗️ PanelSafe: Architectural Principles
This project prioritizes Software Fundamentals to manage complexity and prevent software entropy. We follow the philosophy outlined by John Ousterhout (A Philosophy of Software Design) and Matt Pocock.

🏔️ 1. Deep vs. Shallow Modules
The core goal is to pull complexity downwards.

Shallow Modules: These have complex interfaces relative to their small functionality. They force the developer to understand too many details to get a result.

Deep Modules: These are like icebergs. They provide a tiny, simple interface (the tip) while hiding massive complexity beneath the surface (the implementation).

📸 2. Example Design (never built): A Client-Side DetectionModule

> Status (checked 2026-09-17): this section describes a planned in-browser design from May 2026. No `startDetection()` / `stopDetection()` module, WebWorker or in-browser YOLO exists. Inference runs server-side on the GPU worker, and the only frontend JS module is `app/frontend/js/modules/BootstrapModule.js`. The deep-module principle in §1 still applies; read §2 as an illustration of it, not a description of the code.
Instead of 10 fragmented scripts for camera handling and AI processing, all logic is encapsulated within a single Deep Module.

The Public Interface (The Tip of the Iceberg)
The rest of the PWA only sees these two functions:

startDetection(): Initializes the camera, loads the YOLO26 model, and starts the live inference loop.

stopDetection(): Tears down the camera stream, clears the WebWorker, and stops the UI overlay.

The Private Implementation (The Submerged Body)
The AI manages the following "Invisible" logic inside the module:

YOLO26 Inference: Handling the NMS-free end-to-end prediction logic.

WebWorker Orchestration: Moving the heavy math off the main UI thread to prevent lag.

Canvas Rendering: Drawing bounding boxes and class labels (MCB, RCD, etc.) at 30+ FPS.

Image Preprocessing: Scaling and normalizing image data for the model.

📜 3. Why This Matters
Low Cognitive Load: As the developer, you only need to know how to "start" and "stop" the feature. You don't need to juggle the 20 variables required for WebGL textures or model weights.

Ease of Testing: We can test the entire "Detection" feature by simply asserting that startDetection() produces a valid stream and stopDetection() releases it.

TDD Compatibility: By having a stable interface, we can write our tests first and let the AI refactor the messy implementation underneath without breaking the app.

☸️ 4. Hybrid MLOps Strategy
We don't just build models; we build the pipes that run them.

Infrastructure as Code (IaC): The deployment is as important as the detection. Deployment is captured in files (docker-compose.yml, the Modal deployment script, the launcher scripts). Kubernetes (K3s) was used in May 2026; the GPU worker now runs as a Windows Scheduled Task, with Modal as failover.

Hardware-Aware Inference: By leveraging a local RTX 3060 (on the Windows workstation, reached over Tailscale), we prioritize performance over the cost-constraints of cloud free-tiers.

Secure Edge Access: Using Cloudflare Tunnels ensures that our high-performance local "brain" is safely accessible to field devices globally.