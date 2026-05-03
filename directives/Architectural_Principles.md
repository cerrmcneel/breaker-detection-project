ARCHITECTURAL_PRINCIPLES.md
🏗️ PanelSafe: Architectural Principles
This project prioritizes Software Fundamentals to manage complexity and prevent software entropy. We follow the philosophy outlined by John Ousterhout (A Philosophy of Software Design) and Matt Pocock.

🏔️ 1. Deep vs. Shallow Modules
The core goal is to pull complexity downwards.

Shallow Modules: These have complex interfaces relative to their small functionality. They force the developer to understand too many details to get a result.

Deep Modules: These are like icebergs. They provide a tiny, simple interface (the tip) while hiding massive complexity beneath the surface (the implementation).

📸 2. Implementation: The DetectionModule
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