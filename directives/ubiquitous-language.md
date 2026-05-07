📚 Ubiquitous Language: PanelSafe ProjectThis document defines the shared vocabulary for the PanelSafe project. All code comments, variable names, and planning discussions must adhere to these definitions.⚡
 1. The 4 Electrical Classes (Target Labels) — Updated 2026-05-04

| Class token | ID | Spanish (REBT) | Description | Visual cues |
|---|---|---|---|---|
| `MCB` | 0 | PIA (Pequeño Interruptor Automático) | Magnetothermic Circuit Breaker. Protects against overloads/short circuits. | Standard 1-module width, toggle switch, marked C10/C16/C25 etc. |
| `MAINBREAKER` | 1 | IGA (Interruptor General Automático) | Main switch for the whole panel. | 2–4 modules wide, master toggle, often larger housing. |
| `OVERSURGE` | 2 | IGA+DPS (Dispositivo de Protección contra Sobretensiones) | Mainbreaker with integrated surge protection. | Combined unit, may have status window (Green=OK / Red=Replace). |
| `OTHER` | 3 | Varios | Timers, contactors, contactors, and any unclassified DIN-rail device. | Variety of sizes; often has a screen, dial, or non-standard form factor. |

> **Note:** RCD (Diferencial) and RCBO are not commonly found in older Spanish residential panels and have been removed from the target taxonomy. If encountered, label as `OTHER`.

🧠 2. YOLO26 & Edge Architecture TermsInference: The process of the model "predicting" or "detecting" breakers in a live image frame.Edge Compute: Local execution on the smartphone browser (via ONNX/TF.js) with zero server dependency.NMS-free: "Non-Maximum Suppression Free." A feature of YOLO26 that allows the model to output final detections without heavy post-processing.Quantization: The process of compressing the model (e.g., from FP32 to INT8) to make it run faster on mobile CPUs.mAP50-95: The primary metric for model accuracy (Mean Average Precision).MuSGD: The advanced optimizer used for YOLO26 training.🏗️
 3. Project-Specific Logistics The Basement Problem: Refers to the technical constraint of zero internet connectivity during field audits. Guided Viewfinder: The UI overlay that tells the user to "Move Closer" or "Hold Still" before an image is captured.Deduplication: The backend process using SHA-256 to ensure the same panel isn't uploaded twice. REBT: Reglamento Electrotécnico para Baja Tensión. The Spanish electrical safety code this project aims to satisfy.