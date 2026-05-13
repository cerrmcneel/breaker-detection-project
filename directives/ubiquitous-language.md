📚 Ubiquitous Language: PanelSafe Project

This document defines the shared vocabulary for the PanelSafe project. All code comments, variable names, and planning discussions must adhere to these definitions.

---

⚡ 1. The 6 Electrical Classes (Target Labels) — Updated 2026-05-10

> **Note:** Full 6-class taxonomy confirmed 2026-05-10. All classes are active detection targets. Class IDs match `data.yaml` and `Compositor.CLASS_MAP`.

| Class token | ID | Spanish (REBT) | Description | Visual cues |
|---|---|---|---|---|
| `MCB` | 0 | PIA (Pequeño Interruptor Automático) | Magnetothermic Circuit Breaker. Protects against overloads/short circuits. | Standard 1-module width, toggle switch, marked C10/C16/C25 etc. |
| `RCD` | 1 | Diferencial AC-type | Residual Current Device (standard). 30mA, 2 modules. Protects against earth leakage. | 2 modules wide, test button visible, often white/gray housing. |
| `RCD_SI` | 2 | Diferencial Superinmunizado (Type A/F) | Superimmunized RCD. DC-sensitive, 30mA, growing in modern installs. | Similar to RCD but often marked "SI" or "Type A", may have additional indicators. |
| `MAINBREAKER` | 3 | IGA (Interruptor General Automático) | Main switch for the whole panel. | 2–4 modules wide, master toggle, often larger housing. |
| `OVERSURGE` | 4 | IGA+DPS (Dispositivo de Protección contra Sobretensiones) | Mainbreaker with integrated surge protection. | Combined unit, may have status window (Green=OK / Red=Replace). |
| `OTHER` | 5 | Varios | Timers, contactors, 300mA fire-protection RCDs, and any unclassified DIN-rail device. | Variety of sizes; often has a screen, dial, or non-standard form factor. |

---

🧠 2. YOLO26 & Edge Architecture Terms

- **Inference**: The process of the model "predicting" or "detecting" breakers in a live image frame.
- **Edge Compute**: Local execution on the smartphone browser (via ONNX/TF.js) with zero server dependency.
- **NMS-free**: "Non-Maximum Suppression Free." A feature of YOLO26 that allows the model to output final detections without heavy post-processing.
- **Quantization**: The process of compressing the model (e.g., from FP32 to INT8) to make it run faster on mobile CPUs.
- **mAP50-95**: The primary metric for model accuracy (Mean Average Precision).
- **MuSGD**: The advanced optimizer used for YOLO26 training.

---

🏗️ 3. Project-Specific Logistics

- **The Basement Problem**: Refers to the technical constraint of zero internet connectivity during field audits.
- **Guided Viewfinder**: The UI overlay that tells the user to "Move Closer" or "Hold Still" before an image is captured.
- **Deduplication**: The backend process using SHA-256 to ensure the same panel isn't uploaded twice.
- **REBT**: Reglamento Electrotécnico para Baja Tensión. The Spanish electrical safety code this project aims to satisfy.
