---
type: Object Detector
title: YOLO26 Object Detector
description: Bounding box locator and visual classifier for DIN-rail electrical components.
tags: [model, yolo26, yolo26-nano, object-detection, onnx]
timestamp: 2026-07-04T00:00:00Z
---

# YOLO26 Object Detector

PanelSafe uses a unified or two-stage YOLO model (specifically the state-of-the-art **YOLO26-Nano**) optimized for real-time edge execution and mobile deployment. Older models like YOLOv10 and YOLOv11 are treated as legacy baseline comparative models evaluated during early research phases.

## Classes & Labeling Taxonomy
The model is trained on a taxonomy matching Spanish domestic electrical panels (configured in `data.yaml` with `nc` classes):
- `0: MCB` (PIA) - Magnetothermic Circuit Breakers (e.g., 15A Single Pole, 20A Double Pole)
- `1: RCD` (Diferencial) - Standard Residual Current Devices (e.g., Type AC RCD)
- `2: RCD_SI` (Diferencial Superinmunizado) - DC-sensitive/Superimmunized RCDs (e.g., Type A RCD)
- `3: MAINBREAKER` (IGA) - Main Circuit Breaker
- `4: OVERSURGE` (IGA+DPS) - Combined Main Switch and Surge Protection Device
- `5: OTHER` - Timers, contactors, pilot lights, etc.

## Model Architecture & Export
- **Decoupled Head:** The YOLO26 architecture separates bounding box regression (localization) from object classification to prevent task interference. This satisfies multi-class visual identification requirements in a single model pass.
- **NMS-Free Detection:** The model utilizes NMS-free training and prediction (natively supported in modern YOLO lines like YOLO26/v10/v11), removing the post-processing Non-Maximum Suppression bottleneck for ultra-low latency.
- **ONNX Export:** To run efficiently on edge hardware (mobile CPUs/GPUs) without heavy PyTorch dependencies, the trained model is exported to the Open Neural Network Exchange (ONNX) format, separating the training stack from mobile inference.

## Transfer Learning & Accuracy Optimization Strategies
To bridge the Sim-to-Real gap, the project evaluates:
- **Strategy E (Two-Stage Pipeline):** YOLO as a single-class localization model (`Breaker`) followed by an EfficientNet crop classifier.
- **Strategy G (Tiny Test-Button Detector):** A micro-model that runs inside RCD crops to verify the presence of a physical "Test" button.
- **Strategy K (Differential Learning Rates):** Frozen backbones ($\eta = 10^{-6}$) and high-rate classification heads ($\eta = 10^{-3}$) to prevent catastrophic forgetting on real photos.
- **Strategy L (Semi-Supervised Pseudo-Labeling):** Self-labeling high-confidence uploads to expand the training set.

## Continuous Learning / Automated Retraining — Deliberately Deferred

Strategy L (self-labeling high-confidence uploads) and the "/active-learning/save"
endpoint already collect the raw material for a continuous-retraining loop, but
automating the retraining trigger itself is a deliberate non-goal for now, not
an oversight. At ~121 real images with single-digit counts for some classes
(RCD_SI, OVERSURGE), an automated loop would mostly automate noisy, unstable
retraining swings -- the kind of run-to-run variance this project already
measured directly when comparing Nano/Medium/Large (see
[ablation_study](/methodology/ablation_study.md)) -- without a human catching
problems before they compound. The evaluation discipline documented in
[evaluation_rigor](/methodology/evaluation_rigor.md) is what has caught every
real bug this project has found; full automation would remove exactly that
checkpoint.

**Revisit threshold**: reconsider automated continuous learning once every
class has at least **100 real examples** (raised from an initial informal
estimate of 30, on the view that a decision this consequential should demand
a higher bar of evidence before trusting an automated loop with it).

## Roadmap: Scaling to the Full Field Taxonomy (Not Yet Implemented)

The current 6-class flat taxonomy covers the essentials of a domestic CGMP. A production tool must recognize the full suite of modular devices electricians use in the field (contactors, timers, *telerruptores*, modular sockets, RCBOs, phase indicators, varied surge arresters, etc.). The scalable path is **not** one detector per class — that is N× compute/VRAM, loses the inter-class contrast a unified head learns, and forces cross-model NMS arbitration. Instead:

- **Hierarchical two-stage:** a class-agnostic module localizer + classifier heads operating on crops. Adding a device type becomes "a new class on the Stage-2 classifier + some crops," with **no new detector**. (Strategy E and the crop classifier are the first scaffolding.)
- **Multi-attribute heads, not an exploding flat class list:** predict composable attributes — `{function, pole_count, has_test_button, dc_sensitive, rating}` — and compose them into device types. The RCD test-button detector (Strategy G) is the **first attribute head**; e.g. an RCBO = `protection + has_test_button + 1-module`.
- **REBT-grounded ontology:** anchor each new device to [ubiquitous-language] and [rebt_rules](/standards/rebt_rules.md) (definition, REBT role, schematic symbol) **before** it enters the model taxonomy, keeping annotation, the schematic generator, and grading coherent as labels grow.

See the [HMM Decoder](/models/hmm_decoder.md) for post-processing error correction details
— **currently disabled in production** (as of 2026-07-04, evidence-based: it measurably
reduced real-world classification accuracy on the full validation set).
