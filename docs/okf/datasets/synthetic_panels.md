---
type: Dataset Generator
title: Synthetic Grammar Panels
description: Synthesizes high-fidelity training data modeling Spanish panel layouts.
tags: [dataset, synthetic, python, domain-randomization]
timestamp: 2026-07-04T00:00:00Z
---

# Synthetic Panels Dataset

Because annotated physical photos of domestic breaker boards are scarce, PanelSafe relies on a high-fidelity synthetic image generation engine combined with real-world target annotations.

## PanelFactory Grammar
- **Compliance Model:** Generates valid sequence layouts based on Spain's REBT standards (e.g. enforcing main breakers before differentials, and grouping MCB feeds).
- **Chaos Injection:** Introduces realistic field deviations, such as missing blank modules, misaligned labels, and random sequence ordering.

## Domain Randomization & Noise Injection
- **Structured Randomization:** To bridge the Sim-to-Real gap, the image generator randomizes placement, spacing, background textures, and simulated lighting conditions.
- **Distractor Injection:** Introduces realistic environmental noises like dust, lens smudge, and overlapping wiring in the composite images to force the models to learn robust breaker geometry rather than superficial patterns.

## Compositor & Simulation Features
- **Strategy A (Alpha Feathering):** Outer `3-5px` of cropped seed images are feathered to remove artificial copy-paste rectangular borders.
- **Strategy B (3D Perspective Warp):** Projects coordinates via homography projection matrices to simulate camera tilts.
- **Strategy C (Shadows & Specular Glare):** Superimposes linear shadow gradients (top cutout shadow) and radial glare highlights to simulate basement closet conditions.

## Mixed-Data Training Strategy
- **Synthetic share:** the current `data/dataset` split holds **425 synthetic train / 75 synthetic val** images (`synth_panel_*`) alongside the real ones (counted 2026-09-17).
- **Real-world data:** **121 real breaker board images** (79 train / 42 val). Mostly Spanish, plus scraped images and a few French/German panels. This grew from an original 54-image baseline across two labeling/retraining rounds (as of 2026-07-04), and anchors the model to real aging plastic, glare and lighting.
- **User-in-the-Loop Feedback:** consumer uploads and HITL corrections are *saved* (`data/images/raw_uploads`, `/active-learning/save`, the predictions/corrections store). They join the training set only after a human labels or reviews them and a retrain is run by hand. Nothing retrains automatically (see [yolo26](/models/yolo26.md)).

See [yolo26](/models/yolo26.md) for training configurations.
