---
type: Methodology
title: Ablation Study & Model Selection
description: The experimental harness used to decide what ships. Four hand-built or intuitive additions were tested against a raw baseline on real images; two were cut on evidence, one model-size choice was confirmed, and one open idea was tested and found wanting.
tags: [methodology, ablation, model-selection, experiment-design]
timestamp: 2026-07-06T00:00:00Z
metrics:
  real_validation_images: 42
  production_config: yolo26-medium_single-stage_no-hmm
---

# Ablation Study & Model Selection

A rule adopted partway through this project: **no feature ships because it
sounds like it should help.** Every addition -- a post-processing corrector,
an extra model stage, a bigger backbone -- gets run through the same
harness (`src/tools/evaluate_pipeline.py`) against the same 42-image
real-world validation set, and is judged on whether it moves real-world
numbers, not synthetic ones and not intuition.

## Why real images, not synthetic mAP

Synthetic validation mAP was consistently much higher than real-world
accuracy throughout this project (see the "Sim-to-Real Gap" framing in
[synthetic_panels](/datasets/synthetic_panels.md)). A feature that improves
synthetic mAP can still hurt real-world accuracy, because the synthetic
distribution is not the real one. Every result below is measured only on
real photographs, held out from training.

## Result 1: the HMM sequence corrector -- cut

**Hypothesis**: a Hidden Markov Model post-processor, using transition
probabilities learned from valid REBT layouts plus OCR/width evidence,
should correct YOLO's classification mistakes by enforcing physically
plausible sequences.

**Result**: it made real-world classification accuracy *worse*, not better.

| Configuration | Classification Accuracy | Latency |
|---|---|---|
| YOLO Baseline | 61.73% | 101.5ms |
| YOLO + HMM | 55.87% | 621.9ms |

Cut from production (`use_hmm: false`). Full detail, including the two
allowlist bugs found while investigating this result, in
[hmm_decoder](/models/hmm_decoder.md).

## Result 2: the two-stage crop classifier -- cut

**Hypothesis**: a dedicated second-stage CNN classifying cropped bounding
boxes, replacing YOLO's own classification head, should resolve visual
confusions (e.g. MCB vs RCD) that a single-pass detector might struggle
with.

**Result**: catastrophic regression -- 3.57% classification accuracy,
because the classifier's weights were trained on the same corrupted,
pre-label-fix data as the original YOLO permutation bug (trained
2026-06-22, ten days before the label-permutation fix landed on
2026-07-02). No reproducible training script for this classifier exists in
the repository, making a clean retrain non-trivial. Given single-stage YOLO
alone already reaches 61.73% with no extra latency, this component was
retired rather than retrained.

## Result 3: augmentation tuning (mixup/scale) -- tested, reverted

**Hypothesis**: `mixup` (blending two full training images) and wide
`scale` jitter might be washing out the small "SI" text that visually
distinguishes RCD from RCD_SI, so disabling mixup and tightening scale
jitter should improve that specific confusion.

**Result**: regressed real-world classification accuracy (61.73% -> 56.38%)
and made the *specific* confusion it targeted worse, not better (23 -> 26
misclassified RCD-as-MCB instances). Reverted. Likely explanation: real
uploaded photos vary widely in camera distance and framing, so the
augmentation this change removed was providing more real-world scale/texture
generalization benefit than it was costing in fine-detail legibility. A
plausible-sounding, mechanistically-reasoned hypothesis, tested and found
false -- kept in this record specifically because a negative result is still
a result.

## Result 4: model backbone size -- Nano vs Medium vs Large

**Hypothesis**: since PanelSafe runs inference on a local GPU rather than
mobile/edge hardware, trading some of Nano's speed advantage for a larger
backbone's classification capacity should be free or near-free.

| Model | Localization Recall | Classification Accuracy | Latency |
|---|---|---|---|
| Nano | 79.34% | 61.73% | 101.5ms |
| **Medium** | **84.18%** | 60.71% | 108.1ms |
| Large | 82.65% | 62.76% | 120.6ms |

**Result**: partially confirmed, with an honest caveat. Medium gave a real,
substantial localization-recall gain over Nano at negligible latency cost.
Large did **not** continue that trend -- its localization recall was
actually lower than Medium's, despite ~5x the parameters, a harder training
run (had to drop batch size 4->2 to fit 12GB VRAM, introducing noisier
batch-normalization statistics), and higher latency. Classification accuracy
differences across all three sizes (60.71-62.76%) are within a few points on
a ~300-330 ground-truth-box sample -- not clearly distinguishable from
run-to-run noise. **Medium was selected for production**: best localization
recall of the three, simplest training (no VRAM workaround needed), lowest
latency of the two larger options.

Why localization recall was the deciding metric rather than classification
accuracy: a missed detection is invisible to a human reviewer in the HITL
step, while a misclassification is still catchable. Recall is the harder
ceiling to raise later; classification errors have a safety net that
localization misses do not.

## Result 5: position-only heuristic for MAINBREAKER -- tested, rejected

**Hypothesis** (brainstormed after noting MAINBREAKER was the weakest class
in the production confusion matrix, at only 30% correct): since a main
breaker is visually near-identical to a 1-module MCB except for its position
(conventionally leftmost/first) and size, a simple positional rule might
identify it more reliably than requiring the vision model to learn a subtle
visual distinction from very few examples.

**Test design**: evaluated directly against ground-truth labels (not model
predictions), removing detection noise entirely, to test the *best possible
case* for the hypothesis. Two rule variants tested on the 41 real images
with labels, using `src/tools/test_mainbreaker_position_heuristic.py`:

| Rule | Precision | Recall |
|---|---|---|
| Leftmost box in whole image | 36.59% | 37.50% |
| Leftmost box on topmost row | 41.46% | 42.50% |

**Result**: rejected, even in the best case. Roughly 60% of the time,
neither variant of "leftmost" corresponds to the actual MAINBREAKER in real
photos -- the assumed physical installation convention (main breaker always
first/leftmost) does not hold consistently across this real-world dataset.
This means the weakness isn't purely a modeling problem solvable by a better
architecture or a cheap heuristic; it may be better addressed by having a
human explicitly flag the main breaker at review time rather than asking
either a vision model or a position rule to infer it. Documented here
because this test had reportedly been done informally before, but never
recorded -- this is the first reproducible version of that result.

## Summary: what shipped, and why

| Component | Decision | Basis |
|---|---|---|
| HMM sequence corrector | Cut | Real accuracy regression (61.73% -> 55.87%) |
| Two-stage crop classifier | Cut | Catastrophic regression (3.57%), trained on corrupted labels |
| Mixup/scale augmentation tuning | Reverted | Regressed the exact confusion it targeted |
| YOLO26-Medium (vs Nano/Large) | Shipped | Best real localization recall, no latency cost |
| MAINBREAKER position heuristic | Not adopted | 36-42% precision/recall even in best case |

The throughline: every one of these was a plausible, well-reasoned idea.
Three were wrong. The harness that tests against real held-out images,
every time, is what made the difference between shipping on intuition and
shipping on evidence.
