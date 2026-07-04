---
type: Hidden Markov Model
title: HMM Viterbi Decoder
description: Sequence correction model representing physical constraints and REBT
  conventions.
tags:
- model
- post-processing
- viterbi
timestamp: '2026-07-04T00:00:00Z'
status: disabled-in-production
metrics:
  laplace_smoothing_alpha: 0.01
  states_count: 6
  unit_tests_pass_rate: 1.0
  simulations_trained: 10000
---



# HMM Viterbi Decoder

> **STATUS (2026-07-04): DISABLED in production** (`pipeline_config.json: "use_hmm": false`).
> Rigorous ablation on the full real validation set showed HMM correction *reduces*
> classification accuracy rather than improving it. See "Why It's Disabled" below before
> assuming this component is active.

The HMM Corrector replaces brittle rule-based heuristics with probabilistic sequence correction.

## Math & Sequence Decoding
To avoid numerical underflow during path probability updates, calculations are computed in log-space:

$$\log V_t(j) = \max_{i} \left[ \log V_{t-1}(i) + \log A_{ij} \right] + \log B_j(o_t)$$

- **Transition Matrix ($A_{ij}$):** Modeled on transition counts from `synthetic_panels` panels generator compliant with REBT rules, smoothed via Laplace method:
  $$P(j \mid i) = \frac{\operatorname{Count}(i \to j) + \alpha}{\left( \sum_{k} \operatorname{Count}(i \to k) \right) + |S| \cdot \alpha}$$
- **Emission Function ($B_j(o_t)$):** Multi-modal probability combining:
  1. YOLO visual confidence
  2. Physical width priors (e.g. standard RCD = 2 modules, MCB = 1 module)
  3. Regex-cleaned OCR text (e.g. finding "SI" increases $P(\text{RCD\_SI})$)

## Bidirectional Sequence Decoding
Since wiring can run Left-to-Right or Right-to-Left, we decode both directions and choose the sequence path yielding the highest overall joint probability.

## Why It's Disabled (2026-07-04)

On the full, correctly-filtered real validation set (42 real images across
`data/dataset/val`), HMM correction **reduces** classification accuracy relative to the
raw YOLO baseline:

| Configuration | Classification Accuracy | Latency |
|---|---|---|
| YOLO Baseline | 61.73% | 92.3ms |
| YOLO + HMM | 55.87% | 602.4ms |

This holds even after recalibrating the HMM's empirical confusion matrix
(`calibrate_hmm_confusion.py`) against the current model — recalibration alone did not
fix the regression, ruling out a stale confusion matrix as the sole cause.

**Two real implementation bugs were found and fixed during this investigation** (both
were silently excluding newer real images via an outdated `SPAIN_`/`FRANCE_` filename
allowlist, instead of excluding by the `synth_panel_` synthetic prefix):
- `src/tools/calibrate_hmm_confusion.py` — was calibrating on a narrower, less
  representative real-image sample than actually available.
- `src/tools/evaluate_pipeline.py` — was silently evaluating on only the original ~16
  real val images, excluding all 26 images added in the most recent data round. This bug
  had been inflating every real-image accuracy number reported before it was caught.

**Remaining unresolved hypothesis (untested):** `hmm_corrector.py`'s `width_priors` are a
static, hand-tuned dictionary, explicitly noted in-code as calibrated to a specific
heuristic's computed box-width behavior — never recalibrated against the current model's
actual outputs. This is structurally the same category of staleness as the confusion
matrix, and is the most likely remaining explanation if this component is investigated
further.

**Current recommendation:** keep `use_hmm: false`. The YOLO baseline alone is both
faster and more accurate on real images.

See [rebt_rules](/standards/rebt_rules.md) for safety standards and [synthetic_panels](/datasets/synthetic_panels.md) for the grammar configurations.
