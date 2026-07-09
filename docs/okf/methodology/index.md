---
type: Index
title: Methodology
description: Sub-index documenting the statistical and experimental reasoning behind PanelSafe's model decisions -- not just what was built, but how each decision was verified.
tags: [methodology, evaluation, statistics, rigor]
timestamp: 2026-07-06T00:00:00Z
---

# Methodology Index

The rest of this knowledge base documents *what* PanelSafe is: the models, the
datasets, the tools. This section documents *how decisions about those things
were made and verified* -- the evaluation discipline that repeatedly caught
silent bugs and falsified plausible-sounding hypotheses before they shipped.

## Concepts

- **Evaluation Rigor**: [evaluation_rigor](/methodology/evaluation_rigor.md)
  - The recurring "verify, don't assume" pattern: four separate instances of
    the same root-cause bug class silently corrupting reported metrics, and
    how each was caught.
- **Ablation Study & Model Selection**: [ablation_study](/methodology/ablation_study.md)
  - The experimental harness used to decide what ships: which hand-built
    features were cut on evidence, and how the production model size (Nano
    vs Medium vs Large) was chosen.

See [tabular_risk_model](/models/tabular_risk_model.md) for a classical-ML
side-analysis that applies the same evidence-over-intuition discipline to a
falsified hypothesis about panel age prediction.
