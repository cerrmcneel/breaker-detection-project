---
type: Methodology
title: Evaluation Rigor -- Verify, Don't Assume
description: A recurring bug pattern silently corrupted reported accuracy four separate times. This documents the pattern, why it kept happening, and how it was caught each time.
tags: [methodology, evaluation, data-integrity, real-world-validation]
timestamp: 2026-07-06T00:00:00Z
metrics:
  distinct_locations_bug_found: 4
  headline_number_before_correction: 0.8108
  headline_number_after_correction: 0.6173
---

# Evaluation Rigor -- Verify, Don't Assume

PanelSafe's most important engineering lesson this year wasn't a model
architecture choice -- it was discovering that **the same root-cause bug
pattern silently corrupted reported accuracy in four separate places**, one
of which had been inflating the project's own headline number for weeks
before it was caught.

## The pattern: allowlist drift

Every one of the four bugs had the identical shape: code that filtered real
(non-synthetic) images used an **allowlist of known filename prefixes**
(`SPAIN_`, `FRANCE_`) instead of excluding by the one prefix that's actually
stable -- the synthetic generator's own `synth_panel_` prefix. Every time a
new real-image naming convention was introduced (e.g. `scraped_approved_*`
from a later data-collection round), the allowlist silently failed to
recognize those images as real, and evaluation/calibration code quietly
excluded them from the dataset without raising an error.

**Why this is dangerous, not just a bug**: an allowlist failure doesn't
crash. It produces a *smaller, wrong* dataset that still returns a plausible,
confident-looking accuracy number. There is no stack trace pointing at the
problem -- only a number that looks fine until someone asks "does this
number correspond to all the data I think it does?"

## Where it was found

1. **`src/tools/calibrate_hmm_confusion.py`** -- was calibrating the HMM's
   empirical confusion matrix on a narrower, less representative sample of
   real images than actually existed.
2. **`src/tools/evaluate_pipeline.py`** -- this is the one that mattered
   most. It had been silently evaluating on only the original ~16 real
   validation images, excluding all 26 images added in a later labeling
   round. **This bug had been inflating every real-image accuracy number
   reported across the whole project up to that point**: a widely-quoted
   headline of **81.08%** was, on the full and correctly-filtered 42-image
   real validation set, actually **61.73%**.
3. **`src/tools/prepare_button_dataset.py`** -- same bug, in a parked/inactive
   feature (the RCD test-button detector), not live in production.
4. **`src/model/heuristics.py`-adjacent scratch tooling** -- same bug pattern
   found via a full-repo grep audit once the pattern was recognized, after
   the first instance was caught.

The fix in every case was the same: stop allowlisting real-image prefixes
(which requires remembering to update the list every time a new naming
convention appears) and instead **exclude by the one prefix that never
changes** -- the synthetic generator's `synth_panel_` prefix. This is a
general pattern worth naming: *prefer exclusion-by-known-generated-pattern
over inclusion-by-assumed-pattern*, because the set of "things I generate
myself" is closed and stable, while the set of "things a human named" grows
in ways you can't fully anticipate.

## A related, earlier instance: the label permutation bug

Before the allowlist bugs, a structurally similar failure corrupted the
*training labels themselves*, not just evaluation filtering: an external
annotation tool's `classes.txt` (defining class name -> index order) did not
match the model's own `data.yaml` class order. This silently permuted all
6 class indices across every LabelImg-annotated real image -- MCB and RCD
labels were swapped, among others -- for weeks, producing a model that
looked like it was learning (loss went down, synthetic mAP was high) while
actually learning the wrong mapping on real data. Real-world classification
accuracy on the corrupted labels was ~6%, close to random-chance for a
6-class problem.

Both failure classes -- label permutation and allowlist drift -- share the
same underlying lesson: **a model or a metric can look healthy while being
silently wrong, because nothing about the failure mode raises an exception.**
The only defense is treating every reported number as a claim to be checked
against its actual denominator (how many images, which images, why those
images), not accepted at face value because it came out of a script that ran
without errors.

## What this means for reading any number elsewhere in this knowledge base

Numbers cited elsewhere in this OKF bundle (e.g. in
[yolo26](/models/yolo26.md), [hmm_decoder](/models/hmm_decoder.md)) are dated
and were re-verified after these bugs were fixed. Where a document says a
number was corrected from an earlier reported figure, that correction is
intentional and should be read as evidence the evaluation discipline works,
not as an inconsistency to be smoothed over.

See [ablation_study](/methodology/ablation_study.md) for how this same
"verify, don't assume" discipline was applied prospectively -- to decide
*before* shipping whether a new feature actually helps, rather than
discovering after the fact that a metric was wrong.
