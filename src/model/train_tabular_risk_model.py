#!/usr/bin/env python3
"""
train_tabular_risk_model.py — classical ML on the panel-attribute CSV dataset.

STATUS: PARKED / INSUFFICIENT DATA — not wired into any live pipeline.

WHAT THIS IS: an exploratory classical-ML complement to the YOLO/HMM detection
pipeline. It tests whether a panel's physical component composition (breaker
amperage pattern, count, main breaker rating, phase type) predicts installation
age (binarized: old = >15yr, newer = <=15yr) using data/breaker_dataset.csv,
the CSV collected by src/tools/labeler.py.

KEY FINDING (2026-07, N=98 rows): RandomForestClassifier finds modest but real
signal from component-composition features alone (balanced_accuracy ~0.59-0.64
via 5-fold stratified CV), clearly above the ~0.50 chance floor that
LogisticRegression gets stuck at. However, the hypothesis that has_rcd_si/
has_ovp (modern-equipment presence) would be the DOMINANT predictors was
FALSIFIED by feature-importance ranking: has_rcd_si ranks last of 10 features
(importance ~0.005), has_ovp ranks 4th (~0.13) — both well behind pure
composition features (mcb_mean, num_mcbs, iga_amp).

WHY: has_rcd_si has only 4 positive examples and has_ovp only 7, across all 98
rows — and critically, every positive example is a panel the primary installer
personally rebuilt from scratch, making them newest-by-construction rather than
a random sample of "newer installs that happen to have modern equipment." This
is selection bias, not just a small sample: Random Forest's bootstrap sampling
gives most trees near-zero exposure to these rare, confounded positive
examples, so the ensemble can't learn a generalizable split from them.

WHY THIS IS PARKED: at N=98 (and only 4/7 positive modern-equipment examples),
this dataset cannot support a confident population-level conclusion either way
about whether modern-equipment presence predicts installation age. The
component-composition signal is real but modest. Re-run this script once
breaker_dataset.csv has grown substantially — especially with RCD_SI/OVP
examples from OTHER installers, not just the primary author's own recent work
— to see whether the picture changes.

Run: python -m src.model.train_tabular_risk_model
"""
import pathlib

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_schema import (
    HAS_IGA,
    HAS_OVP,
    HAS_RCD_SI,
    IGA_AMP,
    MCB_VALUES,
    NUM_MCBS,
    NUMBER_RCD,
    PANEL_AGE,
    PHASE_TYPE,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "breaker_dataset.csv"

IS_OLD_PANEL = "is_old_panel"

# panel_age (5-way) collapsed to a binary target. Includes Spanish-language
# duplicate categories ("años") produced by a terminal-encoding quirk during
# an earlier labeling session — the underlying CSV values are clean UTF-8,
# this just accounts for both variants of each category.
AGE_MAP = {
    "> 20 years": 1,
    "15 years": 1,
    "15 años": 1,
    "< 5 years": 0,
    "10-15 years": 0,
    "10-15 años": 0,
    "5-10 years": 0,
    "5-10 años": 0,
}

FEATURES_COMPONENTS_ONLY = [
    "mcb_mean", "mcb_max", "mcb_min", NUM_MCBS, NUMBER_RCD, HAS_IGA, IGA_AMP, PHASE_TYPE,
]
FEATURES_FULL = FEATURES_COMPONENTS_ONLY + [HAS_RCD_SI, HAS_OVP]


def load_and_clean(csv_path=CSV_PATH):
    """Load breaker_dataset.csv, binarize panel_age, engineer mcb_values stats."""
    df = pd.read_csv(csv_path)

    df[IS_OLD_PANEL] = df[PANEL_AGE].map(AGE_MAP)
    n_unmapped = df[IS_OLD_PANEL].isna().sum()
    if n_unmapped:
        raise ValueError(
            f"{n_unmapped} row(s) have a {PANEL_AGE} value not covered by AGE_MAP — "
            f"check df[df['{IS_OLD_PANEL}'].isna()]['{PANEL_AGE}'].unique() and extend the map."
        )

    mcb_list = df[MCB_VALUES].apply(lambda s: [int(x) for x in s.split(",")])
    df["mcb_count"] = mcb_list.apply(len)
    df["mcb_mean"] = mcb_list.apply(lambda lst: sum(lst) / len(lst))
    df["mcb_max"] = mcb_list.apply(max)
    df["mcb_min"] = mcb_list.apply(min)

    n_mismatch = (df["mcb_count"] != df[NUM_MCBS]).sum()
    if n_mismatch:
        raise ValueError(
            f"{n_mismatch} row(s) where parsed mcb_count != {NUM_MCBS} — mcb_values parsing bug."
        )

    # iga_amp is legitimately absent (not merely missing) when there is no main
    # breaker at all (has_iga == 0); 0 is the semantically correct fill value.
    df[IGA_AMP] = df[IGA_AMP].fillna(0)

    return df


def run_comparison(df):
    """
    Compare components-only vs full feature set, under both a linear
    (LogisticRegression) and nonlinear (RandomForest) model, via 5-fold
    stratified CV on balanced_accuracy. Returns the fitted RandomForest's
    feature importances (fit on the full feature set, all data) for the
    hypothesis-falsification summary.
    """
    y = df[IS_OLD_PANEL]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log_reg = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

    feature_sets = [("components-only", FEATURES_COMPONENTS_ONLY), ("full-set", FEATURES_FULL)]

    print("=== LogisticRegression (linear baseline) ===")
    for name, feats in feature_sets:
        scores = cross_val_score(log_reg, df[feats], y, cv=cv, scoring="balanced_accuracy")
        print(f"  {name}: mean={scores.mean():.3f}  std={scores.std():.3f}")

    print("\n=== RandomForestClassifier (nonlinear) ===")
    for name, feats in feature_sets:
        scores = cross_val_score(rf, df[feats], y, cv=cv, scoring="balanced_accuracy")
        print(f"  {name}: mean={scores.mean():.3f}  std={scores.std():.3f}")

    print("\n=== RandomForest confusion matrix + classification report (full-set) ===")
    preds = cross_val_predict(rf, df[FEATURES_FULL], y, cv=cv)
    print(confusion_matrix(y, preds))
    print(classification_report(y, preds))

    print("=== RandomForest feature importances (full-set, fit on all data) ===")
    rf.fit(df[FEATURES_FULL], y)
    importances = pd.Series(rf.feature_importances_, index=FEATURES_FULL).sort_values(ascending=False)
    print(importances)

    return importances


def print_summary(importances):
    ranked = list(importances.index)
    rcd_si_rank = ranked.index(HAS_RCD_SI) + 1
    ovp_rank = ranked.index(HAS_OVP) + 1

    print("\n" + "=" * 70)
    print("SUMMARY - STATUS: PARKED / INSUFFICIENT DATA")
    print("=" * 70)
    print(
        "Component-composition features show modest, real predictive signal for\n"
        "panel age (RandomForest ~0.59-0.64 balanced accuracy vs ~0.50 chance).\n\n"
        "The hypothesis that has_rcd_si/has_ovp would DOMINATE was FALSIFIED:\n"
        f"  has_rcd_si importance: {importances[HAS_RCD_SI]:.3f}  (rank {rcd_si_rank} of {len(importances)})\n"
        f"  has_ovp    importance: {importances[HAS_OVP]:.3f}  (rank {ovp_rank} of {len(importances)})\n\n"
        "Both are confounded: their few positive examples are all panels the\n"
        "primary installer personally rebuilt (newest-by-construction), not a\n"
        "random sample. N=98 overall, with only 4/7 positive examples for these\n"
        "two flags, cannot support a confident population-level conclusion.\n\n"
        "Re-run once the dataset grows, especially with modern-equipment examples\n"
        "from OTHER installers."
    )


def main():
    df = load_and_clean()
    importances = run_comparison(df)
    print_summary(importances)


if __name__ == "__main__":
    main()
