"""
Unit tests for the Installation-Era Estimation module (src.model.era_estimator).
Covers catalog brand/model matching, REBT composition rules, and unified range reporting.
"""

import pytest

from src.model.era_estimator import (
    EraEstimate,
    estimate_panel_era,
    estimate_rebt_composition_era,
    match_catalog_signatures,
)

# --- CATALOG SIGNATURE MATCHING TESTS ---------------------------------------

def test_schneider_multi9_signature():
    text = "MERLIN GERIN Multi 9 C60N C16 400V~"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert m.brand == "Schneider Electric"
    assert "Multi9" in m.model_series
    assert m.era_start == 1974
    assert m.era_end == 2011
    assert m.confidence == "high"


def test_schneider_resi9_signature():
    text = "ERL Scbeider Resi9 MCB C16 R9F12216"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert m.brand == "Schneider Electric"
    assert "Resi9" in m.model_series
    assert m.era_start == 2016
    assert m.era_end is None  # Ongoing


def test_schneider_acti9_signature():
    text = "Schneider Electric Acti9 iC60N C20"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert "Acti9" in m.model_series
    assert m.era_start == 2011


def test_legrand_dx3_signature():
    text = "LEGRAND DX3 C16 407784"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    assert any("DX3" in m.model_series for m in matches)
    assert matches[0].era_start == 2011


def test_removed_unverified_signatures_return_no_matches():
    # Entries removed in T1 because manufacturer years were unverified/hallucinated
    # must safely return no match and fall back to REBT composition rules.
    unverified_texts = [
        "SIEMENS 5SN5 25A 380V~",
        "SIEMENS 5SX2 116-7 C16",
        "SIEMENS SENTRON 5SL6 116-7 C16",
        "Hager MBN116 C16 6kA",
        "ABB S201-C16 System Pro M",
        "LEGRAND 013 00 C16",
        "GE REDLINE EP60 C16",
        "CHINT NB1-63 C16",
    ]
    for text in unverified_texts:
        assert match_catalog_signatures(text) == []


def test_empty_or_garbled_ocr_returns_no_catalog_matches():
    assert match_catalog_signatures("") == []
    assert match_catalog_signatures("XYZ123 999 NO MATCH") == []


# --- REBT COMPOSITION BASELINE TESTS ---------------------------------------

def test_pre_1973_composition_no_rcd():
    preds = [
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    era, std, evidence = estimate_rebt_composition_era(preds)
    assert "Pre-1973" in era
    assert "Obsolete" in std or "Pre-REBT" in std
    assert any("No differential" in e for e in evidence)


def test_rebt_1973_composition_rcd_no_iga():
    preds = [
        {"class": "RCD"},
        {"class": "MCB"},
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    era, std, evidence = estimate_rebt_composition_era(preds)
    assert "1973–2002" in era
    assert "REBT 1973" in std
    assert any("No surge protection" in e for e in evidence)


def test_rebt_2002_composition_standard():
    preds = [
        {"class": "MAINBREAKER"},
        {"class": "RCD"},
        {"class": "MCB"},
        {"class": "MCB"},
        {"class": "MCB"},
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    era, std, evidence = estimate_rebt_composition_era(preds)
    assert "2002–2019" in era
    assert "REBT 2002" in std
    assert any("Dedicated General Automatic Switch (IGA)" in e for e in evidence)


def test_modern_2020_composition_with_oversurge():
    preds = [
        {"class": "OVERSURGE"},
        {"class": "MAINBREAKER"},
        {"class": "RCD"},
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    era, std, evidence = estimate_rebt_composition_era(preds)
    assert "2020–Present" in era
    assert "ITC-BT-23/25" in std
    assert any("Combined permanent & transient surge protection" in e for e in evidence)


def test_modern_2020_composition_with_rcd_si():
    preds = [
        {"class": "MAINBREAKER"},
        {"class": "RCD_SI"},
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    era, std, evidence = estimate_rebt_composition_era(preds)
    assert "2020–Present" in era
    assert any("Superinmunizado" in e for e in evidence)


# --- UNIFIED ESTIMATION RECONCILIATION TESTS --------------------------------

def test_unified_estimate_with_catalog_and_composition():
    preds = [
        {"class": "MAINBREAKER"},
        {"class": "RCD"},
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    ocr_texts = [
        "SCHNEIDER Multi 9 C60N C16",
        "SCHNEIDER Multi 9 C60N C20",
    ]
    result = estimate_panel_era(preds, ocr_texts, current_year=2026)

    assert isinstance(result, EraEstimate)
    assert result.era_range == "1974–2011"
    assert "15–52 years" in result.estimated_age_range
    assert result.confidence == "high"
    assert len(result.catalog_matches) == 1
    assert "Multi9" in result.catalog_matches[0].model_series
    assert "REBT 2002" in result.rebt_standard
    assert "Estimación de Época de Instalación" in result.feedback_es
    assert "Estimated Installation Era" in result.feedback_en


def test_unified_estimate_without_ocr_falls_back_to_composition():
    preds = [
        {"class": "MAINBREAKER"},
        {"class": "RCD"},
        {"class": "MCB"},
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    result = estimate_panel_era(preds, ocr_texts=[], current_year=2026)

    assert result.era_range == "2002–2019"
    assert result.confidence == "medium"
    assert len(result.catalog_matches) == 0
    assert "REBT-2002 Installation Era" in result.era_label


def test_era_estimate_serialization():
    preds = [{"class": "MAINBREAKER"}, {"class": "RCD"}, {"class": "MCB"}]
    ocr_texts = ["Legrand DX3 C16"]
    est = estimate_panel_era(preds, ocr_texts, current_year=2026)
    d = est.to_dict()

    assert "era_range" in d
    assert "rebt_standard" in d
    assert "catalog_matches" in d
    assert isinstance(d["catalog_matches"], list)
    assert d["catalog_matches"][0]["brand"] == "Legrand"


# --- conflict reconciliation (found in review 2026-08-16) ------------------------------
# A catalog match previously overrode the composition signal outright and reported
# "high" confidence even when the two flatly disagreed -- e.g. zero RCD detected
# (composition: Pre-1973) but a single OCR token matching a 2012-present catalog
# series. This project has already hit the "one token overrides a strong signal"
# failure mode once this cycle (the unguarded "SI" substring match), so a lone
# brand-name token must not be able to silently overrule the panel's composition.

def test_catalog_match_conflicting_with_composition_downgrades_confidence():
    # No RCD, minimal MCBs -> composition baseline is Pre-1973. A modern-era
    # catalog token (2012-present) directly contradicts that.
    preds = [{"class": "MCB"}, {"class": "MCB"}]
    ocr_texts = ["LEGRAND DX3 C16"]
    result = estimate_panel_era(preds, ocr_texts, current_year=2026)

    assert result.composition_era.startswith("Pre-1973")
    assert result.confidence == "low"
    assert any("CONFLICTING EVIDENCE" in e for e in result.evidence)


def test_catalog_match_consistent_with_composition_stays_high_confidence():
    # Same fixture as test_unified_estimate_with_catalog_and_composition: a
    # 1990-2010 catalog match against a 2002-2019 composition baseline overlaps
    # (2002-2010), so this must NOT be flagged as conflicting.
    preds = [
        {"class": "MAINBREAKER"},
        {"class": "RCD"},
        {"class": "MCB"},
        {"class": "MCB"},
    ]
    ocr_texts = ["SCHNEIDER Multi 9 C60N C16", "SCHNEIDER Multi 9 C60N C20"]
    result = estimate_panel_era(preds, ocr_texts, current_year=2026)

    assert result.confidence == "high"
    assert not any("CONFLICTING EVIDENCE" in e for e in result.evidence)


def test_catalog_match_conflicting_with_modern_composition_downgrades_confidence():
    # Composition says modern (surge protector present -> 2020-Present), but the
    # only catalog hit is an obsolete legacy series (Multi 9: 1974-2011) -- the other
    # direction of conflict from the first test.
    preds = [
        {"class": "MAINBREAKER"},
        {"class": "RCD"},
        {"class": "OVERSURGE"},
        {"class": "MCB"},
    ]
    ocr_texts = ["SCHNEIDER Multi 9 C60N C16"]
    result = estimate_panel_era(preds, ocr_texts, current_year=2026)

    assert result.composition_era.startswith("2020")
    assert result.confidence == "low"
    assert any("CONFLICTING EVIDENCE" in e for e in result.evidence)


# --- confidence surfaced to the user (Ontological-framework audit, 2026-08-16) ---------
# EraEstimate.confidence and the CONFLICTING EVIDENCE note were computed but never
# reached feedback_es/feedback_en -- a low-confidence estimate looked identical to a
# high-confidence one in the report a homeowner actually sees.

def test_low_confidence_estimate_carries_a_visible_qualifier_in_both_languages():
    preds = [{"class": "MCB"}, {"class": "MCB"}]  # composition -> Pre-1973
    result = estimate_panel_era(preds, ["LEGRAND DX3 C16"], current_year=2026)  # catalog -> 2012-Present

    assert result.confidence == "low"
    assert "Confianza baja" in result.feedback_es
    assert "Low confidence" in result.feedback_en


def test_high_confidence_estimate_carries_no_qualifier():
    preds = [
        {"class": "MAINBREAKER"}, {"class": "RCD"}, {"class": "MCB"}, {"class": "MCB"},
    ]
    result = estimate_panel_era(
        preds, ["SCHNEIDER Multi 9 C60N C16", "SCHNEIDER Multi 9 C60N C20"], current_year=2026
    )

    assert result.confidence == "high"
    assert "Confianza baja" not in result.feedback_es
    assert "Low confidence" not in result.feedback_en


def test_medium_confidence_composition_only_carries_no_low_confidence_qualifier():
    # No catalog match at all -> falls back to composition-only, which is "medium",
    # not "low". The qualifier is specifically for the CONFLICT case, not every
    # non-"high" result.
    preds = [
        {"class": "MAINBREAKER"}, {"class": "RCD"}, {"class": "MCB"}, {"class": "MCB"}, {"class": "MCB"},
    ]
    result = estimate_panel_era(preds, ocr_texts=[], current_year=2026)

    assert result.confidence == "medium"
    assert "Confianza baja" not in result.feedback_es
    assert "Low confidence" not in result.feedback_en
