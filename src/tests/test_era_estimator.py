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
    assert m.era_start == 1990
    assert m.era_end == 2010
    assert m.confidence == "high"


def test_schneider_resi9_signature():
    text = "ERL Scbeider Resi9 MCB C16 R9F12216"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert m.brand == "Schneider Electric"
    assert "Resi9" in m.model_series
    assert m.era_start == 2015
    assert m.era_end is None  # Ongoing


def test_schneider_acti9_signature():
    text = "Schneider Electric Acti9 iC60N C20"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert "Acti9" in m.model_series
    assert m.era_start == 2011


def test_siemens_5sn_vintage_signature():
    text = "SIEMENS 5SN5 25A 380V~"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert m.brand == "Siemens"
    assert "5SN" in m.model_series
    assert m.era_start == 1980
    assert m.era_end == 1996


def test_siemens_5sx_legacy_signature():
    text = "SIEMENS 5SX2 116-7 C16"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert "5SX" in m.model_series
    assert m.era_start == 1996
    assert m.era_end == 2008


def test_siemens_sentron_modern_signature():
    text = "SIEMENS SENTRON 5SL6 116-7 C16"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    m = matches[0]
    assert "SENTRON" in m.model_series
    assert m.era_start == 2008


def test_hager_modern_signature():
    text = "Hager MBN116 C16 6kA"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    assert any("MBN" in m.model_series for m in matches)


def test_abb_s200_modern_signature():
    text = "ABB S201-C16 System Pro M"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    assert any("S200" in m.model_series for m in matches)


def test_legrand_dx3_signature():
    text = "LEGRAND DX3 C16 407784"
    matches = match_catalog_signatures(text)
    assert len(matches) >= 1
    assert any("DX3" in m.model_series for m in matches)
    assert matches[0].era_start == 2012


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
    assert result.era_range == "1990–2010"
    assert "16–36 years" in result.estimated_age_range
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
    # only catalog hit is an obsolete pre-1996 Siemens series -- the other
    # direction of conflict from the first test.
    preds = [
        {"class": "MAINBREAKER"},
        {"class": "RCD"},
        {"class": "OVERSURGE"},
        {"class": "MCB"},
    ]
    ocr_texts = ["SIEMENS 5SN2 C16"]
    result = estimate_panel_era(preds, ocr_texts, current_year=2026)

    assert result.composition_era.startswith("2020")
    assert result.confidence == "low"
    assert any("CONFLICTING EVIDENCE" in e for e in result.evidence)
