"""
Installation-Era Estimation for PanelSafe.

Provides defensible, range-based era estimates for electrical panels using two
complementary sources of evidence:
1. Catalog Brand/Model Lookup: Recognizes manufacturer product line signatures
   from OCR text (e.g. Schneider Multi9 -> ~1990-2010; Siemens 5SX -> ~1996-2008;
   Legrand DX3 -> ~2012-Present) to establish component generation.
2. REBT Composition Baseline: Evaluates panel architecture against Spain's
   Reglamento Electrotécnico para Baja Tensión (REBT) regulatory epochs
   (Pre-1973, REBT 1973, REBT 2002, and Modern 2020+ Surge Protection).

Deliberately does NOT build a continuous age regressor (which suffers from
unobtainable ground truth and camera/yellowing confounders). Reports explicit
ranges and confidence bounds.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CatalogMatch:
    brand: str
    model_series: str
    era_start: int
    era_end: Optional[int]  # None means ongoing / current
    era_label: str
    confidence: str  # "high", "medium", "low"
    notes: str
    matched_token: str


@dataclass
class EraEstimate:
    era_range: str
    era_label: str
    estimated_age_range: str
    rebt_standard: str
    confidence: str  # "high", "medium", "low"
    catalog_matches: List[CatalogMatch] = field(default_factory=list)
    composition_era: str = ""
    evidence: List[str] = field(default_factory=list)
    feedback_es: str = ""
    feedback_en: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "era_range": self.era_range,
            "era_label": self.era_label,
            "estimated_age_range": self.estimated_age_range,
            "rebt_standard": self.rebt_standard,
            "confidence": self.confidence,
            "catalog_matches": [
                {
                    "brand": m.brand,
                    "model_series": m.model_series,
                    "era_start": m.era_start,
                    "era_end": m.era_end,
                    "era_label": m.era_label,
                    "confidence": m.confidence,
                    "notes": m.notes,
                    "matched_token": m.matched_token,
                }
                for m in self.catalog_matches
            ],
            "composition_era": self.composition_era,
            "evidence": self.evidence,
            "feedback_es": self.feedback_es,
            "feedback_en": self.feedback_en,
        }


# Curated catalog database for major European/Spanish residential & commercial brands
CATALOG_SERIES_DB = [
    # --- Schneider Electric / Merlin Gerin / Telemecanique ---
    {
        "brand": "Schneider Electric",
        "model_series": "Multi9 (C60N / C60H / C32N)",
        "patterns": [
            r"\bMULTI\s*9\b",
            r"\bMULTI9\b",
            r"\bC60N\b",
            r"\bC60H\b",
            r"\bC32N\b",
            r"\bC45N\b",
            r"\bMERLIN\s+GERIN\b",
        ],
        "era_start": 1990,
        "era_end": 2010,
        "era_label": "1990–2010 (Legacy / Discontinued)",
        "notes": "Classic Merlin Gerin / Schneider modular line; replaced by Acti9 in 2011.",
    },
    {
        "brand": "Schneider Electric",
        "model_series": "Acti9 (iC60 / iID)",
        "patterns": [
            r"\bACTI\s*9\b",
            r"\bACTI9\b",
            r"\bIC60N?\b",
            r"\bIC60H\b",
            r"\bIID\b",
        ],
        "era_start": 2011,
        "era_end": None,
        "era_label": "2011–Present (Modern Modular)",
        "notes": "Current flagship commercial/residential platform introduced in 2011.",
    },
    {
        "brand": "Schneider Electric",
        "model_series": "Resi9 (R9F / R9R)",
        "patterns": [
            r"\bRESI\s*9\b",
            r"\bRESI9\b",
            r"\bR9F\d{5}\b",
            r"\bR9R\d{5}\b",
        ],
        "era_start": 2015,
        "era_end": None,
        "era_label": "2015–Present (Modern Residential)",
        "notes": "Schneider residential modular line launched in 2015.",
    },
    {
        "brand": "Schneider Electric",
        "model_series": "Domae",
        "patterns": [
            r"\bDOMAE\b",
            r"\bDOM\d{2,}\b",
        ],
        "era_start": 2002,
        "era_end": 2018,
        "era_label": "2002–2018 (Legacy Residential)",
        "notes": "Standard domestic builder line throughout the 2000s and 2010s.",
    },
    # --- Siemens ---
    {
        "brand": "Siemens",
        "model_series": "5SN / 5SZ (Vintage)",
        "patterns": [
            r"\b5SN\d?\b",
            r"\b5SZ\d?\b",
            r"\b5SM\d?\b",
        ],
        "era_start": 1980,
        "era_end": 1996,
        "era_label": "1980–1996 (Obsolete / Vintage)",
        "notes": "Early Siemens modular series; predates REBT 2002.",
    },
    {
        "brand": "Siemens",
        "model_series": "5SX / 5SM3 (Legacy)",
        "patterns": [
            r"\b5SX\d?\b",
            r"\b5SM3\b",
        ],
        "era_start": 1996,
        "era_end": 2008,
        "era_label": "1996–2008 (Legacy)",
        "notes": "Dominant Siemens DIN-rail series across Europe in late 90s/early 2000s.",
    },
    {
        "brand": "Siemens",
        "model_series": "SENTRON (5SL / 5SY / 5SV)",
        "patterns": [
            r"\b5SL\d?\b",
            r"\b5SY\d?\b",
            r"\b5SV\d?\b",
            r"\bSENTRON\b",
        ],
        "era_start": 2008,
        "era_end": None,
        "era_label": "2008–Present (Modern SENTRON)",
        "notes": "Modern Siemens SENTRON platform.",
    },
    # --- Hager ---
    {
        "brand": "Hager",
        "model_series": "MW / ML (Legacy)",
        "patterns": [
            r"\bMW\d{3}\b",
            r"\bML\d{3}\b",
        ],
        "era_start": 1985,
        "era_end": 2000,
        "era_label": "1985–2000 (Legacy)",
        "notes": "Earlier Hager modular range with screw clamp terminals.",
    },
    {
        "brand": "Hager",
        "model_series": "MBN / NBN / CDA (SanVis / Modern)",
        "patterns": [
            r"\bMBN\d{3}\b",
            r"\bNBN\d{3}\b",
            r"\bCDA\d{3}\b",
            r"\bCDC\d{3}\b",
            r"\bHAGER\b",
        ],
        "era_start": 2000,
        "era_end": None,
        "era_label": "2000–Present (Modern Hager)",
        "notes": "Standard Hager modular line with QuickConnect/SanVis technology.",
    },
    # --- ABB ---
    {
        "brand": "ABB",
        "model_series": "S250 / S260 / S270 / S280 (Vintage)",
        "patterns": [
            r"\bS25\d\b",
            r"\bS26\d\b",
            r"\bS27\d\b",
            r"\bS28\d\b",
            r"\bBBC\b",
        ],
        "era_start": 1982,
        "era_end": 2000,
        "era_label": "1982–2000 (Legacy ABB/BBC)",
        "notes": "Classic Brown Boveri / ABB modular breakers.",
    },
    {
        "brand": "ABB",
        "model_series": "System Pro M Compact (S200 / SN201)",
        "patterns": [
            r"\bS20[1-4]\b",
            r"\bS200\b",
            r"\bSN201\b",
            r"\bF20[24]\b",
        ],
        "era_start": 2000,
        "era_end": None,
        "era_label": "2000–Present (Modern System Pro M)",
        "notes": "Current ABB System Pro M compact line.",
    },
    # --- Legrand ---
    {
        "brand": "Legrand",
        "model_series": "Rex / 013 / 014 (Vintage)",
        "patterns": [
            r"\bREX\b",
            r"\bLEGRAND\s+01[34]\b",
        ],
        "era_start": 1975,
        "era_end": 1995,
        "era_label": "1975–1995 (Obsolete / Vintage)",
        "notes": "Pre-DX Legrand modular equipment.",
    },
    {
        "brand": "Legrand",
        "model_series": "Lexic / DX",
        "patterns": [
            r"\bLEXIC\b",
            r"\bDX\b(?!\d)",
        ],
        "era_start": 1995,
        "era_end": 2012,
        "era_label": "1995–2012 (Legacy Lexic/DX)",
        "notes": "Predecessor to DX3 series.",
    },
    {
        "brand": "Legrand",
        "model_series": "DX3 / TX3 / RX3",
        "patterns": [
            r"\bDX3\b",
            r"\bTX3\b",
            r"\bRX3\b",
            r"\bDX\^3\b",
        ],
        "era_start": 2012,
        "era_end": None,
        "era_label": "2012–Present (Modern DX3)",
        "notes": "Current Legrand commercial and residential range.",
    },
    # --- General Electric ---
    {
        "brand": "General Electric",
        "model_series": "Redline (EP60 / EP100)",
        "patterns": [
            r"\bREDLINE\b",
            r"\bEP60\b",
            r"\bEP100\b",
            r"\bDMS\b",
        ],
        "era_start": 1995,
        "era_end": 2018,
        "era_label": "1995–2018 (Legacy GE Redline)",
        "notes": "General Electric industrial solutions series before ABB acquisition in 2018.",
    },
    # --- Chint ---
    {
        "brand": "Chint",
        "model_series": "eBG / NB1 / DZ47",
        "patterns": [
            r"\bCHINT\b",
            r"\bEBG\b",
            r"\bNB1\b",
            r"\bDZ47\b",
        ],
        "era_start": 2005,
        "era_end": None,
        "era_label": "2005–Present (Modern Chint)",
        "notes": "Standard Chint modular breaker lines in European distribution.",
    },
]


def match_catalog_signatures(text: str) -> List[CatalogMatch]:
    """
    Matches raw OCR text or tokens against the curated catalog database.
    Returns all detected catalog matches with token boundaries.
    """
    if not text:
        return []

    text_upper = text.upper()
    matches: List[CatalogMatch] = []
    seen_series = set()

    for item in CATALOG_SERIES_DB:
        for pat in item["patterns"]:
            m = re.search(pat, text_upper)
            if m:
                series_key = (item["brand"], item["model_series"])
                if series_key not in seen_series:
                    seen_series.add(series_key)
                    # Assess confidence based on pattern specificity
                    matched_tok = m.group(0)
                    is_generic_brand_only = matched_tok in ["HAGER", "CHINT"]
                    confidence = "medium" if is_generic_brand_only else "high"

                    matches.append(
                        CatalogMatch(
                            brand=item["brand"],
                            model_series=item["model_series"],
                            era_start=item["era_start"],
                            era_end=item["era_end"],
                            era_label=item["era_label"],
                            confidence=confidence,
                            notes=item["notes"],
                            matched_token=matched_tok,
                        )
                    )
                break  # Don't duplicate matches within same series

    return matches


def estimate_rebt_composition_era(predictions: List[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
    """
    Infers the installation era baseline from panel component composition
    according to Spanish electrical code (REBT) evolutions.

    Returns:
        (composition_era, rebt_standard, evidence_list)
    """
    rcd_count = sum(1 for p in predictions if p.get("class") in ["RCD", "RCD_SI"])
    mcb_count = sum(1 for p in predictions if p.get("class") == "MCB")
    main_breaker_count = sum(1 for p in predictions if p.get("class") == "MAINBREAKER")
    oversurge_count = sum(1 for p in predictions if p.get("class") == "OVERSURGE")
    has_rcd_si = any(p.get("class") == "RCD_SI" for p in predictions)

    evidence = []

    # 1. Pre-1973: No RCD protection
    if rcd_count == 0 and mcb_count <= 3:
        evidence.append("No differential (RCD) protection detected; minimal circuit separation.")
        return (
            "Pre-1973 (Pre-Normative)",
            "Pre-REBT 1973 (Obsolete)",
            evidence,
        )

    # 2. REBT-1973 (1973–2002): Has RCD, but no dedicated IGA and no oversurge
    if main_breaker_count == 0 and oversurge_count == 0:
        if rcd_count > 0:
            evidence.append("RCD detected but lacking dedicated General Automatic Switch (IGA).")
        evidence.append("No surge protection (Sobretensiones / DPS) present.")
        return (
            "1973–2002 (REBT-1973 Era)",
            "REBT 1973 (Real Decreto 2413/1973)",
            evidence,
        )

    # 3. Modern REBT 2020+ (ITC-BT-23 / ITC-BT-25 update): Has IGA+DPS / Oversurge or Superinmunizado
    if oversurge_count > 0 or has_rcd_si:
        if oversurge_count > 0:
            evidence.append("Combined permanent & transient surge protection (OVERSURGE / IGA+DPS) detected.")
        if has_rcd_si:
            evidence.append("Superinmunizado / Type A differential protection (RCD_SI) present.")
        return (
            "2020–Present (Modern REBT Update)",
            "REBT 2002 + ITC-BT-23/25 Actualizada",
            evidence,
        )

    # 4. Standard REBT-2002 (2002–2019): Dedicated IGA present, 30mA RCDs, standard MCBs
    evidence.append("Dedicated General Automatic Switch (IGA) detected.")
    if rcd_count > 0:
        evidence.append("Individual 30mA differential (RCD) circuit segmentation present.")
    if oversurge_count == 0:
        evidence.append("No surge protector (DPS) detected (standard for 2002–2019 domestic builds).")

    return (
        "2002–2019 (REBT-2002 Era)",
        "REBT 2002 (Real Decreto 842/2002)",
        evidence,
    )


def _composition_numeric_range(comp_era: str, current_year: int) -> Tuple[int, int]:
    """Approximate (start, end) year bounds for one of the four fixed strings
    returned by estimate_rebt_composition_era(), used only to sanity-check a
    catalog match against the composition signal -- not shown to the user."""
    if "Pre-1973" in comp_era:
        return (0, 1973)
    if "1973" in comp_era and "2002" in comp_era:
        return (1973, 2002)
    if "2020" in comp_era:
        return (2020, current_year)
    return (2002, 2019)  # "2002-2019 (REBT-2002 Era)", the default branch


def estimate_panel_era(
    predictions: List[Dict[str, Any]],
    ocr_texts: Optional[List[str]] = None,
    current_year: int = 2026,
) -> EraEstimate:
    """
    Computes a unified, defensible era estimate for a panel by synthesizing
    catalog OCR evidence with REBT composition rules.
    """
    # 1. Run composition baseline
    comp_era, rebt_std, comp_evidence = estimate_rebt_composition_era(predictions)

    # 2. Extract catalog signatures from OCR texts
    catalog_matches: List[CatalogMatch] = []
    if ocr_texts:
        combined_text = " ".join(t for t in ocr_texts if t)
        catalog_matches = match_catalog_signatures(combined_text)

    all_evidence = list(comp_evidence)

    # 3. Reconcile evidence into unified era range
    if catalog_matches:
        min_start = min(m.era_start for m in catalog_matches)
        max_end_candidates = [m.era_end for m in catalog_matches if m.era_end is not None]
        has_ongoing = any(m.era_end is None for m in catalog_matches)

        if max_end_candidates and not has_ongoing:
            max_end = max(max_end_candidates)
            era_range = f"{min_start}–{max_end}"
            min_age = current_year - max_end
            max_age = current_year - min_start
            age_range = f"{min_age}–{max_age} years"
            era_label = f"Catalog Match ({era_range})"
        elif has_ongoing and max_end_candidates:
            era_range = f"{min_start}–Present"
            max_age = current_year - min_start
            age_range = f"0–{max_age} years"
            era_label = f"Catalog Match ({era_range})"
        elif has_ongoing:
            era_range = f"{min_start}–Present"
            max_age = current_year - min_start
            age_range = f"0–{max_age} years"
            era_label = f"Catalog Match ({era_range})"
        else:
            era_range = f"~{min_start}s"
            age_range = f"~{current_year - min_start} years"
            era_label = f"Catalog Match ({era_range})"

        for cm in catalog_matches:
            end_str = str(cm.era_end) if cm.era_end else "Present"
            all_evidence.append(
                f"Detected {cm.brand} {cm.model_series} ({cm.era_start}–{end_str}): {cm.notes}"
            )
        confidence = "high" if any(m.confidence == "high" for m in catalog_matches) else "medium"

        # A catalog match was previously trusted outright, even when it flatly
        # contradicted the composition signal (e.g. zero RCD -> Pre-1973 baseline,
        # but a single stray OCR token from a brand name -> "2012-Present, high
        # confidence"). This project has already hit exactly this failure mode
        # once this cycle (an unguarded "SI" substring match silently overriding
        # correct OCR reads) -- a lone token should not be able to overrule a
        # strong structural signal at high confidence.
        catalog_max_end = current_year if has_ongoing else max(max_end_candidates, default=current_year)
        comp_start, comp_end = _composition_numeric_range(comp_era, current_year)
        overlaps = min_start <= comp_end and comp_start <= catalog_max_end
        if not overlaps:
            confidence = "low"
            all_evidence.append(
                "CONFLICTING EVIDENCE: the panel's component composition points to a "
                f"different era ({comp_era}) than the catalog match above. Reporting "
                "the catalog range with reduced confidence rather than resolving the "
                "conflict silently -- one OCR token is weaker evidence than the "
                "overall panel composition."
            )
    else:
        # Fallback strictly to composition era
        if "Pre-1973" in comp_era:
            era_range = "Pre-1973"
            age_range = ">50 years"
            era_label = "Pre-Normative (~1970s or earlier)"
            confidence = "medium"
        elif "1973–2002" in comp_era:
            era_range = "1973–2002"
            age_range = "24–53 years"
            era_label = "REBT-1973 Installation Era"
            confidence = "medium"
        elif "2020" in comp_era:
            era_range = "2020–Present"
            age_range = "<6 years"
            era_label = "Modern REBT Installation Era (Surge Protected)"
            confidence = "high"
        else:  # 2002–2019
            era_range = "2002–2019"
            age_range = "7–24 years"
            era_label = "REBT-2002 Installation Era"
            confidence = "medium"

    # Format feedback summaries (ES / EN)
    feedback_es = (
        f"📅 <strong>Estimación de Época de Instalación:</strong> {era_label} ({era_range}, aprox. {age_range}). "
        f"Normativa de referencia: <em>{rebt_std}</em>."
    )
    feedback_en = (
        f"📅 <strong>Estimated Installation Era:</strong> {era_label} ({era_range}, approx. {age_range}). "
        f"Applicable standard: <em>{rebt_std}</em>."
    )

    # confidence == "low" means this call's own conflict check (above) found the
    # catalog match and the panel's composition pointing at DIFFERENT eras. Without
    # this, the user sees a clean, confident-looking one-liner with no hint that two
    # signals disagreed and one was picked over the other -- the exact "smoothed-over
    # certainty" this project has deliberately avoided everywhere else (the degraded-
    # score fix, the OCR neutral fallback). Surface the ambiguity instead of hiding it.
    if confidence == "low":
        feedback_es += (
            " ⚠️ <em>Confianza baja: la composición del cuadro sugiere una época distinta "
            "a la indicada por el texto detectado en los dispositivos; este rango debe "
            "tratarse como orientativo.</em>"
        )
        feedback_en += (
            " ⚠️ <em>Low confidence: the panel's composition suggests a different era than "
            "the text detected on its devices; treat this range as tentative.</em>"
        )

    return EraEstimate(
        era_range=era_range,
        era_label=era_label,
        estimated_age_range=age_range,
        rebt_standard=rebt_std,
        confidence=confidence,
        catalog_matches=catalog_matches,
        composition_era=comp_era,
        evidence=all_evidence,
        feedback_es=feedback_es,
        feedback_en=feedback_en,
    )
