"""
RED-phase spec for the _clean_ocr_text rewrite (roadmap: "Breaker Text Reading &
Installation-Era Estimation", Stage 1).

Every raw_text string below is REAL OCR output captured from the 1,060-crop dump on
2026-07-28, not invented. The `expected` column encodes the agreed design:

  1. Tokenize on [A-Z0-9]+ and match WHOLE TOKENS, never substrings.
     (This alone kills all 58 measured SI false positives -- in every one of them SI is
     embedded in a longer token: SIEMENS, RESI9, SIMON, EBSIN, MNSI6V, JENSION.)
  2. A curve+rating is a whole token [BCD]\\d{1,2} whose value is a REAL IEC rating.
  3. The bare-amperage fallback is DELETED: a bare "40A" carries zero MCB-vs-RCD
     information because RCDs are rated in amps too. Only the curve letter discriminates.
  4. "SI" is a MODIFIER on an RCD verdict, never a standalone verdict -- an RCD_SI is a
     subtype of RCD. It requires a standalone SI token AND corroborating RCD evidence.
  5. Leakage marker outranks curve+rating when both appear (measured 93.8% precision on
     the RCD family, and it is a far more specific pattern than two chars of curve+digits).

Valid IEC ratings: {1,2,3,4,6,10,13,16,20,25,32,40,50,63}
"""
import pytest

from src.model.ocr_reader import OCRReader


@pytest.fixture(scope="module")
def clean():
    return OCRReader()._clean_ocr_text


# --- BUG 1: "SI" substring collision (58/58 false positives, measured) ---------------
# The killer detail: OCR read "MCB C16" perfectly -- twice in one string -- and the old
# code threw it away because "Resi9" contains "SI".

SI_COLLISION_CASES = [
    ("ERL Scbeider Resi9 MCB C16 R9F12216} 1 LON L.On 4OOv~ C16 6000.3", "C16"),
    ("Schleider Resi9 MCB C10 R9F12610 1 I.on I.ON 230v~ C10 600018", "C10"),
    ("Scbeider Electric Resi9 J MCB C25 8 R9F12225 1 O.OFF O.OFF N", "C25"),
    ("Scteider Resi9 MCB C25 R9F12225 1 L.ON L.ON 40Ov~ C25 Go00", "C25"),
    # SIEMENS collision on a true MCB; "(25A}" is a bare amperage -> no longer usable,
    # and there is no curve letter anywhere, so the safe answer is no signal.
    ("SIEMENS 5 SN 5 No.28-U NIL (25A} 320 Jenon", ""),
    # SIEMENS on a true RCD, no recoverable marker at all.
    ("SIEMENS NA-Schulschalet SSZI 443 e4 De Tvea Rov = Be", ""),
    # True RCD: leakage marker present and must win.
    ("ial Scbzeider Resi9 ID Regular test R9R51240 8 O.OFF Typa AC 40A 30mA", "30MA"),
]


@pytest.mark.parametrize("raw,expected", SI_COLLISION_CASES)
def test_si_substring_never_produces_a_bare_si_verdict(clean, raw, expected):
    assert clean(raw) == expected


def test_no_si_verdict_survives_the_whole_measured_collision_set(clean):
    """Blanket guard: none of the real collision strings may return 'SI'."""
    for raw, _ in SI_COLLISION_CASES:
        assert clean(raw) != "SI", f"SI false positive still fires on: {raw!r}"


# --- BUG 2: amperage fallback scraping digits out of MODEL NUMBERS (40 verdicts) ------
# Chint NL1-63 is an RCD model. The old fallback read "63" out of the part number and
# returned "C63", pushing a true RCD toward MCB at 19:1.

MODEL_NUMBER_CASES = [
    ("TES CHNT  NL1-63 40Alan Q.024 ZJ0v~ 0.15 Test regularly Lm 5004", ""),
    ("CHNT NL1-63 40AIan= 230v ~ Test Kneun leele EQudA IECJENG1008", ""),
    # "IECIENB1008-1" previously yielded B10 via a substring search.
    ("CHNT NL1-63 40AIan 0.624 Tebl Ielmeeooa Ec= Lad =Boioa IECIENB1008-1 B@DOHI", ""),
    ("40 008", ""),
]


@pytest.mark.parametrize("raw,expected", MODEL_NUMBER_CASES)
def test_digits_inside_model_numbers_are_not_ratings(clean, raw, expected):
    assert clean(raw) == expected


# --- BUG 3: physically impossible ratings (11% of verdicts, 34/313) ------------------

IMPOSSIBLE_RATING_CASES = [
    # "Rc 8" produced C8 via substring search; C16 is the real, valid reading present.
    ('Scbleider" S Re DomA67 @ Rc 8 20A Boop C16', "C16"),
    # "C625" is 3 digits -> not a whole-token rating; old code sliced "C62" out of it.
    ("MERLIN GERIN (multi9 KGON C625 231", ""),
    # C8 is a well-formed token but 8A is not a real IEC rating.
    ("JMERLIN GERINI multi9 KGON C8 2301 Dqql 13 flo", ""),
    ("GEAN 4614 (Fn elr? 7uil'g 304 251~ tod 95At 74 ' 150 Mekln", ""),
]


@pytest.mark.parametrize("raw,expected", IMPOSSIBLE_RATING_CASES)
def test_impossible_ratings_are_rejected(clean, raw, expected):
    assert clean(raw) == expected


# --- Signal that MUST be preserved (guard against over-correction) -------------------
# Precision is worthless if the parser becomes so strict it returns "" for everything.

GOOD_SIGNAL_CASES = [
    ("MERLIN GERIN multlg KGON C25 23OV", "C25"),
    ("MERLIN GERIN multlg KGON C25 20v", "C25"),
    # "NXB-63" is a model number; C16 is the real rating. Both present -- pick correctly.
    ("CHNT NXB-63 C16 4QOV = @Vadd", "C16"),
    ("BD62 In:40A  Zpd Ian-0.03A Un-230V~ 02 V/3o4.124031 599516", "30MA"),
    ("Scoreider dotel (Rale Aa 30ma  40A", "30MA"),
    ("25A ZP BP Pulsar T Frecuentemonte 30mA 230v ~ 807- ION", "30MA"),
]


@pytest.mark.parametrize("raw,expected", GOOD_SIGNAL_CASES)
def test_valid_signal_is_still_extracted(clean, raw, expected):
    assert clean(raw) == expected


# --- Structural rules ----------------------------------------------------------------

def test_leakage_marker_outranks_curve_rating(clean):
    """Both markers present: the far more specific leakage pattern wins."""
    assert clean("Resi9 ID C16 40A 30mA Typa AC") == "30MA"


def test_si_requires_a_standalone_token_and_rcd_evidence(clean):
    """A real RCD_SI: standalone SI token corroborated by a leakage marker."""
    assert clean("HAGER CDC742D 40A 30mA SI Type A") == "SI"


def test_si_alone_without_rcd_evidence_is_not_enough(clean):
    """Two loose characters are not sufficient evidence on their own."""
    assert clean("ACME SI 230V") == ""


def test_bare_amperage_is_not_a_rating(clean):
    """RCDs are rated in amps too, so a bare amperage cannot discriminate MCB vs RCD."""
    assert clean("40A 230V~") == ""
    assert clean("25A 2P") == ""


def test_literal_pipe_is_not_a_curve_letter(clean):
    """The old pattern [B|C|D] was a character class containing a literal pipe.
    Panel edges OCR as pipes, so this must not be read as a curve."""
    assert clean("Scbzeider | 16 R9F12216") == ""


def test_empty_and_garbage_input_is_neutral(clean):
    assert clean("") == ""
    assert clean("   ") == ""
    assert clean("@#$ ~~~ ...") == ""


# --- Regressions found by external review 2026-08-09 -----------------------------------
# All three were real by construction but had ZERO occurrences in the 1,060-crop
# sample, so they are guarded by tests rather than left to resurface.

def test_300ma_rcd_is_not_read_as_30ma(clean):
    """0.3A is a 300mA fire-protection RCD -- a different device from a 30mA one.

    An earlier `0[.,]0?3` made the second zero optional and matched it.
    """
    assert clean("ID 40A 0.3A Type AC") != "30MA"
    assert clean("ID 40A 0,3A Type AC") != "30MA"


def test_genuine_30ma_still_matches_in_messy_ocr(clean):
    """The lookarounds must not undo the leading-glyph tolerance."""
    assert clean("BD62 InF404 Isn*0034 Ian-0,03A Un 230") == "30MA"
    assert clean("40 A Ly ~ 8060 94 30 mAL Le Lu") == "30MA"


def test_leakage_marker_does_not_match_inside_a_longer_number(clean):
    """"130MA" is a model-number fragment, not a 30mA rating."""
    assert clean("CHNT NXB 130MA 25") != "30MA"
    assert clean("SERIES 2130 MA") != "30MA"


def test_zero_padded_rating_is_canonicalised(clean):
    """"C06" and "C6" are the same rating; emit one spelling."""
    assert clean("SCHNEIDER C06 6000") == "C6"
