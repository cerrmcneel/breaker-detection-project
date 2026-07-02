"""
Single source of truth for the panel-attribute CSV schema.

This schema is shared by the two sides of the tabular safety-scoring track so they
cannot silently drift:
  - WRITER: src/tools/labeler.py  -> writes data/breaker_dataset.csv (columns = CSV_FIELDNAMES)
  - READER: src/scoring.py        -> reads a subset of those columns (+ derived fields)

If you rename/add/remove a column, change it HERE. The string VALUES below are the wire
format (the actual CSV headers); src/tests/test_data_schema.py enforces that the labeler
and scorer stay consistent with this module.
"""

# --- CSV columns authored by the labeler (this order = CSV column order) ---
FILENAME    = "filename"
TIMESTAMP   = "timestamp"
COUNTRY     = "country"
PANEL_AGE   = "panel_age"
PHASE_TYPE  = "phase_type"
HAS_RCD     = "has_rcd"
HAS_RCD_SI  = "has_rcd_si"
HAS_IGA     = "has_iga"
IGA_AMP     = "iga_amp"
HAS_OVP     = "has_ovp"
MCB_VALUES  = "mcb_values"
NUM_MCBS    = "num_mcbs"
COMMENTS    = "comments"
NUMBER_RCD  = "number_rcd"

CSV_FIELDNAMES = [
    FILENAME, TIMESTAMP, COUNTRY, PANEL_AGE, PHASE_TYPE, HAS_RCD, HAS_RCD_SI,
    HAS_IGA, IGA_AMP, HAS_OVP, MCB_VALUES, NUM_MCBS, COMMENTS, NUMBER_RCD,
]

# --- Derived / runtime fields used by scoring but NOT written by the labeler ---
# rcd_load_ratio is computed downstream from number_rcd / num_mcbs; rcd_test_result
# arrives from the upload form in app/main.py (the user's monthly RCD test answer).
RCD_LOAD_RATIO  = "rcd_load_ratio"
RCD_TEST_RESULT = "rcd_test_result"

DERIVED_FIELDS = [RCD_LOAD_RATIO, RCD_TEST_RESULT]

# --- Fields consumed by src/scoring.py (used by the enforcing test) ---
SCORING_INPUT_FIELDS = [
    HAS_OVP, HAS_RCD_SI, PANEL_AGE, HAS_RCD, COMMENTS,
    RCD_LOAD_RATIO, RCD_TEST_RESULT,
]
