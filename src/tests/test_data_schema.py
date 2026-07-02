"""
Guards the shared panel-attribute CSV schema (src/data_schema.py) so the labeler
(writer) and scoring (reader) cannot silently drift apart.
"""
import importlib

import pytest

from src import data_schema as S


def test_csv_fieldnames_unique():
    assert len(S.CSV_FIELDNAMES) == len(set(S.CSV_FIELDNAMES)), "duplicate CSV columns"


def test_every_csv_column_has_a_named_constant():
    constants = {v for k, v in vars(S).items() if isinstance(v, str) and not k.startswith("_")}
    for col in S.CSV_FIELDNAMES:
        assert col in constants, f"CSV column {col!r} has no named constant in data_schema"


def test_scoring_inputs_are_known_fields():
    """Every field scoring reads must be a real CSV column or a declared derived field."""
    known = set(S.CSV_FIELDNAMES) | set(S.DERIVED_FIELDS)
    for f in S.SCORING_INPUT_FIELDS:
        assert f in known, f"scoring input {f!r} is not defined in the schema"


def test_labeler_uses_shared_schema():
    """The labeler's CSV header must come from the shared schema (no local literal list)."""
    try:
        labeler = importlib.import_module("src.tools.labeler")
    except Exception as e:  # e.g. tkinter unavailable in a headless runner
        pytest.skip(f"labeler import unavailable: {e}")
    assert labeler.FIELDNAMES == S.CSV_FIELDNAMES
