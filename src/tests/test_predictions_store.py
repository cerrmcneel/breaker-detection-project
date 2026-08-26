import json
import sqlite3
from unittest.mock import patch

import pytest

from src.storage.predictions_store import (
    init_db,
    record_correction,
    record_prediction,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "predictions.db")


def test_init_db_creates_both_tables(db_path):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        conn.close()
    assert {"predictions", "corrections"} <= names


def test_init_db_is_idempotent(db_path):
    init_db(db_path)
    init_db(db_path)  # must not raise or wipe existing data
    row_id = record_prediction(
        source_endpoint="/upload/", predictions=[], inference_ok=True, db_path=db_path,
    )
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    finally:
        conn.close()
    assert row_id is not None
    assert count == 1


def test_record_prediction_round_trips_raw_output(db_path):
    init_db(db_path)
    preds = [
        {"box": [1, 2, 3, 4], "class": "MCB", "conf": 0.91, "ocr_text": "C16"},
        {"box": [5, 6, 7, 8], "class": "RCD", "conf": 0.87, "ocr_text": "30MA"},
    ]
    row_id = record_prediction(
        source_endpoint="/upload/",
        predictions=preds,
        inference_ok=True,
        tracking_id="BKR-TEST1",
        image_hash="deadbeef",
        model_version="v1.0.0",
        country="ES",
        rcd_test_result="Responsive",
        inference_engine="K3s-GPU-Cluster-Pipeline",
        computed_score=87,
        computed_feedback="<div>report</div>",
        db_path=db_path,
    )
    assert row_id is not None

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT tracking_id, image_hash, model_version, country, inference_ok, "
            "inference_engine, raw_output, computed_score FROM predictions WHERE id=?",
            (row_id,),
        ).fetchone()
    finally:
        conn.close()

    tracking_id, image_hash, model_version, country, inference_ok, engine, raw_output, score = row
    assert tracking_id == "BKR-TEST1"
    assert image_hash == "deadbeef"
    assert model_version == "v1.0.0"
    assert country == "ES"
    assert inference_ok == 1
    assert engine == "K3s-GPU-Cluster-Pipeline"
    assert score == 87
    assert json.loads(raw_output) == preds  # the actual detection output survives intact


def test_record_prediction_accepts_all_nullable_fields_omitted(db_path):
    init_db(db_path)
    row_id = record_prediction(
        source_endpoint="/predict/",
        predictions=[],
        inference_ok=False,
        db_path=db_path,
    )
    assert row_id is not None


def test_record_prediction_failure_is_non_fatal(db_path):
    init_db(db_path)
    with patch("src.storage.predictions_store.sqlite3.connect", side_effect=sqlite3.OperationalError("locked")):
        result = record_prediction(
            source_endpoint="/upload/", predictions=[], inference_ok=True, db_path=db_path,
        )
    assert result is None  # must not raise -- this is a secondary audit trail


def test_record_correction_references_a_prediction(db_path):
    init_db(db_path)
    pred_id = record_prediction(
        source_endpoint="/upload/", predictions=[{"class": "MCB"}], inference_ok=True, db_path=db_path,
    )
    corr_id = record_correction(
        prediction_id=pred_id,
        corrected_payload={"class": "RCD"},
        source="active_learning_endpoint",
        note="test correction",
        db_path=db_path,
    )
    assert corr_id is not None

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT prediction_id, corrected_payload, source, note FROM corrections WHERE id=?",
            (corr_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == pred_id
    assert json.loads(row[1]) == {"class": "RCD"}
    assert row[2] == "active_learning_endpoint"
    assert row[3] == "test correction"


def test_record_correction_failure_is_non_fatal(db_path):
    init_db(db_path)
    with patch("src.storage.predictions_store.sqlite3.connect", side_effect=sqlite3.OperationalError("locked")):
        result = record_correction(
            prediction_id=1, corrected_payload={}, source="test", db_path=db_path,
        )
    assert result is None


def test_wal_mode_is_enabled(db_path):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


# --- find_prediction_id_by_tracking_id (tracking_id linking, 2026-08-26) ---------------

def test_find_prediction_id_by_tracking_id_returns_the_match(db_path):
    init_db(db_path)
    pred_id = record_prediction(
        source_endpoint="/predict/",
        predictions=[{"class": "MCB"}],
        inference_ok=True,
        tracking_id="BKR-ABCDE",
        db_path=db_path,
    )
    from src.storage.predictions_store import find_prediction_id_by_tracking_id
    found = find_prediction_id_by_tracking_id("BKR-ABCDE", db_path=db_path)
    assert found == pred_id


def test_find_prediction_id_by_tracking_id_returns_none_when_not_found(db_path):
    init_db(db_path)
    from src.storage.predictions_store import find_prediction_id_by_tracking_id
    assert find_prediction_id_by_tracking_id("BKR-NOPE1", db_path=db_path) is None


def test_find_prediction_id_by_tracking_id_returns_the_most_recent_on_collision(db_path):
    init_db(db_path)
    record_prediction(
        source_endpoint="/predict/", predictions=[], inference_ok=True,
        tracking_id="BKR-DUP01", db_path=db_path,
    )
    newest_id = record_prediction(
        source_endpoint="/predict/", predictions=[], inference_ok=True,
        tracking_id="BKR-DUP01", db_path=db_path,
    )
    from src.storage.predictions_store import find_prediction_id_by_tracking_id
    assert find_prediction_id_by_tracking_id("BKR-DUP01", db_path=db_path) == newest_id


def test_find_prediction_id_by_tracking_id_failure_is_non_fatal(db_path):
    init_db(db_path)
    from src.storage.predictions_store import find_prediction_id_by_tracking_id
    with patch("src.storage.predictions_store.sqlite3.connect", side_effect=sqlite3.OperationalError("locked")):
        assert find_prediction_id_by_tracking_id("BKR-ABCDE", db_path=db_path) is None
