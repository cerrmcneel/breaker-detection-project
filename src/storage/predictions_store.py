"""
Temporal audit trail for PanelSafe predictions.

WHY THIS EXISTS (audited 2026-08-16 against an "ontological framework" pitch for
a much bigger, multi-team enterprise data platform -- most of that pitch was
rejected as premature for a one-developer project; see
directives/production_hardening_roadmap.md under "Data Architecture: Selective
Adoption of the 'Ontological Framework'" for the full reasoning). The one piece
worth keeping: every hard bug this cycle -- the classes.txt/data.yaml label
permutation, the OCR "SI"/"RESI9" collision, HMM emission recalibration -- was a
"what did this classification mean at time T, and how did it drift" question,
answered by hand-writing a one-off CSV dump script each time, because nothing
queryable existed. app/main.py's upload_log.json is a flat, denormalized JSON
array mixing raw upload facts with computed grading output, and once a request
completes its RAW detection output (boxes, classes, confidences, OCR text) is
gone entirely -- there is no way to re-audit a past classification without
re-running inference against the saved image.

This module does not replace upload_log.json. It is additive: a second,
queryable record of the same events, split into what the model actually said
(raw_output -- append-only, never mutated) versus what was derived from it
(computed_score/computed_feedback -- recomputable later without re-inference).

DELIBERATE SCOPE BOUNDARY: a `corrections` table is defined here, ready for a
future active-learning correction to reference a prediction by id, but nothing
currently WRITES to it. /active-learning/save's request contract does not carry
a tracking_id or prediction reference today (checked 2026-08-16), so there is no
reliable way to link a correction back to its parent prediction without also
changing that endpoint's contract -- a separate, larger decision this module
should not make silently as a side effect of "add a database."

To query this DB directly (no Python API is provided for reads on purpose --
that is the whole point of using a real, inspectable store instead of another
one-off script):
    sqlite3 data/predictions.db "SELECT model_version, COUNT(*) FROM predictions GROUP BY model_version;"
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PREDICTIONS_DB_PATH = os.getenv("PREDICTIONS_DB_PATH", "data/predictions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,      -- ISO 8601 UTC
    source_endpoint  TEXT NOT NULL,      -- '/predict/' or '/upload/'
    tracking_id      TEXT,
    image_hash       TEXT,
    model_version    TEXT,               -- best-effort; see record_prediction() docstring
    country          TEXT,
    rcd_test_result  TEXT,
    inference_ok     INTEGER NOT NULL,   -- 0/1
    inference_engine TEXT,
    raw_output       TEXT,               -- JSON: the actual detection list, append-only
    computed_score   INTEGER,
    computed_feedback TEXT               -- HTML report; NULL when inference_ok=0
);
CREATE INDEX IF NOT EXISTS idx_predictions_image_hash    ON predictions(image_hash);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at    ON predictions(created_at);

CREATE TABLE IF NOT EXISTS corrections (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id     INTEGER REFERENCES predictions(id),
    corrected_at      TEXT NOT NULL,
    corrected_payload TEXT NOT NULL,     -- JSON
    source            TEXT NOT NULL,     -- e.g. 'active_learning_endpoint'
    note              TEXT
);
"""


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    db_path = db_path or PREDICTIONS_DB_PATH
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # WAL lets concurrent readers (a query while the gateway is writing) proceed
    # without blocking, and lets sqlite serialize writers itself rather than the
    # app needing its own lock -- unlike upload_log.json, which reimplements this
    # with an asyncio.Lock because a flat file can't do it for you.
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Create the schema if it doesn't exist. Safe to call on every startup."""
    conn = _connect(db_path or PREDICTIONS_DB_PATH)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def record_prediction(
    *,
    source_endpoint: str,
    predictions: List[Dict[str, Any]],
    inference_ok: bool,
    tracking_id: Optional[str] = None,
    image_hash: Optional[str] = None,
    model_version: Optional[str] = None,
    country: Optional[str] = None,
    rcd_test_result: Optional[str] = None,
    inference_engine: Optional[str] = None,
    computed_score: Optional[int] = None,
    computed_feedback: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[int]:
    """Blocking write -- call via starlette.concurrency.run_in_threadpool from an
    async handler, the same pattern already used for upload_log.json.

    `model_version` is best-effort: the gateway forwards inference over HTTP to a
    separate host (see app/main.py's INFERENCE_URL/FAILOVER_URL), and the
    inference response does not currently include a version tag. Callers should
    pass whatever their local pipeline_config.json claims, which reflects what
    the gateway believes is active, not a verified guarantee of what actually
    served this specific request.

    Returns the new row's id, or None if the write failed. Never raises -- this
    is a secondary audit trail, not the primary source of truth (upload_log.json
    remains that during the migration period), so a failure here must not fail
    the HTTP request it's describing.
    """
    try:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO predictions (
                    created_at, source_endpoint, tracking_id, image_hash, model_version,
                    country, rcd_test_result, inference_ok, inference_engine,
                    raw_output, computed_score, computed_feedback
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    source_endpoint,
                    tracking_id,
                    image_hash,
                    model_version,
                    country,
                    rcd_test_result,
                    1 if inference_ok else 0,
                    inference_engine,
                    json.dumps(predictions, ensure_ascii=False),
                    computed_score,
                    computed_feedback,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"predictions_store: failed to record prediction (non-fatal): {e}")
        return None


def record_correction(
    *,
    prediction_id: int,
    corrected_payload: Dict[str, Any],
    source: str,
    note: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[int]:
    """Not yet called from any endpoint -- see the scope-boundary note at the top
    of this file. Provided so wiring a future correction flow is additive rather
    than requiring a schema change."""
    try:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO corrections (prediction_id, corrected_at, corrected_payload, source, note)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    prediction_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(corrected_payload, ensure_ascii=False),
                    source,
                    note,
                ),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"predictions_store: failed to record correction (non-fatal): {e}")
        return None
