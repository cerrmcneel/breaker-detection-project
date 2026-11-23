import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import app.main as main_module
from src.image_hash import image_content_hash
from src.storage.predictions_store import init_db, record_correction, record_prediction
from src.tools.delete_by_tracking_id import main as delete_main


def _make_jpeg_bytes(seed_value: int, width: int = 40, height: int = 32) -> bytes:
    """Create a deterministic, decodable JPEG whose pixels depend on ``seed_value``."""
    img = np.full((height, width, 3), seed_value % 256, dtype=np.uint8)
    # Draw a distinct rectangle so JPEG compression preserves strong pixel differences
    img[4:height - 4, 4:width - 4] = (seed_value * 53 + 17) % 256
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _build_fake_data_dir(tmp_path: Path, db_override: Path | None = None) -> dict:
    """Populate an isolated fake ``data/`` tree under ``tmp_path`` (never real ``data/``)."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "images" / "raw_uploads"
    train_img_dir = data_dir / "dataset" / "train" / "images"
    train_lbl_dir = data_dir / "dataset" / "train" / "labels"
    val_img_dir = data_dir / "dataset" / "val" / "images"
    val_lbl_dir = data_dir / "dataset" / "val" / "labels"
    vault_dir = data_dir / "active_learning"

    for directory in (
        raw_dir,
        train_img_dir,
        train_lbl_dir,
        val_img_dir,
        val_lbl_dir,
        vault_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    target_id = "BKR-DEL01"
    other_id = "BKR-KEEP9"

    target_bytes = _make_jpeg_bytes(10)
    # Re-exported copy (trailing comment bytes): different raw bytes, identical pixel hash
    target_reexport_bytes = target_bytes + b"\xff\xfe" + b"EXIF-REWRITE" * 4
    other_bytes = _make_jpeg_bytes(200)
    trap_bytes = _make_jpeg_bytes(111)

    target_hash = image_content_hash(target_bytes)
    other_hash = image_content_hash(other_bytes)
    trap_hash = image_content_hash(trap_bytes)
    assert target_hash is not None
    assert other_hash is not None
    assert trap_hash is not None
    assert len({target_hash, other_hash, trap_hash}) == 3
    assert image_content_hash(target_reexport_bytes) == target_hash

    # 1 & 2. Kept raw upload + unrelated raw upload + upload_log.json
    target_saved_name = "ES_target-uuid-0001.jpg"
    target_original_name = "IMG_20261001_143022_my_panel.jpg"
    other_saved_name = "ES_other-uuid-9999.jpg"

    target_raw = raw_dir / target_saved_name
    other_raw = raw_dir / other_saved_name
    target_raw.write_bytes(target_bytes)
    other_raw.write_bytes(other_bytes)

    upload_log_path = data_dir / "images" / "upload_log.json"
    upload_log_entries = [
        {
            "timestamp": "2026-10-01T14:30:22",
            "original_filename": target_original_name,
            "saved_filename": target_saved_name,
            "training_consent": True,
            "photo_stored": True,
            "duplicate": False,
            "country": "ES",
            "hash": target_hash,
            "rcd_test_result": "Responsive",
            "tracking_id": target_id,
            "inference_ok": True,
            "inference_engine": "Local-GPU-RTX3060",
            "manual_score": 90,
            "manual_feedback": "<div>ok</div>",
        },
        {
            "timestamp": "2026-10-02T09:00:00",
            "original_filename": "other_panel.jpg",
            "saved_filename": other_saved_name,
            "training_consent": True,
            "photo_stored": True,
            "duplicate": False,
            "country": "ES",
            "hash": other_hash,
            "rcd_test_result": "Responsive",
            "tracking_id": other_id,
            "inference_ok": True,
            "inference_engine": "Local-GPU-RTX3060",
            "manual_score": 85,
            "manual_feedback": "<div>other</div>",
        },
    ]
    upload_log_path.write_text(json.dumps(upload_log_entries, indent=4), encoding="utf-8")

    # 3. Dataset copies (train + val) matched by pixel hash, plus an unrelated photo,
    #    plus a filename-collision trap (same filename as target_saved_name, DIFFERENT pixels!)
    target_train_img = train_img_dir / "imported_panel_001.jpg"
    target_train_lbl = train_lbl_dir / "imported_panel_001.txt"
    target_train_img.write_bytes(target_reexport_bytes)
    target_train_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    target_val_img = val_img_dir / "val_panel_copy.jpg"
    target_val_lbl = val_lbl_dir / "val_panel_copy.txt"
    target_val_img.write_bytes(target_bytes)
    target_val_lbl.write_text("1 0.4 0.4 0.1 0.1\n", encoding="utf-8")

    other_train_img = train_img_dir / "unrelated_panel.jpg"
    other_train_lbl = train_lbl_dir / "unrelated_panel.txt"
    other_train_img.write_bytes(other_bytes)
    other_train_lbl.write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    filename_trap_img = train_img_dir / target_saved_name
    filename_trap_lbl = train_lbl_dir / f"{Path(target_saved_name).stem}.txt"
    filename_trap_img.write_bytes(trap_bytes)
    filename_trap_lbl.write_text("2 0.3 0.3 0.2 0.2\n", encoding="utf-8")

    # 4. Active learning vault pairs (target with tracking_id in JSON + unrelated pair)
    target_vault_img = vault_dir / "correction_1700000001_target.jpg"
    target_vault_json = vault_dir / "correction_1700000001_target.json"
    target_vault_img.write_bytes(target_bytes)
    target_vault_json.write_text(
        json.dumps(
            {
                "tracking_id": target_id,
                "training_consent": True,
                "timestamp": "2026-10-01T15:00:00+00:00",
                "annotations": [{"class": "MCB", "box": [1, 2, 10, 12]}],
            }
        ),
        encoding="utf-8",
    )

    other_vault_img = vault_dir / "correction_1700000002_other.jpg"
    other_vault_json = vault_dir / "correction_1700000002_other.json"
    other_vault_img.write_bytes(other_bytes)
    other_vault_json.write_text(
        json.dumps(
            {
                "tracking_id": other_id,
                "training_consent": True,
                "timestamp": "2026-10-02T10:00:00+00:00",
                "annotations": [{"class": "RCD", "box": [2, 3, 12, 14]}],
            }
        ),
        encoding="utf-8",
    )

    # 5. SQLite predictions DB initialized via predictions_store.init_db
    db_path = db_override if db_override is not None else data_dir / "predictions.db"
    init_db(str(db_path))

    target_pred_id = record_prediction(
        source_endpoint="/upload/",
        predictions=[{"class": "MCB", "conf": 0.9}],
        inference_ok=True,
        tracking_id=target_id,
        image_hash=target_hash,
        country="ES",
        rcd_test_result="Responsive",
        computed_score=90,
        db_path=str(db_path),
    )
    assert target_pred_id is not None
    target_corr_id = record_correction(
        prediction_id=target_pred_id,
        corrected_payload={"annotations": [{"class": "MCB"}]},
        source="active_learning_endpoint",
        note="correction_1700000001_target",
        db_path=str(db_path),
    )
    assert target_corr_id is not None

    other_pred_id = record_prediction(
        source_endpoint="/upload/",
        predictions=[{"class": "RCD", "conf": 0.85}],
        inference_ok=True,
        tracking_id=other_id,
        image_hash=other_hash,
        country="ES",
        rcd_test_result="Responsive",
        computed_score=85,
        db_path=str(db_path),
    )
    assert other_pred_id is not None
    other_corr_id = record_correction(
        prediction_id=other_pred_id,
        corrected_payload={"annotations": [{"class": "RCD"}]},
        source="active_learning_endpoint",
        note="correction_1700000002_other",
        db_path=str(db_path),
    )
    assert other_corr_id is not None

    return {
        "data_dir": data_dir,
        "db_path": db_path,
        "upload_log_path": upload_log_path,
        "deletions_log_path": data_dir / "deletions_log.json",
        "target_id": target_id,
        "other_id": other_id,
        "target_saved_name": target_saved_name,
        "target_original_name": target_original_name,
        "target_hash": target_hash,
        "target_files": [
            target_raw,
            target_train_img,
            target_train_lbl,
            target_val_img,
            target_val_lbl,
            target_vault_img,
            target_vault_json,
        ],
        "unrelated_files": [
            other_raw,
            other_train_img,
            other_train_lbl,
            filename_trap_img,
            filename_trap_lbl,
            other_vault_img,
            other_vault_json,
        ],
    }


def _db_counts(db_path: Path) -> tuple[int, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        corrs = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
        return preds, corrs
    finally:
        conn.close()


# --- Extracted image_content_hash equivalence proof ------------------------------------


def test_extracted_image_content_hash_matches_app_main_and_reference():
    assert image_content_hash is main_module.image_content_hash

    jpeg = _make_jpeg_bytes(42, width=48, height=36)
    reexported = jpeg + b"\xff\xfe" + b"TRAILING-EXIF" * 4

    # Reference formula matching app/main.py's original implementation
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    assert decoded is not None
    arr = np.ascontiguousarray(decoded)
    expected = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()

    assert image_content_hash(jpeg) == expected
    assert image_content_hash(reexported) == expected
    assert image_content_hash(_make_jpeg_bytes(99, width=48, height=36)) != expected
    assert image_content_hash(b"not an image") is None


# --- Dry run ---------------------------------------------------------------------------


def test_dry_run_changes_nothing_and_reports_matches(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    original_upload_log = env["upload_log_path"].read_text(encoding="utf-8")

    rc = delete_main(
        [
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
        ]
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert env["target_id"] in out
    for target_file in env["target_files"]:
        assert str(target_file) in out
        assert target_file.exists(), f"dry run unexpectedly deleted {target_file}"
    for keep_file in env["unrelated_files"]:
        assert keep_file.exists()

    # DB rows untouched
    assert _db_counts(env["db_path"]) == (2, 2)
    # upload_log.json untouched
    assert env["upload_log_path"].read_text(encoding="utf-8") == original_upload_log
    # deletions_log.json must not be created on dry run
    assert not env["deletions_log_path"].exists()


# --- Apply -----------------------------------------------------------------------------


def test_apply_deletes_exact_targets_leaves_unrelated_and_logs_safely(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)

    rc = delete_main(
        [
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
            "--apply",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "APPLIED" in out

    # Every targeted file is removed
    for target_file in env["target_files"]:
        assert not target_file.exists(), f"expected {target_file} to be deleted"

    # Every unrelated file (including the same-filename / different-pixels trap!) survives
    for keep_file in env["unrelated_files"]:
        assert keep_file.exists(), f"unrelated file {keep_file} was wrongly deleted"

    # SQLite DB: target prediction + correction deleted; unrelated rows survive
    assert _db_counts(env["db_path"]) == (1, 1)
    conn = sqlite3.connect(str(env["db_path"]))
    try:
        remaining_ids = [
            row[0] for row in conn.execute("SELECT tracking_id FROM predictions").fetchall()
        ]
    finally:
        conn.close()
    assert remaining_ids == [env["other_id"]]

    # upload_log.json: both entries survive; target entry has no filename and has deleted_on_request
    upload_entries = json.loads(env["upload_log_path"].read_text(encoding="utf-8"))
    assert len(upload_entries) == 2
    by_id = {entry["tracking_id"]: entry for entry in upload_entries}

    deleted_entry = by_id[env["target_id"]]
    assert deleted_entry["saved_filename"] is None
    assert deleted_entry["original_filename"] is None
    assert isinstance(deleted_entry.get("deleted_on_request"), str)
    assert deleted_entry["deleted_on_request"]
    assert deleted_entry["training_consent"] is True  # consent record preserved

    kept_entry = by_id[env["other_id"]]
    assert kept_entry["saved_filename"] == "ES_other-uuid-9999.jpg"
    assert "deleted_on_request" not in kept_entry

    # deletions_log.json: counts per location, NO file names / paths / hashes
    assert env["deletions_log_path"].exists()
    raw_deletions_text = env["deletions_log_path"].read_text(encoding="utf-8")
    deletions_log = json.loads(raw_deletions_text)
    assert len(deletions_log) == 1

    record = deletions_log[0]
    assert record["tracking_id"] == env["target_id"]
    assert isinstance(record["timestamp"], str) and record["timestamp"]
    assert record["counts"] == {
        "raw_uploads": 1,
        "dataset_images": 2,
        "dataset_labels": 2,
        "active_learning": 2,
        "upload_log_updated": 1,
        "db_predictions": 1,
        "db_corrections": 1,
    }

    # Verify zero filenames, paths, or hashes leaked into deletions_log.json
    forbidden_substrings = [
        env["target_saved_name"],
        env["target_original_name"],
        env["target_hash"],
        "imported_panel_001",
        "val_panel_copy",
        "correction_1700000001_target",
        ".jpg",
        ".json",
        ".txt",
    ]
    for forbidden in forbidden_substrings:
        assert forbidden not in raw_deletions_text, f"deletions_log leaked {forbidden!r}"


# --- Custom --db path & vault matched via DB correction note (no tracking_id in JSON) ---


def test_apply_respects_db_flag_and_deletes_db_linked_vault_pair(tmp_path):
    custom_db = tmp_path / "elsewhere" / "custom_predictions.db"
    env = _build_fake_data_dir(tmp_path, db_override=custom_db)

    # Add a second vault pair whose JSON lacks tracking_id (matching /active-learning/save)
    # and whose image is a distinct crop, linked ONLY through corrections.note.
    vault_dir = env["data_dir"] / "active_learning"
    linked_stem = "correction_1700000099_dblinked"
    linked_img = vault_dir / f"{linked_stem}.jpg"
    linked_json = vault_dir / f"{linked_stem}.json"
    linked_img.write_bytes(_make_jpeg_bytes(77))
    linked_json.write_text(json.dumps({"training_consent": True, "annotations": []}), encoding="utf-8")

    conn = sqlite3.connect(str(custom_db))
    try:
        pred_id = conn.execute(
            "SELECT id FROM predictions WHERE tracking_id = ?", (env["target_id"],)
        ).fetchone()[0]
    finally:
        conn.close()

    record_correction(
        prediction_id=pred_id,
        corrected_payload={"annotations": []},
        source="active_learning_endpoint",
        note=linked_stem,
        db_path=str(custom_db),
    )

    rc = delete_main(
        [
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
            "--db",
            str(custom_db),
            "--apply",
        ]
    )
    assert rc == 0
    assert not linked_img.exists()
    assert not linked_json.exists()
    assert _db_counts(custom_db) == (1, 1)


# --- Unknown tracking ID ---------------------------------------------------------------


@pytest.mark.parametrize("extra_args", [[], ["--apply"]])
def test_unknown_tracking_id_exits_nonzero_and_changes_nothing(tmp_path, capsys, extra_args):
    env = _build_fake_data_dir(tmp_path)
    original_upload_log = env["upload_log_path"].read_text(encoding="utf-8")

    rc = delete_main(
        [
            "--tracking-id",
            "BKR-NOPE0",
            "--data-dir",
            str(env["data_dir"]),
            *extra_args,
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "BKR-NOPE0" in err

    for path in env["target_files"] + env["unrelated_files"]:
        assert path.exists()
    assert _db_counts(env["db_path"]) == (2, 2)
    assert env["upload_log_path"].read_text(encoding="utf-8") == original_upload_log
    assert not env["deletions_log_path"].exists()


# --- Appending to deletions_log.json, legacy vault JSON, and module CLI ----------------


def test_deletions_log_appends_across_multiple_requests(tmp_path):
    env = _build_fake_data_dir(tmp_path)

    assert delete_main(["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]) == 0
    assert delete_main(["--tracking-id", env["other_id"], "--data-dir", str(env["data_dir"]), "--apply"]) == 0

    log = json.loads(env["deletions_log_path"].read_text(encoding="utf-8"))
    assert [entry["tracking_id"] for entry in log] == [env["target_id"], env["other_id"]]
    assert _db_counts(env["db_path"]) == (0, 0)


def test_vault_legacy_list_sidecar_matched_by_pixel_hash(tmp_path):
    env = _build_fake_data_dir(tmp_path)
    vault_dir = env["data_dir"] / "active_learning"
    legacy_img = vault_dir / "correction_legacy_001.jpg"
    legacy_json = vault_dir / "correction_legacy_001.json"
    # Same decoded pixels as target, bare list JSON (pre-T18 shape, no tracking_id key)
    legacy_img.write_bytes(_make_jpeg_bytes(10))
    legacy_json.write_text(json.dumps([{"class": "MCB", "box": [0, 0, 5, 5]}]), encoding="utf-8")

    rc = delete_main(
        ["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc == 0
    assert not legacy_img.exists()
    assert not legacy_json.exists()


def test_cli_module_invocation_and_lightweight_import(tmp_path):
    env = _build_fake_data_dir(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tools.delete_by_tracking_id",
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "DRY RUN" in proc.stdout
    assert env["target_id"] in proc.stdout

    # Confirm importing the CLI module does not drag in FastAPI / app.main
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, src.tools.delete_by_tracking_id; "
            "assert 'app.main' not in sys.modules; "
            "assert 'fastapi' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_check.returncode == 0, import_check.stderr

