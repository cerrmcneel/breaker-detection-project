import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from src.image_hash import image_content_hash
from src.storage.predictions_store import init_db, record_correction, record_prediction
from src.tools.delete_by_tracking_id import (
    apply_deletion_plan,
    build_deletion_plan,
)
from src.tools.delete_by_tracking_id import main as delete_main


@pytest.fixture(autouse=True)
def _isolate_storage_env_vars(monkeypatch):
    """Keep host/container env vars from leaking into isolated tmp_path tests."""
    monkeypatch.delenv("PREDICTIONS_DB_PATH", raising=False)
    monkeypatch.delenv("UPLOAD_DIR", raising=False)


def _make_jpeg_bytes(seed_value: int, width: int = 40, height: int = 32) -> bytes:
    """Create a deterministic, decodable JPEG whose pixels depend on ``seed_value``."""
    img = np.full((height, width, 3), seed_value % 256, dtype=np.uint8)
    # Draw a distinct rectangle so JPEG compression preserves strong pixel differences
    img[4:height - 4, 4:width - 4] = (seed_value * 53 + 17) % 256
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _build_fake_data_dir(tmp_path: Path, db_override: Optional[Path] = None) -> Dict[str, Any]:
    """Populate an isolated fake ``data/`` tree under ``tmp_path / "data"``."""
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
        "target_bytes": target_bytes,
        "target_hash": target_hash,
        "other_bytes": other_bytes,
        "other_hash": other_hash,
        "target_pred_id": target_pred_id,
        "other_pred_id": other_pred_id,
        "target_raw": target_raw,
        "target_train_img": target_train_img,
        "target_train_lbl": target_train_lbl,
        "target_val_img": target_val_img,
        "target_val_lbl": target_val_lbl,
        "target_vault_img": target_vault_img,
        "target_vault_json": target_vault_json,
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


def _db_counts(db_path: Path) -> Tuple[int, int]:
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
    assert "WARNING" not in out
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
    assert env["target_id"] in out

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


# --- Custom --db path & vault matched ONLY via DB correction note (no tracking_id in JSON) ---


def test_apply_respects_db_flag_and_deletes_db_linked_vault_pair(tmp_path, capsys):
    custom_db = tmp_path / "elsewhere" / "custom_predictions.db"
    env = _build_fake_data_dir(tmp_path, db_override=custom_db)

    # Real pre-fix /active-learning/save shape: JSON has NO tracking_id key, and
    # image bytes are re-encoded / distinct pixels so pixel-hash matching cannot find it --
    # it can ONLY be discovered via corrections.note in the DB.
    vault_dir = env["data_dir"] / "active_learning"
    linked_stem = "correction_1700000099_dblinked"
    linked_img = vault_dir / f"{linked_stem}.jpg"
    linked_json = vault_dir / f"{linked_stem}.json"
    linked_img.write_bytes(_make_jpeg_bytes(77))
    linked_json.write_text(
        json.dumps(
            {
                "training_consent": True,
                "timestamp": "2026-10-01T16:00:00+00:00",
                "annotations": [{"class": "RCD", "box": [1, 2, 3, 4]}],
            }
        ),
        encoding="utf-8",
    )

    record_correction(
        prediction_id=env["target_pred_id"],
        corrected_payload={"annotations": [{"class": "RCD", "box": [1, 2, 3, 4]}]},
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
    out = capsys.readouterr().out
    assert "APPLIED" in out
    assert not linked_img.exists()
    assert not linked_json.exists()
    assert _db_counts(custom_db) == (1, 1)


# --- Security 1a: unstored hashes & shared hashes --------------------------------------


def test_unstored_photo_hash_from_predict_or_duplicate_does_not_delete_other_users_files(
    tmp_path, capsys
):
    env = _build_fake_data_dir(tmp_path)

    # Another user runs the EXACT SAME photo pixels through /predict/ (no consent,
    # no upload_log entry, only predictions.image_hash).
    predict_only_id = "BKR-PRD01"
    record_prediction(
        source_endpoint="/predict/",
        predictions=[{"class": "MCB", "conf": 0.9}],
        inference_ok=True,
        tracking_id=predict_only_id,
        image_hash=env["target_hash"],
        db_path=str(env["db_path"]),
    )

    # A third user uploads the same photo with duplicate=True / photo_stored=False.
    dup_id = "BKR-DUP01"
    upload_entries = json.loads(env["upload_log_path"].read_text(encoding="utf-8"))
    upload_entries.append(
        {
            "timestamp": "2026-10-03T11:00:00",
            "original_filename": None,
            "saved_filename": None,
            "training_consent": True,
            "photo_stored": False,
            "duplicate": True,
            "country": "ES",
            "hash": env["target_hash"],
            "tracking_id": dup_id,
        }
    )
    env["upload_log_path"].write_text(json.dumps(upload_entries, indent=4), encoding="utf-8")
    record_prediction(
        source_endpoint="/upload/",
        predictions=[{"class": "MCB", "conf": 0.9}],
        inference_ok=True,
        tracking_id=dup_id,
        image_hash=env["target_hash"],
        db_path=str(env["db_path"]),
    )

    # Deleting the /predict/-only ID must delete ONLY its DB row -- never the original uploader's files.
    rc = delete_main(
        [
            "--tracking-id",
            predict_only_id,
            "--data-dir",
            str(env["data_dir"]),
            "--apply",
            "--include-shared",
        ]
    )
    assert rc == 0
    assert "APPLIED" in capsys.readouterr().out

    # Deleting the photo_stored=False / duplicate=True ID must ALSO never touch the original uploader's files.
    rc = delete_main(
        [
            "--tracking-id",
            dup_id,
            "--data-dir",
            str(env["data_dir"]),
            "--apply",
            "--include-shared",
        ]
    )
    assert rc == 0
    assert "APPLIED" in capsys.readouterr().out

    for owner_file in env["target_files"]:
        assert owner_file.exists(), f"owner's file {owner_file} was deleted by non-owner request!"
    assert _db_counts(env["db_path"]) == (2, 2)


def _add_shared_owner(env: Dict[str, Any], shared_id: str = "BKR-SHR02") -> Dict[str, Path]:
    """Add a second stored record claiming the exact same pixel hash as target_id."""
    raw_dir = env["data_dir"] / "images" / "raw_uploads"
    vault_dir = env["data_dir"] / "active_learning"

    shared_raw = raw_dir / "ES_shared-owner-0002.jpg"
    shared_raw.write_bytes(env["target_bytes"])

    shared_vault_img = vault_dir / "correction_1700000055_shared.jpg"
    shared_vault_json = vault_dir / "correction_1700000055_shared.json"
    shared_vault_img.write_bytes(env["target_bytes"])
    shared_vault_json.write_text(
        json.dumps(
            {
                "tracking_id": shared_id,
                "training_consent": True,
                "timestamp": "2026-10-04T12:00:00+00:00",
                "annotations": [{"class": "MCB", "box": [1, 1, 5, 5]}],
            }
        ),
        encoding="utf-8",
    )

    # Also add an unattributed legacy vault pair with the shared pixels
    legacy_img = vault_dir / "correction_legacy_shared.jpg"
    legacy_json = vault_dir / "correction_legacy_shared.json"
    legacy_img.write_bytes(env["target_bytes"])
    legacy_json.write_text(json.dumps([{"class": "MCB", "box": [0, 0, 5, 5]}]), encoding="utf-8")

    upload_entries = json.loads(env["upload_log_path"].read_text(encoding="utf-8"))
    upload_entries.append(
        {
            "timestamp": "2026-10-04T12:00:00",
            "original_filename": "shared_copy.jpg",
            "saved_filename": shared_raw.name,
            "training_consent": True,
            "photo_stored": True,
            "duplicate": False,
            "country": "ES",
            "hash": env["target_hash"],
            "tracking_id": shared_id,
        }
    )
    env["upload_log_path"].write_text(json.dumps(upload_entries, indent=4), encoding="utf-8")

    return {
        "shared_raw": shared_raw,
        "shared_vault_img": shared_vault_img,
        "shared_vault_json": shared_vault_json,
        "legacy_img": legacy_img,
        "legacy_json": legacy_json,
    }


def test_shared_hash_flagged_in_dry_run_and_skipped_without_include_shared(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    shared = _add_shared_owner(env, shared_id="BKR-SHR02")

    # Dry run flags the shared hash, names the other owner, and mentions --include-shared
    rc = delete_main(["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"])])
    assert rc == 0
    dry_out = capsys.readouterr().out
    assert "WARNING" in dry_out
    assert "BKR-SHR02" in dry_out
    assert "--include-shared" in dry_out
    assert str(env["target_train_img"]) in dry_out
    assert str(shared["legacy_img"]) in dry_out

    # --apply without --include-shared deletes target's directly-owned files/rows,
    # skips shared dataset/unattributed-vault files, and leaves BKR-SHR02 untouched.
    rc = delete_main(
        ["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc == 0
    apply_out = capsys.readouterr().out
    assert "APPLIED" in apply_out
    assert "shared files skipped" in apply_out
    assert "--include-shared" in apply_out

    # Target's directly-owned files are deleted
    assert not env["target_raw"].exists()
    assert not env["target_vault_img"].exists()
    assert not env["target_vault_json"].exists()
    assert _db_counts(env["db_path"]) == (1, 1)

    # Shared hash-matched dataset + unattributed legacy vault files survive
    assert env["target_train_img"].exists()
    assert env["target_train_lbl"].exists()
    assert env["target_val_img"].exists()
    assert env["target_val_lbl"].exists()
    assert shared["legacy_img"].exists()
    assert shared["legacy_json"].exists()

    # Other owner's directly-owned files survive
    assert shared["shared_raw"].exists()
    assert shared["shared_vault_img"].exists()
    assert shared["shared_vault_json"].exists()


def test_shared_hash_deleted_when_include_shared_is_passed(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    shared = _add_shared_owner(env, shared_id="BKR-SHR02")

    # Dry run with --include-shared still warns and states shared files WILL BE DELETED
    rc = delete_main(
        [
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
            "--include-shared",
        ]
    )
    assert rc == 0
    dry_out = capsys.readouterr().out
    assert "WARNING" in dry_out
    assert "BKR-SHR02" in dry_out
    assert "WILL BE DELETED" in dry_out

    # --apply --include-shared deletes shared dataset + unattributed legacy vault files,
    # while STILL never deleting BKR-SHR02's own raw upload or directly-owned vault pair.
    rc = delete_main(
        [
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
            "--apply",
            "--include-shared",
        ]
    )
    assert rc == 0
    apply_out = capsys.readouterr().out
    assert "APPLIED" in apply_out
    assert "shared files skipped" not in apply_out

    assert not env["target_raw"].exists()
    assert not env["target_train_img"].exists()
    assert not env["target_train_lbl"].exists()
    assert not env["target_val_img"].exists()
    assert not env["target_val_lbl"].exists()
    assert not shared["legacy_img"].exists()
    assert not shared["legacy_json"].exists()

    # BKR-SHR02's directly-owned files survive
    assert shared["shared_raw"].exists()
    assert shared["shared_vault_img"].exists()
    assert shared["shared_vault_json"].exists()


# --- Security 1c: tracking ID collision warning ----------------------------------------


def test_dry_run_warns_on_tracking_id_collision(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    upload_entries = json.loads(env["upload_log_path"].read_text(encoding="utf-8"))
    upload_entries.append(
        {
            "timestamp": "2026-10-05T08:00:00",
            "original_filename": "second_user.jpg",
            "saved_filename": None,
            "training_consent": False,
            "photo_stored": False,
            "duplicate": False,
            "country": "ES",
            "hash": None,
            "tracking_id": env["target_id"],
        }
    )
    env["upload_log_path"].write_text(json.dumps(upload_entries, indent=4), encoding="utf-8")

    rc = delete_main(["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"])])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "collision" in out.lower()


# --- /predict/-only ID (no upload_log entry) -------------------------------------------


def test_predict_only_id_without_upload_log_deletes_db_and_vault_pair(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    original_upload_log = env["upload_log_path"].read_text(encoding="utf-8")

    pred_only_id = "BKR-HITL7"
    vault_bytes = _make_jpeg_bytes(61)
    vault_hash = image_content_hash(vault_bytes)
    assert vault_hash is not None

    vault_dir = env["data_dir"] / "active_learning"
    stem = "correction_1700000123_hitl"
    vault_img = vault_dir / f"{stem}.jpg"
    vault_json = vault_dir / f"{stem}.json"
    vault_img.write_bytes(vault_bytes)
    # Real pre-fix shape: no tracking_id key in sidecar
    vault_json.write_text(
        json.dumps(
            {
                "training_consent": True,
                "timestamp": "2026-10-05T10:00:00+00:00",
                "annotations": [{"class": "MCB", "box": [1, 2, 10, 10]}],
            }
        ),
        encoding="utf-8",
    )

    # Dataset copy imported from this vault image
    ds_img = env["data_dir"] / "dataset" / "train" / "images" / f"{stem}.jpg"
    ds_lbl = env["data_dir"] / "dataset" / "train" / "labels" / f"{stem}.txt"
    ds_img.write_bytes(vault_bytes)
    ds_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    pred_id = record_prediction(
        source_endpoint="/predict/",
        predictions=[{"class": "MCB", "conf": 0.9}],
        inference_ok=True,
        tracking_id=pred_only_id,
        image_hash=vault_hash,
        db_path=str(env["db_path"]),
    )
    assert pred_id is not None
    record_correction(
        prediction_id=pred_id,
        corrected_payload={"annotations": [{"class": "MCB", "box": [1, 2, 10, 10]}]},
        source="active_learning_endpoint",
        note=stem,
        db_path=str(env["db_path"]),
    )

    rc = delete_main(
        ["--tracking-id", pred_only_id, "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc == 0
    assert "APPLIED" in capsys.readouterr().out

    assert not vault_img.exists()
    assert not vault_json.exists()
    assert not ds_img.exists()
    assert not ds_lbl.exists()
    assert _db_counts(env["db_path"]) == (2, 2)
    assert env["upload_log_path"].read_text(encoding="utf-8") == original_upload_log
    for untouched in env["target_files"] + env["unrelated_files"]:
        assert untouched.exists()


# --- /active-learning/save writes tracking_id into sidecar -----------------------------


def test_active_learning_save_writes_tracking_id_to_sidecar(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    client = TestClient(main_module.app)

    with patch("app.main.find_prediction_id_by_tracking_id", return_value=None):
        resp = client.post(
            "/active-learning/save",
            files={"file": ("panel.jpg", _make_jpeg_bytes(33), "image/jpeg")},
            data={
                "training_consent": "true",
                "annotations": json.dumps([{"class": "RCD", "box": [1, 2, 8, 8]}]),
                "tracking_id": "bkr-side1",
            },
        )
    assert resp.status_code == 200

    vault_dir = tmp_path / "data" / "active_learning"
    (sidecar_path,) = list(vault_dir.glob("correction_*.json"))
    (img_path,) = list(vault_dir.glob("correction_*.jpg"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["tracking_id"] == "BKR-SIDE1"

    # Even with NO predictions.db and NO upload_log.json, the deletion tool finds and deletes it!
    rc = delete_main(
        ["--tracking-id", "BKR-SIDE1", "--data-dir", str(tmp_path / "data"), "--apply"]
    )
    assert rc == 0
    assert "APPLIED" in capsys.readouterr().out
    assert not sidecar_path.exists()
    assert not img_path.exists()


# --- Environment variables: PREDICTIONS_DB_PATH and UPLOAD_DIR -------------------------


def test_env_vars_predictions_db_path_and_upload_dir_are_honoured(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    custom_raw_dir = tmp_path / "custom_uploads" / "raw_uploads"
    custom_log_path = tmp_path / "custom_uploads" / "upload_log.json"
    custom_db_path = tmp_path / "custom_db" / "prod_predictions.db"
    custom_raw_dir.mkdir(parents=True, exist_ok=True)
    custom_db_path.parent.mkdir(parents=True, exist_ok=True)

    img_bytes = _make_jpeg_bytes(88)
    img_hash = image_content_hash(img_bytes)
    assert img_hash is not None
    raw_file = custom_raw_dir / "ES_env-test.jpg"
    raw_file.write_bytes(img_bytes)
    custom_log_path.write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-10-06T09:00:00",
                    "original_filename": "env.jpg",
                    "saved_filename": raw_file.name,
                    "training_consent": True,
                    "photo_stored": True,
                    "duplicate": False,
                    "hash": img_hash,
                    "tracking_id": "BKR-ENV01",
                }
            ]
        ),
        encoding="utf-8",
    )
    init_db(str(custom_db_path))
    record_prediction(
        source_endpoint="/upload/",
        predictions=[{"class": "MCB"}],
        inference_ok=True,
        tracking_id="BKR-ENV01",
        image_hash=img_hash,
        db_path=str(custom_db_path),
    )

    monkeypatch.setenv("PREDICTIONS_DB_PATH", str(custom_db_path))
    monkeypatch.setenv("UPLOAD_DIR", str(custom_raw_dir))

    # No --db and no --data-dir passed: must pick up both env vars
    rc = delete_main(["--tracking-id", "BKR-ENV01", "--apply"])
    assert rc == 0
    assert "APPLIED" in capsys.readouterr().out
    assert not raw_file.exists()
    assert _db_counts(custom_db_path) == (0, 0)
    updated = json.loads(custom_log_path.read_text(encoding="utf-8"))
    assert updated[0]["saved_filename"] is None
    assert updated[0]["deleted_on_request"]


# --- Correctness: re-runs, lowercase IDs, late corrections, traversal, bad JSON, missing DB ---


def test_apply_twice_preserves_original_timestamp_and_does_not_duplicate_audit_log(
    tmp_path, capsys
):
    env = _build_fake_data_dir(tmp_path)

    rc1 = delete_main(
        ["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc1 == 0
    assert "APPLIED" in capsys.readouterr().out

    entries_after_first = json.loads(env["upload_log_path"].read_text(encoding="utf-8"))
    first_ts = next(
        e["deleted_on_request"] for e in entries_after_first if e["tracking_id"] == env["target_id"]
    )
    audit_after_first = json.loads(env["deletions_log_path"].read_text(encoding="utf-8"))
    assert len(audit_after_first) == 1

    # Second --apply must exit cleanly with "already deleted", keep original timestamp,
    # and NOT append another all-zero entry to deletions_log.json.
    rc2 = delete_main(
        ["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "already deleted" in out2.lower()

    entries_after_second = json.loads(env["upload_log_path"].read_text(encoding="utf-8"))
    second_ts = next(
        e["deleted_on_request"]
        for e in entries_after_second
        if e["tracking_id"] == env["target_id"]
    )
    assert second_ts == first_ts

    audit_after_second = json.loads(env["deletions_log_path"].read_text(encoding="utf-8"))
    assert len(audit_after_second) == 1


def test_lowercase_tracking_id_matches_uppercase_records(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)

    rc = delete_main(
        [
            "--tracking-id",
            env["target_id"].lower(),
            "--data-dir",
            str(env["data_dir"]),
            "--apply",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "APPLIED" in out
    assert env["target_id"] in out
    for target_file in env["target_files"]:
        assert not target_file.exists()
    assert _db_counts(env["db_path"]) == (1, 1)


def test_apply_deletes_correction_added_after_plan_was_built(tmp_path):
    env = _build_fake_data_dir(tmp_path)
    plan = build_deletion_plan(tracking_id=env["target_id"], data_dir=env["data_dir"])

    # Simulate a correction saved between dry-run/plan and --apply
    late_corr_id = record_correction(
        prediction_id=env["target_pred_id"],
        corrected_payload={"annotations": [{"class": "OVERSURGE"}]},
        source="active_learning_endpoint",
        note="late_correction",
        db_path=str(env["db_path"]),
    )
    assert late_corr_id is not None

    audit = apply_deletion_plan(plan)
    assert audit["counts"]["db_predictions"] == 1
    assert audit["counts"]["db_corrections"] == 2
    assert _db_counts(env["db_path"]) == (1, 1)


def test_path_traversal_saved_filename_is_ignored(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    outside_file = env["data_dir"] / "images" / "x"
    outside_file.write_bytes(_make_jpeg_bytes(55))

    entries = json.loads(env["upload_log_path"].read_text(encoding="utf-8"))
    entries.append(
        {
            "timestamp": "2026-10-07T10:00:00",
            "original_filename": "evil.jpg",
            "saved_filename": "../x",
            "training_consent": True,
            "photo_stored": True,
            "duplicate": False,
            "hash": None,
            "tracking_id": "BKR-TRAV1",
        }
    )
    env["upload_log_path"].write_text(json.dumps(entries, indent=4), encoding="utf-8")

    # Dry run must not list the traversal target
    rc_dry = delete_main(["--tracking-id", "BKR-TRAV1", "--data-dir", str(env["data_dir"])])
    assert rc_dry == 0
    assert str(outside_file) not in capsys.readouterr().out

    # Apply must leave ../x untouched while still redacting the log entry
    rc_apply = delete_main(
        ["--tracking-id", "BKR-TRAV1", "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc_apply == 0
    assert "APPLIED" in capsys.readouterr().out
    assert outside_file.exists()


@pytest.mark.parametrize("bad_content", ["{not valid json", '{"not": "a list"}'])
def test_malformed_upload_log_deletes_nothing_and_exits_nonzero(tmp_path, capsys, bad_content):
    env = _build_fake_data_dir(tmp_path)
    env["upload_log_path"].write_text(bad_content, encoding="utf-8")

    rc = delete_main(
        ["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc != 0
    assert capsys.readouterr().err

    for path in env["target_files"] + env["unrelated_files"]:
        assert path.exists()
    assert _db_counts(env["db_path"]) == (2, 2)
    assert not env["deletions_log_path"].exists()


def test_malformed_deletions_log_deletes_nothing_and_exits_nonzero(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    env["deletions_log_path"].write_text("{corrupt", encoding="utf-8")

    rc = delete_main(
        ["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]
    )
    assert rc != 0
    assert capsys.readouterr().err

    for path in env["target_files"] + env["unrelated_files"]:
        assert path.exists()
    assert _db_counts(env["db_path"]) == (2, 2)


def test_partial_failure_still_writes_audit_entry_with_reached_counts(tmp_path):
    env = _build_fake_data_dir(tmp_path)
    plan = build_deletion_plan(tracking_id=env["target_id"], data_dir=env["data_dir"])

    with patch(
        "src.tools.delete_by_tracking_id.sqlite3.connect",
        side_effect=sqlite3.OperationalError("simulated db failure"),
    ):
        with pytest.raises(sqlite3.OperationalError):
            apply_deletion_plan(plan)

    # Files were deleted before DB failed, so deletions_log.json MUST record what was deleted
    assert env["deletions_log_path"].exists()
    log = json.loads(env["deletions_log_path"].read_text(encoding="utf-8"))
    assert len(log) == 1
    assert log[0]["tracking_id"] == env["target_id"]
    assert log[0]["counts"]["raw_uploads"] == 1
    assert log[0]["counts"]["dataset_images"] == 2
    assert log[0]["counts"]["db_predictions"] == 0


def test_nonexistent_db_flag_succeeds_without_creating_db(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)
    missing_db = tmp_path / "does_not_exist" / "predictions.db"

    rc_dry = delete_main(
        [
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
            "--db",
            str(missing_db),
        ]
    )
    assert rc_dry == 0
    dry_out = capsys.readouterr().out
    assert "0 prediction(s), 0 correction(s)" in dry_out
    assert not missing_db.exists()

    rc_apply = delete_main(
        [
            "--tracking-id",
            env["target_id"],
            "--data-dir",
            str(env["data_dir"]),
            "--db",
            str(missing_db),
            "--apply",
        ]
    )
    assert rc_apply == 0
    assert "APPLIED" in capsys.readouterr().out
    assert not missing_db.exists()
    for target_file in env["target_files"]:
        assert not target_file.exists()


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


def test_deletions_log_appends_across_multiple_requests(tmp_path, capsys):
    env = _build_fake_data_dir(tmp_path)

    assert (
        delete_main(
            ["--tracking-id", env["target_id"], "--data-dir", str(env["data_dir"]), "--apply"]
        )
        == 0
    )
    assert "APPLIED" in capsys.readouterr().out
    assert (
        delete_main(
            ["--tracking-id", env["other_id"], "--data-dir", str(env["data_dir"]), "--apply"]
        )
        == 0
    )
    assert "APPLIED" in capsys.readouterr().out

    log = json.loads(env["deletions_log_path"].read_text(encoding="utf-8"))
    assert [entry["tracking_id"] for entry in log] == [env["target_id"], env["other_id"]]
    assert _db_counts(env["db_path"]) == (0, 0)


def test_vault_legacy_list_sidecar_matched_by_pixel_hash(tmp_path, capsys):
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
    assert "APPLIED" in capsys.readouterr().out
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
