#!/usr/bin/env python3
"""GDPR photo deletion tool: remove all stored artifacts for a tracking ID.

Searches every local store where a panel photo or its derived metadata can live:
1. ``<data-dir>/images/upload_log.json`` (keeps consent log entry, nulls filename,
   adds ``deleted_on_request`` timestamp)
2. ``<data-dir>/images/raw_uploads/<saved_filename>``
3. ``<data-dir>/dataset/{train,val}/images/`` + matching ``.../labels/<stem>.txt``
   (matched strictly by decoded-pixel fingerprint, never by filename)
4. ``<data-dir>/active_learning/`` pairs (``<stem>.<ext>`` + ``<stem>.json``),
   matched by ``tracking_id`` in the sidecar JSON, linked DB correction record,
   or decoded-pixel fingerprint
5. ``<data-dir>/predictions.db`` (or ``--db PATH``): ``predictions`` rows and
   linked ``corrections`` rows for that ``tracking_id``

Dry-run by default: prints everything that would be deleted and exits 0 without
modifying disk. Pass ``--apply`` to perform the deletion and append a
filename-free audit entry to ``<data-dir>/deletions_log.json``.

Usage
-----
    python -m src.tools.delete_by_tracking_id --tracking-id BKR-XXXXX
    python -m src.tools.delete_by_tracking_id --tracking-id BKR-XXXXX --apply
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from src.image_hash import image_content_hash

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")


@dataclass
class DeletionPlan:
    tracking_id: str
    data_dir: Path
    db_path: Path
    upload_log_path: Path
    deletions_log_path: Path
    upload_log_indices: List[int] = field(default_factory=list)
    raw_upload_files: List[Path] = field(default_factory=list)
    dataset_image_files: List[Path] = field(default_factory=list)
    dataset_label_files: List[Path] = field(default_factory=list)
    active_learning_files: List[Path] = field(default_factory=list)
    prediction_ids: List[int] = field(default_factory=list)
    correction_ids: List[int] = field(default_factory=list)
    found_in_upload_log: bool = False
    found_in_db: bool = False
    found_in_active_learning: bool = False

    @property
    def is_known(self) -> bool:
        return self.found_in_upload_log or self.found_in_db or self.found_in_active_learning


def _is_safe_basename(name: Any) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and os.path.basename(name) == name
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
    )


def _hash_file(path: Path) -> Optional[str]:
    try:
        return image_content_hash(path.read_bytes())
    except OSError:
        return None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _load_json_list(path: Path) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def _inspect_db_readonly(
    db_path: Path,
    tracking_id: str,
) -> tuple[List[int], List[int], Set[str], Set[str]]:
    """Return (prediction_ids, correction_ids, image_hashes, linked_vault_stems)."""
    if not db_path.is_file():
        return [], [], set(), set()

    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "predictions" not in tables:
            return [], [], set(), set()

        pred_rows = conn.execute(
            "SELECT id, image_hash FROM predictions WHERE tracking_id = ? ORDER BY id",
            (tracking_id,),
        ).fetchall()
        prediction_ids = [int(row[0]) for row in pred_rows]
        hashes = {row[1] for row in pred_rows if isinstance(row[1], str) and row[1]}

        correction_ids: List[int] = []
        linked_vault_stems: Set[str] = set()
        if prediction_ids and "corrections" in tables:
            placeholders = ",".join("?" for _ in prediction_ids)
            corr_rows = conn.execute(
                f"SELECT id, source, note FROM corrections WHERE prediction_id IN ({placeholders}) ORDER BY id",
                prediction_ids,
            ).fetchall()
            for corr_id, source, note in corr_rows:
                correction_ids.append(int(corr_id))
                if source == "active_learning_endpoint" and _is_safe_basename(note):
                    linked_vault_stems.add(note)

        return prediction_ids, correction_ids, hashes, linked_vault_stems
    finally:
        conn.close()


def build_deletion_plan(
    tracking_id: str,
    data_dir: Path,
    db_path: Optional[Path] = None,
) -> DeletionPlan:
    """Discover every file, log entry, and DB row belonging to ``tracking_id``."""
    data_dir = Path(data_dir)
    resolved_db_path = Path(db_path) if db_path is not None else data_dir / "predictions.db"
    upload_log_path = data_dir / "images" / "upload_log.json"
    raw_uploads_dir = data_dir / "images" / "raw_uploads"
    active_learning_dir = data_dir / "active_learning"
    deletions_log_path = data_dir / "deletions_log.json"

    plan = DeletionPlan(
        tracking_id=tracking_id,
        data_dir=data_dir,
        db_path=resolved_db_path,
        upload_log_path=upload_log_path,
        deletions_log_path=deletions_log_path,
    )

    target_hashes: Set[str] = set()
    seen_raw_paths: Set[Path] = set()

    # 1 & 2. upload_log.json and raw_uploads/<saved_filename>
    if upload_log_path.is_file():
        entries = _load_json_list(upload_log_path)
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict) or entry.get("tracking_id") != tracking_id:
                continue
            plan.found_in_upload_log = True
            plan.upload_log_indices.append(idx)

            entry_hash = entry.get("hash")
            if isinstance(entry_hash, str) and entry_hash:
                target_hashes.add(entry_hash)

            saved_filename = entry.get("saved_filename")
            if _is_safe_basename(saved_filename):
                raw_path = raw_uploads_dir / saved_filename
                if raw_path.is_file() and raw_path not in seen_raw_paths:
                    seen_raw_paths.add(raw_path)
                    plan.raw_upload_files.append(raw_path)
                    digest = _hash_file(raw_path)
                    if digest is not None:
                        target_hashes.add(digest)

    # 5. predictions.db (read-only inspection before scanning dataset/vault)
    pred_ids, corr_ids, db_hashes, linked_vault_stems = _inspect_db_readonly(
        resolved_db_path, tracking_id
    )
    if pred_ids:
        plan.found_in_db = True
        plan.prediction_ids = pred_ids
        plan.correction_ids = corr_ids
        target_hashes.update(db_hashes)

    # 4. active_learning/ pairs (<stem>.<ext> and <stem>.json)
    if active_learning_dir.is_dir():
        json_by_stem: Dict[str, Path] = {}
        images_by_stem: Dict[str, List[Path]] = {}

        for item in sorted(active_learning_dir.iterdir()):
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix == ".json":
                json_by_stem[item.stem] = item
            elif suffix in IMAGE_EXTENSIONS:
                images_by_stem.setdefault(item.stem, []).append(item)

        matched_stems: Set[str] = set()

        # Pass 1: JSON sidecars that directly record tracking_id
        for stem, json_path in json_by_stem.items():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(sidecar, dict) and sidecar.get("tracking_id") == tracking_id:
                plan.found_in_active_learning = True
                matched_stems.add(stem)

        # Pass 2: vault pairs linked by a matched DB correction row
        for stem in linked_vault_stems:
            if stem in json_by_stem or stem in images_by_stem:
                matched_stems.add(stem)

        # Collect pixel hashes from already-matched vault images so dataset copies
        # imported from the vault are also found.
        for stem in list(matched_stems):
            for img_path in images_by_stem.get(stem, []):
                digest = _hash_file(img_path)
                if digest is not None:
                    target_hashes.add(digest)

        # Pass 3: pixel-fingerprint match across remaining vault images
        if target_hashes:
            for stem, img_paths in images_by_stem.items():
                if stem in matched_stems:
                    continue
                for img_path in img_paths:
                    digest = _hash_file(img_path)
                    if digest is not None and digest in target_hashes:
                        matched_stems.add(stem)
                        break

        for stem in sorted(matched_stems):
            for img_path in sorted(images_by_stem.get(stem, [])):
                plan.active_learning_files.append(img_path)
            if stem in json_by_stem:
                plan.active_learning_files.append(json_by_stem[stem])

    # 3. dataset/{train,val}/images/ + matching .../labels/<stem>.txt
    # Strictly matched by decoded-pixel hash -- NEVER by filename.
    if target_hashes:
        for split in ("train", "val"):
            images_dir = data_dir / "dataset" / split / "images"
            labels_dir = data_dir / "dataset" / split / "labels"
            if not images_dir.is_dir():
                continue
            for img_path in sorted(images_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                digest = _hash_file(img_path)
                if digest is not None and digest in target_hashes:
                    plan.dataset_image_files.append(img_path)
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.is_file():
                        plan.dataset_label_files.append(label_path)

    return plan


def _unlink_paths(paths: Sequence[Path]) -> int:
    deleted = 0
    for path in paths:
        if path.is_file():
            path.unlink()
            deleted += 1
    return deleted


def apply_deletion_plan(plan: DeletionPlan) -> Dict[str, Any]:
    """Execute ``plan`` on disk and append an audit entry to ``deletions_log.json``."""
    timestamp = datetime.now(timezone.utc).isoformat()

    raw_uploads_deleted = _unlink_paths(plan.raw_upload_files)
    dataset_images_deleted = _unlink_paths(plan.dataset_image_files)
    dataset_labels_deleted = _unlink_paths(plan.dataset_label_files)
    active_learning_deleted = _unlink_paths(plan.active_learning_files)

    db_predictions_deleted = 0
    db_corrections_deleted = 0
    if plan.prediction_ids and plan.db_path.is_file():
        conn = sqlite3.connect(str(plan.db_path))
        try:
            with conn:
                if plan.correction_ids:
                    corr_placeholders = ",".join("?" for _ in plan.correction_ids)
                    cur = conn.execute(
                        f"DELETE FROM corrections WHERE id IN ({corr_placeholders})",
                        plan.correction_ids,
                    )
                    db_corrections_deleted = cur.rowcount

                pred_placeholders = ",".join("?" for _ in plan.prediction_ids)
                cur = conn.execute(
                    f"DELETE FROM predictions WHERE id IN ({pred_placeholders})",
                    plan.prediction_ids,
                )
                db_predictions_deleted = cur.rowcount
        finally:
            conn.close()

    upload_log_updated = 0
    if plan.upload_log_indices and plan.upload_log_path.is_file():
        entries = _load_json_list(plan.upload_log_path)
        for entry in entries:
            if isinstance(entry, dict) and entry.get("tracking_id") == plan.tracking_id:
                entry["saved_filename"] = None
                if "original_filename" in entry:
                    entry["original_filename"] = None
                entry["deleted_on_request"] = timestamp
                upload_log_updated += 1
        _write_json_atomic(plan.upload_log_path, entries)

    # Audit log intentionally stores ONLY tracking ID, timestamp, and per-location
    # counts -- no file names, paths, pixel hashes, or image data.
    audit_entry = {
        "tracking_id": plan.tracking_id,
        "timestamp": timestamp,
        "counts": {
            "raw_uploads": raw_uploads_deleted,
            "dataset_images": dataset_images_deleted,
            "dataset_labels": dataset_labels_deleted,
            "active_learning": active_learning_deleted,
            "upload_log_updated": upload_log_updated,
            "db_predictions": db_predictions_deleted,
            "db_corrections": db_corrections_deleted,
        },
    }

    existing_log: List[Any] = []
    if plan.deletions_log_path.is_file():
        existing_log = _load_json_list(plan.deletions_log_path)
    existing_log.append(audit_entry)
    _write_json_atomic(plan.deletions_log_path, existing_log)

    return audit_entry


def format_dry_run_report(plan: DeletionPlan) -> str:
    lines = [
        f"DRY RUN for tracking ID: {plan.tracking_id} (no changes made; pass --apply to delete)",
        "",
        f"raw_uploads files ({len(plan.raw_upload_files)}):",
    ]
    lines.extend(f"  - {p}" for p in plan.raw_upload_files)

    lines.append(f"dataset images ({len(plan.dataset_image_files)}):")
    lines.extend(f"  - {p}" for p in plan.dataset_image_files)

    lines.append(f"dataset labels ({len(plan.dataset_label_files)}):")
    lines.extend(f"  - {p}" for p in plan.dataset_label_files)

    lines.append(f"active_learning files ({len(plan.active_learning_files)}):")
    lines.extend(f"  - {p}" for p in plan.active_learning_files)

    lines.append(
        f"upload_log.json entries to update ({len(plan.upload_log_indices)}): {plan.upload_log_path}"
    )
    lines.append(
        f"SQLite DB rows ({plan.db_path}): "
        f"{len(plan.prediction_ids)} prediction(s), {len(plan.correction_ids)} correction(s)"
    )
    return "\n".join(lines)


def format_apply_report(plan: DeletionPlan, audit_entry: Dict[str, Any]) -> str:
    counts = audit_entry["counts"]
    lines = [
        f"APPLIED deletion for tracking ID: {plan.tracking_id} at {audit_entry['timestamp']}",
        f"  raw_uploads deleted:        {counts['raw_uploads']}",
        f"  dataset images deleted:     {counts['dataset_images']}",
        f"  dataset labels deleted:     {counts['dataset_labels']}",
        f"  active_learning deleted:    {counts['active_learning']}",
        f"  upload_log entries updated: {counts['upload_log_updated']}",
        f"  db predictions deleted:     {counts['db_predictions']}",
        f"  db corrections deleted:     {counts['db_corrections']}",
        f"  audit log appended:         {plan.deletions_log_path}",
    ]
    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete stored photos and records for a PanelSafe tracking ID (GDPR erasure).",
    )
    parser.add_argument(
        "--tracking-id",
        required=True,
        help="Tracking ID to delete (e.g. BKR-XXXXX).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to predictions SQLite DB (default: <data-dir>/predictions.db).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files/rows and update logs. Without this flag, runs as a dry run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tracking_id = args.tracking_id.strip()
    if not tracking_id:
        print("Error: --tracking-id must not be empty.", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir)
    db_path = Path(args.db) if args.db else None

    plan = build_deletion_plan(tracking_id=tracking_id, data_dir=data_dir, db_path=db_path)
    if not plan.is_known:
        print(
            f"Error: tracking ID '{tracking_id}' was not found in upload_log.json, "
            f"predictions DB, or active_learning/.",
            file=sys.stderr,
        )
        return 1

    if not args.apply:
        print(format_dry_run_report(plan))
        return 0

    audit_entry = apply_deletion_plan(plan)
    print(format_apply_report(plan, audit_entry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
