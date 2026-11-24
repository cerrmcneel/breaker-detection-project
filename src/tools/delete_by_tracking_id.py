#!/usr/bin/env python3
"""GDPR photo deletion tool: remove all stored artifacts for a tracking ID.

Searches every local store where a panel photo or its derived metadata can live:
1. ``<upload-dir>/../upload_log.json`` (default ``<data-dir>/images/upload_log.json``;
   keeps consent log entry, nulls filenames, adds ``deleted_on_request`` timestamp)
2. ``<upload-dir>/<saved_filename>`` (default ``<data-dir>/images/raw_uploads/``)
3. ``<data-dir>/dataset/{train,val}/images/`` + matching ``.../labels/<stem>.txt``
   (matched strictly by decoded-pixel fingerprint from stored records owned by
   this tracking ID, never by filename or ``predictions.image_hash``)
4. ``<data-dir>/active_learning/`` pairs (``<stem>.<ext>`` + ``<stem>.json``),
   matched by ``tracking_id`` in the sidecar JSON, linked DB correction record,
   or decoded-pixel fingerprint (unattributed legacy pairs only)
5. ``predictions.db`` (``--db``, or ``PREDICTIONS_DB_PATH``, or
   ``<data-dir>/predictions.db``): ``predictions`` rows and linked
   ``corrections`` rows for that ``tracking_id``

Dry-run by default: prints everything that would be deleted and exits 0 without
modifying disk. Pass ``--apply`` to perform the deletion and append a
filename-free audit entry to ``<data-dir>/deletions_log.json``. If a pixel hash
is also claimed by another stored record, affected hash-matched files are
flagged in the dry run and skipped unless ``--include-shared`` is passed.

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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.image_hash import image_content_hash

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")


@dataclass
class DeletionPlan:
    tracking_id: str
    data_dir: Path
    db_path: Path
    raw_uploads_dir: Path
    upload_log_path: Path
    deletions_log_path: Path
    include_shared: bool = False
    upload_log_indices: List[int] = field(default_factory=list)
    raw_upload_files: List[Path] = field(default_factory=list)
    dataset_image_files: List[Path] = field(default_factory=list)
    dataset_label_files: List[Path] = field(default_factory=list)
    active_learning_files: List[Path] = field(default_factory=list)
    shared_dataset_image_files: List[Path] = field(default_factory=list)
    shared_dataset_label_files: List[Path] = field(default_factory=list)
    shared_active_learning_files: List[Path] = field(default_factory=list)
    shared_hashes: Dict[str, List[str]] = field(default_factory=dict)
    prediction_ids: List[int] = field(default_factory=list)
    correction_ids: List[int] = field(default_factory=list)
    found_in_upload_log: bool = False
    found_in_db: bool = False
    found_in_active_learning: bool = False
    already_deleted_in_upload_log: bool = False

    @property
    def is_known(self) -> bool:
        return self.found_in_upload_log or self.found_in_db or self.found_in_active_learning

    @property
    def has_id_collision(self) -> bool:
        return len(self.upload_log_indices) > 1 or len(self.prediction_ids) > 1

    @property
    def has_shared_files(self) -> bool:
        return bool(
            self.shared_dataset_image_files
            or self.shared_dataset_label_files
            or self.shared_active_learning_files
        )


def _normalize_tracking_id(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


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
) -> Tuple[List[int], List[int], Set[str], Dict[str, Set[str]]]:
    """Return ``(prediction_ids, correction_ids, linked_vault_stems, other_vault_stems)``.

    Intentionally does NOT return ``predictions.image_hash`` for file matching:
    ``/predict/`` records ``image_hash`` without storing any photo, so matching
    files by ``predictions.image_hash`` would let a ``/predict/`` caller delete
    another user's training photo.
    """
    if not db_path.is_file():
        return [], [], set(), {}

    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "predictions" not in tables:
            return [], [], set(), {}

        pred_rows = conn.execute(
            "SELECT id FROM predictions WHERE UPPER(TRIM(tracking_id)) = ? ORDER BY id",
            (tracking_id,),
        ).fetchall()
        prediction_ids = [int(row[0]) for row in pred_rows]

        correction_ids: List[int] = []
        linked_vault_stems: Set[str] = set()
        other_vault_stems: Dict[str, Set[str]] = {}

        if "corrections" in tables:
            if prediction_ids:
                placeholders = ",".join("?" for _ in prediction_ids)
                corr_rows = conn.execute(
                    f"SELECT id, source, note FROM corrections WHERE prediction_id IN ({placeholders}) ORDER BY id",
                    prediction_ids,
                ).fetchall()
                for corr_id, source, note in corr_rows:
                    correction_ids.append(int(corr_id))
                    if source == "active_learning_endpoint" and _is_safe_basename(note):
                        linked_vault_stems.add(note)

            other_corr_rows = conn.execute(
                """
                SELECT c.note, p.tracking_id
                FROM corrections c
                JOIN predictions p ON c.prediction_id = p.id
                WHERE c.source = 'active_learning_endpoint'
                  AND c.note IS NOT NULL
                  AND (p.tracking_id IS NULL OR UPPER(TRIM(p.tracking_id)) != ?)
                """,
                (tracking_id,),
            ).fetchall()
            for note, other_tid in other_corr_rows:
                if not _is_safe_basename(note):
                    continue
                owner = _normalize_tracking_id(other_tid) or "<db-correction>"
                other_vault_stems.setdefault(note, set()).add(owner)

        return prediction_ids, correction_ids, linked_vault_stems, other_vault_stems
    finally:
        conn.close()


def _resolve_db_path(data_dir: Path, db_path: Optional[Path]) -> Path:
    if db_path is not None:
        return Path(db_path)
    env_db = os.getenv("PREDICTIONS_DB_PATH")
    if env_db:
        return Path(env_db)
    return data_dir / "predictions.db"


def _resolve_upload_paths(data_dir: Path, upload_dir: Optional[Path]) -> Tuple[Path, Path]:
    if upload_dir is not None:
        raw_uploads_dir = Path(upload_dir)
        return raw_uploads_dir, raw_uploads_dir.parent / "upload_log.json"
    return data_dir / "images" / "raw_uploads", data_dir / "images" / "upload_log.json"


def build_deletion_plan(
    tracking_id: str,
    data_dir: Path,
    db_path: Optional[Path] = None,
    upload_dir: Optional[Path] = None,
    include_shared: bool = False,
) -> DeletionPlan:
    """Discover every file, log entry, and DB row belonging to ``tracking_id``."""
    normalized_id = _normalize_tracking_id(tracking_id) or ""
    data_dir = Path(data_dir)
    resolved_db_path = _resolve_db_path(data_dir, db_path)
    raw_uploads_dir, upload_log_path = _resolve_upload_paths(data_dir, upload_dir)
    active_learning_dir = data_dir / "active_learning"
    deletions_log_path = data_dir / "deletions_log.json"

    plan = DeletionPlan(
        tracking_id=normalized_id,
        data_dir=data_dir,
        db_path=resolved_db_path,
        raw_uploads_dir=raw_uploads_dir,
        upload_log_path=upload_log_path,
        deletions_log_path=deletions_log_path,
        include_shared=include_shared,
    )

    target_hashes: Set[str] = set()
    seen_raw_paths: Set[Path] = set()
    upload_entries: Optional[List[Any]] = None

    # 1 & 2. upload_log.json and raw_uploads/<saved_filename>
    if upload_log_path.is_file():
        upload_entries = _load_json_list(upload_log_path)
        for idx, entry in enumerate(upload_entries):
            if not isinstance(entry, dict):
                continue
            if _normalize_tracking_id(entry.get("tracking_id")) != normalized_id:
                continue

            # Skip entries already marked deleted so re-running never overwrites
            # the original deleted_on_request timestamp.
            if entry.get("deleted_on_request"):
                plan.already_deleted_in_upload_log = True
                continue

            plan.found_in_upload_log = True
            plan.upload_log_indices.append(idx)

            # Only trust entry["hash"] when a photo was actually stored for this
            # tracking ID (never for duplicate=True or training_consent=False).
            if entry.get("photo_stored") is True:
                entry_hash = entry.get("hash")
                if isinstance(entry_hash, str) and entry_hash:
                    target_hashes.add(entry_hash)

            saved_filename = entry.get("saved_filename")
            if entry.get("photo_stored") is not False and _is_safe_basename(saved_filename):
                raw_path = raw_uploads_dir / saved_filename
                if raw_path.is_file() and raw_path not in seen_raw_paths:
                    seen_raw_paths.add(raw_path)
                    plan.raw_upload_files.append(raw_path)
                    digest = _hash_file(raw_path)
                    if digest is not None:
                        target_hashes.add(digest)

    # 5. predictions.db (read-only inspection before scanning dataset/vault)
    pred_ids, corr_ids, linked_vault_stems, other_db_vault_stems = _inspect_db_readonly(
        resolved_db_path, normalized_id
    )
    if pred_ids:
        plan.found_in_db = True
        plan.prediction_ids = pred_ids
        plan.correction_ids = corr_ids

    # 4 (Part A). Inspect active_learning/ for directly-owned pairs (both target and other IDs)
    json_by_stem: Dict[str, Path] = {}
    images_by_stem: Dict[str, List[Path]] = {}
    direct_vault_stems: Set[str] = set()
    other_owned_vault_stems: Dict[str, Set[str]] = {}

    if active_learning_dir.is_dir():
        for item in sorted(active_learning_dir.iterdir()):
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix == ".json":
                json_by_stem[item.stem] = item
            elif suffix in IMAGE_EXTENSIONS:
                images_by_stem.setdefault(item.stem, []).append(item)

        # Pass 1: JSON sidecars that directly record tracking_id
        for stem, json_path in json_by_stem.items():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(sidecar, dict):
                sidecar_tid = _normalize_tracking_id(sidecar.get("tracking_id"))
                if sidecar_tid == normalized_id:
                    plan.found_in_active_learning = True
                    direct_vault_stems.add(stem)
                elif sidecar_tid is not None:
                    other_owned_vault_stems.setdefault(stem, set()).add(sidecar_tid)

        # Pass 2: vault pairs linked by a matched DB correction row
        for stem in linked_vault_stems:
            if stem in json_by_stem or stem in images_by_stem:
                direct_vault_stems.add(stem)

        # Track vault pairs linked in DB to other tracking IDs
        for stem, owners in other_db_vault_stems.items():
            if stem in direct_vault_stems:
                continue
            if stem in json_by_stem or stem in images_by_stem:
                other_owned_vault_stems.setdefault(stem, set()).update(owners)

        # Collect pixel hashes from vault images this ID owns directly so dataset
        # copies imported from the vault are also found.
        for stem in sorted(direct_vault_stems):
            for img_path in images_by_stem.get(stem, []):
                digest = _hash_file(img_path)
                if digest is not None:
                    target_hashes.add(digest)

    # Detect hashes in target_hashes that another active stored record also claims
    other_hash_owners: Dict[str, Set[str]] = {}
    if target_hashes:
        if upload_entries is not None:
            for idx, entry in enumerate(upload_entries):
                if not isinstance(entry, dict) or entry.get("deleted_on_request"):
                    continue
                other_tid = _normalize_tracking_id(entry.get("tracking_id"))
                if other_tid == normalized_id:
                    continue
                owner_label = other_tid or f"upload_log[{idx}]"

                if entry.get("photo_stored") is True:
                    other_hash = entry.get("hash")
                    if isinstance(other_hash, str) and other_hash in target_hashes:
                        other_hash_owners.setdefault(other_hash, set()).add(owner_label)

                other_saved = entry.get("saved_filename")
                if entry.get("photo_stored") is not False and _is_safe_basename(other_saved):
                    other_raw = raw_uploads_dir / other_saved
                    if other_raw.is_file() and other_raw not in seen_raw_paths:
                        digest = _hash_file(other_raw)
                        if digest is not None and digest in target_hashes:
                            other_hash_owners.setdefault(digest, set()).add(owner_label)

        for stem, owners in other_owned_vault_stems.items():
            for img_path in images_by_stem.get(stem, []):
                digest = _hash_file(img_path)
                if digest is not None and digest in target_hashes:
                    other_hash_owners.setdefault(digest, set()).update(owners)

    plan.shared_hashes = {h: sorted(owners) for h, owners in sorted(other_hash_owners.items())}
    shared_hash_set = set(plan.shared_hashes)
    exclusive_hashes = target_hashes - shared_hash_set

    # 4 (Part B). Pass 3: pixel-fingerprint match across unattributed vault pairs only
    if active_learning_dir.is_dir():
        matched_exclusive_stems: Set[str] = set(direct_vault_stems)
        matched_shared_stems: Set[str] = set()

        if target_hashes:
            for stem, img_paths in images_by_stem.items():
                if stem in direct_vault_stems or stem in other_owned_vault_stems:
                    continue
                matched_exclusive = False
                matched_shared = False
                for img_path in img_paths:
                    digest = _hash_file(img_path)
                    if digest is None:
                        continue
                    if digest in exclusive_hashes:
                        matched_exclusive = True
                        break
                    if digest in shared_hash_set:
                        matched_shared = True
                if matched_exclusive:
                    matched_exclusive_stems.add(stem)
                elif matched_shared:
                    matched_shared_stems.add(stem)

        for stem in sorted(matched_exclusive_stems):
            for img_path in sorted(images_by_stem.get(stem, [])):
                plan.active_learning_files.append(img_path)
            if stem in json_by_stem:
                plan.active_learning_files.append(json_by_stem[stem])

        for stem in sorted(matched_shared_stems):
            for img_path in sorted(images_by_stem.get(stem, [])):
                plan.shared_active_learning_files.append(img_path)
            if stem in json_by_stem:
                plan.shared_active_learning_files.append(json_by_stem[stem])

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
                if digest is None:
                    continue
                label_path = labels_dir / f"{img_path.stem}.txt"
                if digest in exclusive_hashes:
                    plan.dataset_image_files.append(img_path)
                    if label_path.is_file():
                        plan.dataset_label_files.append(label_path)
                elif digest in shared_hash_set:
                    plan.shared_dataset_image_files.append(img_path)
                    if label_path.is_file():
                        plan.shared_dataset_label_files.append(label_path)

    return plan


def _unlink_paths_counting(paths: Sequence[Path], counts: Dict[str, int], key: str) -> None:
    for path in paths:
        if path.is_file():
            path.unlink()
            counts[key] += 1


def apply_deletion_plan(plan: DeletionPlan) -> Dict[str, Any]:
    """Execute ``plan`` on disk and append an audit entry to ``deletions_log.json``.

    Loads and validates both JSON files before deleting anything, and writes
    the audit log in a ``try/finally`` block so even a mid-run failure leaves
    an accurate record of whatever counts were reached.
    """
    # Pre-validate both JSON stores before performing the first delete.
    upload_entries: Optional[List[Any]] = None
    if plan.upload_log_indices and plan.upload_log_path.is_file():
        upload_entries = _load_json_list(plan.upload_log_path)

    existing_log: List[Any] = []
    if plan.deletions_log_path.is_file():
        existing_log = _load_json_list(plan.deletions_log_path)

    timestamp = datetime.now(timezone.utc).isoformat()
    counts: Dict[str, int] = {
        "raw_uploads": 0,
        "dataset_images": 0,
        "dataset_labels": 0,
        "active_learning": 0,
        "upload_log_updated": 0,
        "db_predictions": 0,
        "db_corrections": 0,
    }
    audit_entry: Dict[str, Any] = {
        "tracking_id": plan.tracking_id,
        "timestamp": timestamp,
        "counts": counts,
    }

    dataset_image_targets = list(plan.dataset_image_files)
    dataset_label_targets = list(plan.dataset_label_files)
    active_learning_targets = list(plan.active_learning_files)
    if plan.include_shared:
        dataset_image_targets.extend(plan.shared_dataset_image_files)
        dataset_label_targets.extend(plan.shared_dataset_label_files)
        active_learning_targets.extend(plan.shared_active_learning_files)

    completed = False
    try:
        _unlink_paths_counting(plan.raw_upload_files, counts, "raw_uploads")
        _unlink_paths_counting(dataset_image_targets, counts, "dataset_images")
        _unlink_paths_counting(dataset_label_targets, counts, "dataset_labels")
        _unlink_paths_counting(active_learning_targets, counts, "active_learning")

        if plan.prediction_ids and plan.db_path.is_file():
            conn = sqlite3.connect(str(plan.db_path))
            try:
                tables = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                pred_placeholders = ",".join("?" for _ in plan.prediction_ids)
                with conn:
                    if "corrections" in tables:
                        cur = conn.execute(
                            f"DELETE FROM corrections WHERE prediction_id IN ({pred_placeholders})",
                            plan.prediction_ids,
                        )
                        counts["db_corrections"] = cur.rowcount

                    cur = conn.execute(
                        f"DELETE FROM predictions WHERE id IN ({pred_placeholders})",
                        plan.prediction_ids,
                    )
                    counts["db_predictions"] = cur.rowcount
            finally:
                conn.close()

        if upload_entries is not None:
            for entry in upload_entries:
                if (
                    isinstance(entry, dict)
                    and _normalize_tracking_id(entry.get("tracking_id")) == plan.tracking_id
                    and not entry.get("deleted_on_request")
                ):
                    entry["saved_filename"] = None
                    if "original_filename" in entry:
                        entry["original_filename"] = None
                    entry["deleted_on_request"] = timestamp
                    counts["upload_log_updated"] += 1
            _write_json_atomic(plan.upload_log_path, upload_entries)

        completed = True
    finally:
        # Audit log intentionally stores ONLY tracking ID, timestamp, and
        # per-location counts -- no file names, paths, pixel hashes, or image data.
        if completed or any(counts.values()):
            existing_log.append(audit_entry)
            _write_json_atomic(plan.deletions_log_path, existing_log)

    return audit_entry


def format_dry_run_report(plan: DeletionPlan) -> str:
    lines = [
        f"DRY RUN for tracking ID: {plan.tracking_id} (no changes made; pass --apply to delete)",
    ]

    if plan.has_id_collision:
        lines.append("")
        lines.append(
            f"WARNING: tracking ID collision detected for {plan.tracking_id} "
            f"({len(plan.upload_log_indices)} upload_log entries, {len(plan.prediction_ids)} prediction rows). "
            f"Verify all matches belong to the same request before running --apply."
        )

    if plan.shared_hashes:
        other_owners = sorted({owner for owners in plan.shared_hashes.values() for owner in owners})
        action_note = (
            "WILL BE DELETED (--include-shared is set)"
            if plan.include_shared
            else "SKIPPED unless --include-shared is passed"
        )
        lines.append("")
        lines.append(
            f"WARNING: pixel hash is shared with other stored record(s): {', '.join(other_owners)}. "
            f"Shared hash-matched files: {action_note}."
        )
        lines.append(f"  shared dataset images ({len(plan.shared_dataset_image_files)}):")
        lines.extend(f"    - {p}" for p in plan.shared_dataset_image_files)
        lines.append(f"  shared dataset labels ({len(plan.shared_dataset_label_files)}):")
        lines.extend(f"    - {p}" for p in plan.shared_dataset_label_files)
        lines.append(f"  shared active_learning files ({len(plan.shared_active_learning_files)}):")
        lines.extend(f"    - {p}" for p in plan.shared_active_learning_files)

    lines.extend(
        [
            "",
            f"raw_uploads files ({len(plan.raw_upload_files)}):",
        ]
    )
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
    if plan.has_shared_files and not plan.include_shared:
        skipped = (
            len(plan.shared_dataset_image_files)
            + len(plan.shared_dataset_label_files)
            + len(plan.shared_active_learning_files)
        )
        other_owners = sorted({owner for owners in plan.shared_hashes.values() for owner in owners})
        lines.append(
            f"  shared files skipped:       {skipped} "
            f"(shared with {', '.join(other_owners)}; pass --include-shared with --apply to delete)"
        )
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
        default=None,
        help="Root data directory (default: data; honours UPLOAD_DIR when omitted).",
    )
    parser.add_argument(
        "--upload-dir",
        default=None,
        help="Raw uploads directory (default: UPLOAD_DIR when --data-dir is omitted, else <data-dir>/images/raw_uploads).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to predictions SQLite DB (default: PREDICTIONS_DB_PATH if set, else <data-dir>/predictions.db).",
    )
    parser.add_argument(
        "--include-shared",
        action="store_true",
        help="Also delete hash-matched dataset/vault files whose pixel hash is claimed by another stored record.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files/rows and update logs. Without this flag, runs as a dry run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tracking_id = _normalize_tracking_id(args.tracking_id)
    if not tracking_id:
        print("Error: --tracking-id must not be empty.", file=sys.stderr)
        return 1

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        upload_dir = Path(args.upload_dir) if args.upload_dir else None
    else:
        data_dir = Path("data")
        if args.upload_dir:
            upload_dir = Path(args.upload_dir)
        else:
            env_upload_dir = os.getenv("UPLOAD_DIR")
            upload_dir = Path(env_upload_dir) if env_upload_dir else None

    db_path = Path(args.db) if args.db else None

    try:
        plan = build_deletion_plan(
            tracking_id=tracking_id,
            data_dir=data_dir,
            db_path=db_path,
            upload_dir=upload_dir,
            include_shared=args.include_shared,
        )
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        print(f"Error: failed to inspect data stores: {exc}", file=sys.stderr)
        return 1

    if not plan.is_known:
        if plan.already_deleted_in_upload_log:
            print(f"Tracking ID '{tracking_id}' was already deleted; no changes made.")
            return 0
        print(
            f"Error: tracking ID '{tracking_id}' was not found in upload_log.json, "
            f"predictions DB, or active_learning/.",
            file=sys.stderr,
        )
        return 1

    if not args.apply:
        print(format_dry_run_report(plan))
        return 0

    try:
        audit_entry = apply_deletion_plan(plan)
    except (json.JSONDecodeError, ValueError, OSError, sqlite3.Error) as exc:
        print(f"Error: deletion failed: {exc}", file=sys.stderr)
        return 1

    print(format_apply_report(plan, audit_entry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
