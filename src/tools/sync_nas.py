#!/usr/bin/env python3
"""
sync_nas.py — Mirror raw panel photos from TrueNAS to local storage.

Reads source and destination paths from .env so no paths are hardcoded.
Uses robocopy (Windows built-in) for reliable, resumable network transfers.

Usage
-----
    python -m src.tools.sync_nas           # sync and report
    python -m src.tools.sync_nas --dry-run # preview what would be copied

Environment variables (.env)
----------------------------
    NAS_UPLOAD_DIR   : UNC path to the NAS folder  (e.g. \\\\NAS\\share\\raw_uploads)
    LOCAL_UPLOAD_DIR : Local destination folder     (e.g. data\\images\\raw_uploads)

Robocopy exit code reference
----------------------------
    0  No files copied — source and destination already in sync
    1  One or more files copied successfully
    2  Extra files exist in destination (not an error)
    3  Files copied + extra files in destination
    4  Mismatched files detected
    5–7 Combinations of the above (all OK)
    8+ Error — access denied, network unreachable, etc.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ── Config (overridable via .env) ──────────────────────────────────────────────
NAS_DIR   = os.getenv("NAS_UPLOAD_DIR",   r"\\TRUENAS\MainStorage\project_breaker\raw_uploads")
LOCAL_DIR = os.getenv("LOCAL_UPLOAD_DIR", r"data\images\raw_uploads")
LOG_FILE  = os.path.join(os.path.dirname(LOCAL_DIR), "sync_log.txt")

# Robocopy flags:
#   /E   — copy all subdirectories including empty ones
#   /R:3 — retry 3 times on failure
#   /W:5 — wait 5 seconds between retries
#   /NP  — suppress per-file progress (cleaner output)
#   /TEE — write output to both console and log file
ROBOCOPY_FLAGS = ["/E", "/R:3", "/W:5", "/NP", "/TEE"]


def run_sync(dry_run: bool = False) -> int:
    """
    Execute the robocopy mirror and return the exit code.

    Parameters
    ----------
    dry_run : bool
        If True, add /L flag (list-only — no files actually copied).

    Returns
    -------
    int
        Robocopy exit code. 0–7 = success/warning. 8+ = error.
    """
    os.makedirs(LOCAL_DIR, exist_ok=True)

    cmd = [
        "robocopy",
        NAS_DIR,
        LOCAL_DIR,
        *ROBOCOPY_FLAGS,
        f"/LOG:{LOG_FILE}",
    ]

    if dry_run:
        cmd.append("/L")  # list only, no copy

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] PanelSafe — NAS Sync {'(DRY RUN) ' if dry_run else ''}")
    print(f"  Source : {NAS_DIR}")
    print(f"  Dest   : {os.path.abspath(LOCAL_DIR)}")
    print(f"  Log    : {os.path.abspath(LOG_FILE)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False)
    code = result.returncode

    print("-" * 60)
    if code <= 7:
        if code == 0:
            status = "[OK] Already in sync -- no new files."
        elif code == 1:
            status = "[OK] New files copied successfully."
        else:
            status = f"[OK] Sync complete (code {code} -- see log for details)."
    else:
        status = (
            f"[FAILED] Sync error (code {code}). "
            "Check network connection and NAS path in .env"
        )

    print(f"  {status}")
    return code


def main():
    parser = argparse.ArgumentParser(
        description="Sync raw panel photos from TrueNAS to local data/images/raw_uploads/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be copied without actually copying them.",
    )
    args = parser.parse_args()

    code = run_sync(dry_run=args.dry_run)

    # Exit with 0 for success (robocopy codes 0-7), 1 for error (8+)
    sys.exit(0 if code <= 7 else 1)


if __name__ == "__main__":
    main()
