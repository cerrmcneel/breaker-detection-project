#!/usr/bin/env python3
"""Predict what /upload/ would do with a folder of photos, without uploading them.

Useful before a bulk import: shows which photos are already held, which are new,
and which would be refused outright.

It imports the SAME functions the gateway uses (``validate_image_upload``,
``image_content_hash``, ``MAX_FILE_SIZE``) rather than reimplementing them, so
this cannot quietly drift from production behaviour.

Usage
-----
    python -m src.tools.check_upload_batch "C:/path/to/photos"
    python -m src.tools.check_upload_batch ./photos --against-hashes vm_hashes.txt

``--against-hashes`` takes a file of one pixel-content hash per line (extra
columns ignored), so a batch can be checked against a REMOTE dataset. Produce it
on the other machine with ``--emit-hashes``.
"""
import argparse
import os
import sys

from app.main import MAX_FILE_SIZE, image_content_hash

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif")
SYNTHETIC_PREFIX = "synth_panel_"

DATASET_DIRS = (
    os.path.join("data", "dataset", "train", "images"),
    os.path.join("data", "dataset", "val", "images"),
    os.path.join("data", "images", "raw_uploads"),
)


def dataset_hashes(root="."):
    """Map pixel-content hash -> (split, filename) for the local dataset."""
    found = {}
    for rel in DATASET_DIRS:
        directory = os.path.join(root, rel)
        if not os.path.isdir(directory):
            continue
        split = os.path.basename(os.path.dirname(directory))
        for filename in sorted(os.listdir(directory)):
            if not filename.lower().endswith(IMAGE_EXTS) or filename.startswith(SYNTHETIC_PREFIX):
                continue
            with open(os.path.join(directory, filename), "rb") as f:
                digest = image_content_hash(f.read())
            if digest:
                found.setdefault(digest, (split, filename))
    return found


def classify(path, known):
    """Return (verdict, note) exactly as the gateway would decide it."""
    size = os.path.getsize(path)
    if size > MAX_FILE_SIZE:
        return "413 TOO LARGE", f"{size / 1024 / 1024:.1f} MB exceeds the {MAX_FILE_SIZE // 1024 // 1024} MB limit"
    with open(path, "rb") as f:
        data = f.read()
    digest = image_content_hash(data)
    if digest is None:
        return "400 INVALID", "does not decode as an image"
    if digest in known:
        split, name = known[digest]
        return "DUPLICATE", f"already held in {split} as {name}"
    return "NEW", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", help="folder of photos to check")
    parser.add_argument("--root", default=".", help="project root holding data/ (default: cwd)")
    parser.add_argument("--against-hashes", metavar="FILE",
                        help="check against hashes from another machine instead of local data/")
    parser.add_argument("--emit-hashes", action="store_true",
                        help="print this machine's dataset hashes and exit (feed to --against-hashes)")
    args = parser.parse_args()

    if args.emit_hashes:
        for digest, (split, name) in sorted(dataset_hashes(args.root).items()):
            print(digest, split, name)
        return 0

    if not args.folder:
        parser.error("a folder is required unless --emit-hashes is given")
    if not os.path.isdir(args.folder):
        parser.error(f"not a folder: {args.folder}")

    if args.against_hashes:
        known = {}
        with open(args.against_hashes, encoding="utf-8") as f:
            for line in f:
                parts = line.split(None, 2)
                if parts:
                    known[parts[0]] = (parts[1] if len(parts) > 1 else "remote",
                                       parts[2].strip() if len(parts) > 2 else "?")
        source = args.against_hashes
    else:
        known = dataset_hashes(args.root)
        source = "local dataset"

    candidates = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(IMAGE_EXTS))
    if not candidates:
        print("no images found in that folder")
        return 0

    print(f"checking {len(candidates)} photo(s) against {len(known)} known images ({source})\n")
    rows = [(name, *classify(os.path.join(args.folder, name), known)) for name in candidates]
    width = max(len(r[0]) for r in rows) + 2

    for name, verdict, note in rows:
        print(f"  {name.ljust(width)}{verdict:<14} {note}")

    counts = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    print("\n  " + " | ".join(f"{v}: {n}" for v, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
