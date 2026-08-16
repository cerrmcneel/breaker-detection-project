"""Move real images from raw_uploads into the validation split, hash-disjointly.

The previous version of this file used ``shutil.copy()`` and left the source in
place, so every image it touched ended up in BOTH splits once
``mix_raw_uploads.py`` folded raw_uploads into train. That is what produced the
87% train/val overlap observed in the VM's dataset copy on 2026-08-11.

Two rules make that unrepresentable here:

1. Images are **moved**, not copied.
2. Selection is by **content hash**, and anything whose hash already exists in
   train (or already in val) is skipped, so the same picture cannot land in both
   splits even if it arrives twice under different filenames.

Run with ``--dry-run`` first; it reports exactly what would move and why.
"""
import argparse
import hashlib
import os
import shutil

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif")
SYNTHETIC_PREFIX = "synth_panel_"

RAW_DIR = os.path.join("data", "images", "raw_uploads")
TRAIN_IMG_DIR = os.path.join("data", "dataset", "train", "images")
VAL_IMG_DIR = os.path.join("data", "dataset", "val", "images")
VAL_LBL_DIR = os.path.join("data", "dataset", "val", "labels")


def file_hash(path):
    """SHA-256 of a file's bytes, read in chunks so large images stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dir(directory, skip_synthetic=True):
    """Map content hash -> list of filenames for every image in `directory`."""
    hashes = {}
    if not os.path.isdir(directory):
        return hashes
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(IMAGE_EXTS):
            continue
        if skip_synthetic and filename.startswith(SYNTHETIC_PREFIX):
            continue
        hashes.setdefault(file_hash(os.path.join(directory, filename)), []).append(filename)
    return hashes


def split_validation_set(dry_run=False, limit=None):
    """Move unseen raw_uploads images into val/. Returns (moved, skipped)."""
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LBL_DIR, exist_ok=True)

    train_hashes = set(hash_dir(TRAIN_IMG_DIR))
    val_hashes = set(hash_dir(VAL_IMG_DIR))
    print(f"train: {len(train_hashes)} distinct real hashes | val: {len(val_hashes)}")

    moved, skipped = [], []
    for filename in sorted(os.listdir(RAW_DIR)) if os.path.isdir(RAW_DIR) else []:
        if not filename.lower().endswith(IMAGE_EXTS) or filename.startswith(SYNTHETIC_PREFIX):
            continue
        if limit is not None and len(moved) >= limit:
            break

        src = os.path.join(RAW_DIR, filename)
        digest = file_hash(src)

        if digest in train_hashes:
            skipped.append((filename, "already in train"))
            continue
        if digest in val_hashes:
            skipped.append((filename, "already in val"))
            continue

        base = os.path.splitext(filename)[0]
        label_src = os.path.join(RAW_DIR, f"{base}.txt")

        if not dry_run:
            shutil.move(src, os.path.join(VAL_IMG_DIR, filename))
            if os.path.exists(label_src):
                shutil.move(label_src, os.path.join(VAL_LBL_DIR, f"{base}.txt"))
        # Claim the hash immediately so a byte-identical file later in the same
        # run is skipped rather than moved in as a second copy.
        val_hashes.add(digest)
        moved.append((filename, os.path.exists(label_src) or not dry_run))

    verb = "would move" if dry_run else "moved"
    print(f"\n{verb} {len(moved)} image(s) to val/")
    for filename, has_label in moved:
        print(f"  + {filename}{'' if has_label else '   (no label file!)'}")
    if skipped:
        print(f"\nskipped {len(skipped)} already-present image(s):")
        for filename, why in skipped[:20]:
            print(f"  - {filename}  ({why})")
    return moved, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would move without touching anything")
    parser.add_argument("--limit", type=int, default=None,
                        help="move at most N images (useful for topping up val)")
    args = parser.parse_args()
    split_validation_set(dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
