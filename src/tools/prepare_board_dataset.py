"""Derive a 1-class BOARD dataset from the existing 6-class device labels.

WHY THIS EXISTS
---------------
The phone-side viewfinder does not need to know which device is which. It needs
one question answered per frame: *is the board framed well enough that the
lettering will be readable?* That is a single-class detection problem, and a
single large target survives the low resolution a live preview stream runs at,
where per-device boxes on a dense DIN rail would not.

No new annotation is required. Every 6-class label file already marks every
device in the image, so the union of those boxes is the region the board
occupies. This script reads the existing labels and writes one BOARD box per
image.

WHAT THE BOX ACTUALLY IS
------------------------
The union of the device boxes, not the enclosure. A real *cuadro* extends past
its outermost breakers (frame, trunking, the cover), so this box is tighter than
the physical board by roughly the width of that border. ``--pad`` grows it back
by a fraction of the union size.

That tightness is arguably correct for the viewfinder: what has to be inside the
frame, sharp and large enough to read, is the device region, not the plastic
surround. If you later want the true enclosure, that needs real annotation --
do not fake it by inflating ``--pad`` until it looks right.

SPLITS
------
Train/val membership is mirrored from the existing split, never recomputed. That
split was made hash-disjoint deliberately (see ``split_validation.py``), and
re-splitting here would risk putting the same panel on both sides.

USAGE
-----
    python -m src.tools.prepare_board_dataset --dry-run
    python -m src.tools.prepare_board_dataset
    python -m src.tools.prepare_board_dataset --exclude-synthetic --pad 0.06
"""
import argparse
import os
import shutil
import statistics

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")

# Exclude by the one prefix that is stable -- the synthetic generator's own --
# rather than allowlisting the real-image prefixes. An allowlist silently drops
# every image named by a convention nobody remembered to add, which is exactly
# how four separate evaluation bugs corrupted this project's metrics. See
# docs/okf/methodology/evaluation_rigor.md.
SYNTHETIC_PREFIX = "synth_panel_"

SRC_ROOT = os.path.join("data", "dataset")
OUT_ROOT = os.path.join("data", "board_dataset")
SPLITS = ("train", "val")
BOARD_CLASS_ID = 0


def parse_label_file(path):
    """Return [(cx, cy, w, h), ...] in normalized YOLO form, skipping bad rows."""
    boxes = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) < 5:
                print(f"  ! {os.path.basename(path)}:{line_no} has {len(parts)} fields, skipped")
                continue
            try:
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                print(f"  ! {os.path.basename(path)}:{line_no} is not numeric, skipped")
                continue
            if w <= 0 or h <= 0:
                print(f"  ! {os.path.basename(path)}:{line_no} has non-positive size, skipped")
                continue
            boxes.append((cx, cy, w, h))
    return boxes


def union_box(boxes, pad=0.0):
    """Union of YOLO boxes -> one padded YOLO box, clamped to the image.

    `pad` is a fraction of the union's own size, so it scales with the board
    rather than adding a fixed slab to a close-up and a wide shot alike.
    """
    x1 = min(cx - w / 2 for cx, _, w, _ in boxes)
    x2 = max(cx + w / 2 for cx, _, w, _ in boxes)
    y1 = min(cy - h / 2 for _, cy, _, h in boxes)
    y2 = max(cy + h / 2 for _, cy, _, h in boxes)

    if pad:
        # Compute both offsets BEFORE moving either edge: shifting x1 first and
        # then reusing (x2 - x1) pads the right edge more than the left.
        dx = (x2 - x1) * pad
        dy = (y2 - y1) * pad
        x1, x2 = x1 - dx, x2 + dx
        y1, y2 = y1 - dy, y2 + dy

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def find_image(images_dir, stem):
    for ext in IMAGE_EXTS:
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def place_image(src, dst, mode):
    """Hardlink by default: no duplicated bytes, and no admin rights (unlike
    symlinks on Windows). Falls back to copying across volumes."""
    if os.path.exists(dst):
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        if mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        else:
            os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_split(split, args):
    src_images = os.path.join(SRC_ROOT, split, "images")
    src_labels = os.path.join(SRC_ROOT, split, "labels")
    out_images = os.path.join(OUT_ROOT, split, "images")
    out_labels = os.path.join(OUT_ROOT, split, "labels")

    if not os.path.isdir(src_labels):
        print(f"[{split}] no labels directory at {src_labels}, skipping")
        return {}

    if not args.dry_run:
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)

    stats = {"written": 0, "no_image": 0, "no_boxes": 0, "too_few": 0,
             "too_small": 0, "synthetic": 0, "areas": []}

    for name in sorted(os.listdir(src_labels)):
        if not name.endswith(".txt") or name == "classes.txt":
            continue
        stem = name[:-4]

        if args.exclude_synthetic and stem.startswith(SYNTHETIC_PREFIX):
            stats["synthetic"] += 1
            continue

        boxes = parse_label_file(os.path.join(src_labels, name))
        if not boxes:
            stats["no_boxes"] += 1
            continue
        if len(boxes) < args.min_boxes:
            # One or two devices is a close-up of a breaker, not a board shot.
            stats["too_few"] += 1
            continue

        image_path = find_image(src_images, stem)
        if image_path is None:
            stats["no_image"] += 1
            continue

        cx, cy, w, h = union_box(boxes, args.pad)
        area = w * h
        if area < args.min_area:
            stats["too_small"] += 1
            continue
        stats["areas"].append(area)

        if not args.dry_run:
            place_image(image_path, os.path.join(out_images, os.path.basename(image_path)),
                        args.link)
            with open(os.path.join(out_labels, name), "w", encoding="utf-8") as fh:
                fh.write(f"{BOARD_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        stats["written"] += 1

    return stats


def write_yaml(args):
    path = "data_board.yaml"
    # Forward slashes: ultralytics reads this on both Windows and the Linux
    # containers, and a backslash from os.path.join breaks the Linux side.
    out_root = OUT_ROOT.replace(os.sep, "/")
    body = (
        "# Generated by src/tools/prepare_board_dataset.py -- do not hand-edit.\n"
        "# One class: the region of the image occupied by breaker devices, used to\n"
        "# train the viewfinder framing model. Regenerate instead of editing.\n"
        f"train: {out_root}/train/images\n"
        f"val: {out_root}/val/images\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: 'BOARD'\n"
    )
    if args.dry_run:
        print(f"\n[dry-run] would write {path}:\n{body}")
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(f"\nWrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pad", type=float, default=0.04,
                    help="grow the union by this fraction of its own size (default 0.04)")
    ap.add_argument("--min-boxes", type=int, default=3,
                    help="skip images with fewer device boxes than this (default 3)")
    ap.add_argument("--min-area", type=float, default=0.01,
                    help="skip boards covering less than this fraction of the image")
    ap.add_argument("--exclude-synthetic", action="store_true",
                    help=f"skip generated images (prefix '{SYNTHETIC_PREFIX}')")
    ap.add_argument("--link", choices=("hardlink", "copy", "symlink"), default="hardlink",
                    help="how to place images in the output (default hardlink)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written, touching nothing")
    args = ap.parse_args()

    if not 0.0 <= args.pad < 1.0:
        ap.error("--pad must be in [0, 1)")

    print(f"Source: {SRC_ROOT}  ->  output: {OUT_ROOT}"
          f"{'  (dry run)' if args.dry_run else ''}")
    print(f"pad={args.pad}  min_boxes={args.min_boxes}  "
          f"exclude_synthetic={args.exclude_synthetic}  link={args.link}\n")

    total = 0
    for split in SPLITS:
        stats = build_split(split, args)
        if not stats:
            continue
        total += stats["written"]
        areas = stats["areas"]
        print(f"[{split}] {stats['written']} board labels"
              f"{' (not written -- dry run)' if args.dry_run else ''}")
        if areas:
            print(f"    board covers {statistics.median(areas) * 100:.1f}% of the frame "
                  f"(median), {min(areas) * 100:.1f}%-{max(areas) * 100:.1f}% range")
        skipped = {k: v for k, v in stats.items()
                   if k not in ("written", "areas") and v}
        if skipped:
            print(f"    skipped: {', '.join(f'{k}={v}' for k, v in skipped.items())}")

    if total:
        write_yaml(args)
        print("\nTrain with:")
        print("    yolo detect train data=data_board.yaml model=yolo26n.pt imgsz=320 epochs=60")
        print("(imgsz 320 on purpose: the viewfinder runs on a downscaled preview stream.)")
    else:
        print("\nNothing written. Check that data/dataset/<split>/labels exists and is populated.")


if __name__ == "__main__":
    main()
