import os
import cv2
import argparse
import random

from src.data_gen.seed_library import SeedLibrary
from src.data_gen.grammar import PanelFactory
from src.data_gen.compositor import Compositor
from src.data_gen.label_writer import write_label_file


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic panel dataset.")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--seed-dir", type=str, default=os.path.join("data", "seeds"), help="Path to seed library")
    parser.add_argument("--bg-dir", type=str, default=os.path.join("data", "backgrounds"), help="Path to background images")
    parser.add_argument("--out-dir", type=str, default=os.path.join("data", "dataset"), help="Output base directory")
    parser.add_argument("--chaos", type=float, default=0.2, help="Chaos factor [0.0 - 1.0]")
    parser.add_argument("--balanced", action="store_true", help="Boost minority class probabilities (OVERSURGE, RCD_SI, OTHER)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of images to reserve for validation [0.0 - 1.0]")
    parser.add_argument("--no-augment", action="store_true", help="Disable per-seed augmentation in compositor")
    parser.add_argument("--resolution", type=int, default=1280, help="Output image size in pixels (e.g. 640 or 1280)")
    args = parser.parse_args()

    # Create output directories (train + val)
    train_img_dir = os.path.join(args.out_dir, "train", "images")
    train_lbl_dir = os.path.join(args.out_dir, "train", "labels")
    val_img_dir   = os.path.join(args.out_dir, "val", "images")
    val_lbl_dir   = os.path.join(args.out_dir, "val", "labels")
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Loading seeds from {args.seed_dir}...")
    seed_lib = SeedLibrary()
    seed_lib.load_seeds(args.seed_dir)

    print(f"Loading backgrounds from {args.bg_dir}...")
    seed_lib.load_backgrounds(args.bg_dir)

    # Report seed counts (after augmentation)
    total_seeds = sum(len(imgs) for imgs in seed_lib.seeds.values())
    print(f"Loaded {total_seeds} total seeds across {len(seed_lib.seeds)} classes:")
    for cls_name in sorted(seed_lib.seeds):
        print(f"  {cls_name:15s}: {len(seed_lib.seeds[cls_name])} seeds")

    if total_seeds == 0:
        print("Error: No seeds found! Please check the seed directory.")
        return

    # Dynamic scaling based on target resolution (default 1280x1280)
    scale_factor = args.resolution / 640.0
    rail_height = int(160 * scale_factor)
    module_width_px = int(40 * scale_factor)
    img_width = args.resolution
    img_height = args.resolution

    factory = PanelFactory(
        chaos_factor=args.chaos,
        rail_height=rail_height,
        boost_minority=args.balanced,
    )
    compositor = Compositor(
        seed_library=seed_lib, 
        img_width=img_width, 
        img_height=img_height,
        module_width_px=module_width_px
    )
    do_augment = not args.no_augment

    # Determine val split indices
    val_count = int(args.count * args.val_split)
    train_count = args.count - val_count
    indices = list(range(args.count))
    random.shuffle(indices)
    val_indices = set(indices[:val_count])

    # Track class distribution for reporting
    class_counts = {}

    print(f"Generating {args.count} images ({train_count} train, {val_count} val)...")
    if args.balanced:
        print("  [BALANCED MODE] Minority class probabilities boosted.")

    for i in range(args.count):
        # 1. Generate grammar layout
        panel = factory.generate()

        # 2. Render image and get raw pixel annotations
        canvas, annotations = compositor.compose(panel, augment=do_augment)

        # 3. Determine output directory (train or val)
        if i in val_indices:
            img_dir, lbl_dir = val_img_dir, val_lbl_dir
        else:
            img_dir, lbl_dir = train_img_dir, train_lbl_dir

        # 4. Save files
        base_name = f"synth_panel_{i:06d}"
        img_path = os.path.join(img_dir, f"{base_name}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")

        cv2.imwrite(img_path, canvas)
        write_label_file(lbl_path, annotations, img_width=img_width, img_height=img_height)

        # Track class distribution
        for ann in annotations:
            cid = ann["class_id"]
            class_counts[cid] = class_counts.get(cid, 0) + 1

        if (i + 1) % 50 == 0 or (i + 1) == args.count:
            print(f"  Processed {i + 1}/{args.count}...")

    # Report final class distribution
    total_anns = sum(class_counts.values())
    class_names = {0: 'MCB', 1: 'RCD', 2: 'RCD_SI', 3: 'MAINBREAKER', 4: 'OVERSURGE', 5: 'OTHER'}
    print(f"\nDone! Dataset saved to {args.out_dir}")
    print(f"Total annotations: {total_anns}")
    for cid in sorted(class_counts):
        name = class_names.get(cid, f'UNK_{cid}')
        pct = class_counts[cid] / total_anns * 100
        print(f"  {cid} ({name:12s}): {class_counts[cid]:5d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

