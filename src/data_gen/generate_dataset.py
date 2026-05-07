import os
import cv2
import argparse

from src.data_gen.seed_library import SeedLibrary
from src.data_gen.grammar import PanelFactory
from src.data_gen.compositor import Compositor
from src.data_gen.label_writer import write_label_file

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic panel dataset.")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--seed-dir", type=str, default=os.path.join("data", "seeds"), help="Path to seed library")
    parser.add_argument("--bg-dir", type=str, default=os.path.join("data", "backgrounds"), help="Path to background images")
    parser.add_argument("--out-dir", type=str, default=os.path.join("data", "dataset", "train"), help="Output directory")
    parser.add_argument("--chaos", type=float, default=0.2, help="Chaos factor [0.0 - 1.0]")
    args = parser.parse_args()

    # Create output directories
    img_dir = os.path.join(args.out_dir, "images")
    lbl_dir = os.path.join(args.out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print(f"Loading seeds from {args.seed_dir}...")
    seed_lib = SeedLibrary()
    seed_lib.load_seeds(args.seed_dir)
    
    print(f"Loading backgrounds from {args.bg_dir}...")
    seed_lib.load_backgrounds(args.bg_dir)
    
    # Optional: check if we have enough seeds loaded
    total_seeds = sum(len(imgs) for imgs in seed_lib.seeds.values())
    print(f"Loaded {total_seeds} total seeds across {len(seed_lib.seeds)} classes.")
    
    if total_seeds == 0:
        print("Error: No seeds found! Please check the seed directory.")
        return

    factory = PanelFactory(chaos_factor=args.chaos, rail_height=160)
    compositor = Compositor(seed_library=seed_lib, img_width=640, img_height=640)

    print(f"Generating {args.count} images...")
    for i in range(args.count):
        # 1. Generate grammar layout
        panel = factory.generate()
        
        # 2. Render image and get raw pixel annotations
        canvas, annotations = compositor.compose(panel)
        
        # 3. Save files
        base_name = f"synth_panel_{i:06d}"
        img_path = os.path.join(img_dir, f"{base_name}.jpg")
        lbl_path = os.path.join(lbl_dir, f"{base_name}.txt")
        
        cv2.imwrite(img_path, canvas)
        write_label_file(lbl_path, annotations, img_width=640, img_height=640)
        
        if (i + 1) % 50 == 0 or (i + 1) == args.count:
            print(f"  Processed {i + 1}/{args.count}...")
            
    print(f"Done! Dataset saved to {args.out_dir}")

if __name__ == "__main__":
    main()
