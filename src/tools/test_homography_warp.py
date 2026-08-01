import os

import cv2

from src.data_gen.compositor import Compositor
from src.data_gen.grammar import PanelFactory
from src.data_gen.seed_library import SeedLibrary


def main():
    print("PanelSafe Homography & Resolution Verification Tool")
    print("--------------------------------------------------")
    
    # 1. Initialize folders
    seed_dir = os.path.join("data", "seeds")
    bg_dir = os.path.join("data", "backgrounds")
    out_dir = os.path.join("data", "test_runs")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(seed_dir):
        print(f"Error: Seed library folder not found at '{seed_dir}'")
        return

    # 2. Load Seed Library & Backgrounds
    seed_lib = SeedLibrary()
    seed_lib.load_seeds(seed_dir)
    seed_lib.load_backgrounds(bg_dir)

    # 3. Create proportional layout parameters
    resolution = 1280
    scale_factor = resolution / 640.0
    rail_height = int(160 * scale_factor)
    module_width_px = int(40 * scale_factor)

    factory = PanelFactory(
        chaos_factor=0.6,
        rail_height=rail_height,
        boost_minority=True
    )
    
    # Enable compositor with 1280x1280 resolution
    compositor = Compositor(
        seed_library=seed_lib, 
        img_width=resolution, 
        img_height=resolution,
        module_width_px=module_width_px
    )

    # 4. Generate & Compose Panel
    panel = factory.generate()
    
    # Enforce perspective warp during compose (force augment=True)
    canvas, annotations = compositor.compose(panel, augment=True)

    # 5. Verify image dimensions
    assert canvas.shape == (resolution, resolution, 3), f"Incorrect dimensions: {canvas.shape}"
    print(f"OK: Native resolution is correct: {canvas.shape[1]}x{canvas.shape[0]} BGR")
    
    # 6. Save original warped canvas
    raw_out = os.path.join(out_dir, "test_warped_panel.jpg")
    cv2.imwrite(raw_out, canvas)
    print(f"OK: Raw warped panel saved to: {raw_out}")

    # 7. Draw bounding boxes on the canvas for visual confirmation
    viz_canvas = canvas.copy()
    class_names = {0: 'MCB', 1: 'RCD', 2: 'RCD_SI', 3: 'MAINBREAKER', 4: 'OVERSURGE', 5: 'OTHER'}
    class_colors = {
        0: (0, 242, 254),   # PIA (Cyan)
        1: (0, 0, 255),     # RCD (Red)
        2: (255, 0, 255),   # RCD_SI (Magenta)
        3: (0, 255, 0),     # IGA (Green)
        4: (255, 255, 0),   # OVERSURGE (Yellow)
        5: (255, 255, 255)  # OTHER (White)
    }

    print(f"Total annotations detected in warped panel: {len(annotations)}")
    for i, ann in enumerate(annotations):
        cid = ann["class_id"]
        x1, y1 = ann["x"], ann["y"]
        x2, y2 = ann["x"] + ann["w"], ann["y"] + ann["h"]
        
        # Verify box constraints
        assert 0 <= x1 < resolution, f"Box x1 out of bounds: {x1}"
        assert 0 <= x2 <= resolution, f"Box x2 out of bounds: {x2}"
        assert 0 <= y1 < resolution, f"Box y1 out of bounds: {y1}"
        assert 0 <= y2 <= resolution, f"Box y2 out of bounds: {y2}"
        
        color = class_colors.get(cid, (255, 255, 255))
        cname = class_names.get(cid, "UNKNOWN")
        
        # Draw box
        cv2.rectangle(viz_canvas, (x1, y1), (x2, y2), color, 3)
        # Label class name
        cv2.putText(viz_canvas, f"{cname}", (x1 + 5, y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    viz_out = os.path.join(out_dir, "test_warped_panel_labels.jpg")
    cv2.imwrite(viz_out, viz_canvas)
    print(f"OK: Labeled warped panel saved to: {viz_out}")
    print("\nVisual verification complete! Open 'data/test_runs/test_warped_panel_labels.jpg' to inspect bounding boxes.")

if __name__ == "__main__":
    main()
