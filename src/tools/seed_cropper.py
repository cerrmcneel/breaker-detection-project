#!/usr/bin/env python3
"""
seed_cropper.py — Interactive OBB crop tool for the PanelSafe seed library.

Run from the project root:
    python -m src.tools.seed_cropper path/to/panel_photo.jpg
    python -m src.tools.seed_cropper path/to/photos_folder/

How it works
------------
1. The image opens in a fixed-size window. 
   - Scroll Wheel to ZOOM
   - Right-Click & Drag to PAN
   
2. Left-click the 4 corners of any breaker:
       Click 1 → Top-Left
       Click 2 → Top-Right
       Click 3 → Bottom-Right
       Click 4 → Bottom-Left

3. After the 4th click, press a class key to save the crop:
       M  →  MCB
       B  →  MAINBREAKER
       R  →  RCD
       S  →  RCD_SI
       O  →  OVERSURGE
       X  →  OTHER

4. The crop is perspective-corrected (via warp_crop) and saved to:
       data/seeds/<CLASS>/crop_NNNN.jpg

5. Other keys:
       U      →  Undo last point
       N / Space →  Skip to next image without saving current selection
       ESC / Q   →  Quit
"""

import glob
import os
import sys

import cv2
import numpy as np

from src.data_gen.extractor import SeedExtractor

# ── Configuration ──────────────────────────────────────────────────────────────
SEEDS_DIR = os.path.join("data", "seeds")
PROGRESS_FILE = os.path.join(SEEDS_DIR, "processed.txt")

CLASS_KEYS = {
    ord('m'): 'MCB',
    ord('M'): 'MCB',
    ord('r'): 'RCD',
    ord('R'): 'RCD',
    ord('s'): 'RCD_SI',
    ord('S'): 'RCD_SI',
    ord('b'): 'MAINBREAKER',
    ord('B'): 'MAINBREAKER',
    ord('o'): 'OVERSURGE',
    ord('O'): 'OVERSURGE',
    ord('x'): 'OTHER',
    ord('X'): 'OTHER',
}

# Visual style
POINT_COLOR  = (0, 255, 255)   
LINE_COLOR   = (0, 180, 255)   
TEXT_COLOR   = (255, 255, 255) 
SAVED_COLOR  = (80, 220, 80)   
POINT_RADIUS = 5

CANVAS_W = 1200
CANVAS_H = 800

# ── Global state ───────────────────────────────────────────────────────────────
clicks         = []    # (x, y) points in ORIGINAL full-resolution space
original_image = None  
save_counts    = {}    
extractor      = SeedExtractor()

# Viewport state
view_x = 0.0
view_y = 0.0
zoom   = 1.0
is_dragging = False
drag_start = (0, 0)

# ── Directory helpers ──────────────────────────────────────────────────────────
def ensure_seed_dirs():
    for cls in set(CLASS_KEYS.values()):
        os.makedirs(os.path.join(SEEDS_DIR, cls), exist_ok=True)

def count_existing_seeds():
    counts = {}
    for cls in set(CLASS_KEYS.values()):
        pattern = os.path.join(SEEDS_DIR, cls, "crop_*.jpg")
        files = glob.glob(pattern)
        max_idx = 0
        for f in files:
            try:
                # Extract NNNN from 'crop_NNNN.jpg'
                idx = int(os.path.basename(f)[5:9])
                if idx >= max_idx:
                    max_idx = idx + 1
            except ValueError:
                pass
        counts[cls] = max_idx
    return counts

def next_save_path(cls):
    n = save_counts.get(cls, 0)
    return os.path.join(SEEDS_DIR, cls, f"crop_{n:04d}.jpg")

# ── Mouse callback (Pan & Zoom) ────────────────────────────────────────────────
def on_mouse(event, x, y, flags, param):
    global clicks, view_x, view_y, zoom, is_dragging, drag_start

    # Map canvas pixel (x,y) to original image coordinate
    orig_x = (x / zoom) + view_x
    orig_y = (y / zoom) + view_y

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicks) < 4:
            clicks.append((int(orig_x), int(orig_y)))
            print(f"  Point {len(clicks)}: ({int(orig_x)}, {int(orig_y)})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        is_dragging = True
        drag_start = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_dragging:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            view_x -= dx / zoom
            view_y -= dy / zoom
            drag_start = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        is_dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0: # Scroll up -> zoom in
            zoom_factor = 1.15
        else:         # Scroll down -> zoom out
            zoom_factor = 1 / 1.15
            
        new_zoom = zoom * zoom_factor
        
        # Keep the original point under the cursor fixed
        view_x = orig_x - (x / new_zoom)
        view_y = orig_y - (y / new_zoom)
        zoom = new_zoom


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_hud(img, counts, n_clicks):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (520, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    hint = (
        f"[{n_clicks}/4 corners]  "
        + ("→ press M / B / R / S / O / X to save" if n_clicks == 4 else "L-Click TL→TR→BR→BL")
    )
    lines = [
        hint,
        "Controls: L-Click = Add Point  |  R-Click+Drag = Pan  |  Scroll = Zoom",
        "Keys:     M=MCB   R=RCD   S=RCD_SI   B=MAINBREAKER   O=OVER   X=OTHER",
        "          U=Undo  N/Space=Next  ESC/Q=Quit",
        "",
        f"Saved  MCB:{counts.get('MCB', 0):>3}   RCD:{counts.get('RCD', 0):>3}   SI:{counts.get('RCD_SI', 0):>3}   MAIN:{counts.get('MAINBREAKER', 0):>3}   OVER:{counts.get('OVERSURGE', 0):>3}   OTHER:{counts.get('OTHER', 0):>3}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (10, 22 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1, cv2.LINE_AA,
        )
    return img

def flash_preview(crop, cls):
    h, w = crop.shape[:2]
    size = 220
    preview = cv2.resize(crop, (size, size))
    cv2.putText(
        preview, f"SAVED: {cls}", (6, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, SAVED_COLOR, 2, cv2.LINE_AA,
    )
    cv2.imshow("Crop Preview", preview)
    cv2.waitKey(900)
    cv2.destroyWindow("Crop Preview")


# ── Per-image processing loop ──────────────────────────────────────────────────
def load_image(path):
    global original_image, clicks, view_x, view_y, zoom
    img = cv2.imread(path)
    if img is None:
        print(f"[SKIP] Cannot read: {path}")
        return False

    original_image = img
    clicks.clear()
    
    # Reset viewport to fit the image
    h, w = img.shape[:2]
    zoom = min(CANVAS_W / w, CANVAS_H / h)
    
    # Center image in canvas
    view_x = - (CANVAS_W / zoom - w) / 2
    view_y = - (CANVAS_H / zoom - h) / 2
    return True


def process_image(path, window_name):
    global clicks

    if not load_image(path):
        return

    cv2.setMouseCallback(window_name, on_mouse)
    print(f"\n[IMAGE] {os.path.basename(path)}  ({original_image.shape[1]}×{original_image.shape[0]}px)")

    while True:
        # 1. Warp the original image into the fixed-size canvas using the current pan/zoom
        M = np.array([[zoom, 0, -view_x * zoom], 
                      [0, zoom, -view_y * zoom]], dtype=np.float32)
        frame = cv2.warpAffine(original_image, M, (CANVAS_W, CANVAS_H))

        # 2. Draw clicks mapped to canvas coordinates
        canvas_pts = []
        for cx, cy in clicks:
            sx = int((cx - view_x) * zoom)
            sy = int((cy - view_y) * zoom)
            canvas_pts.append((sx, sy))
            cv2.circle(frame, (sx, sy), POINT_RADIUS, POINT_COLOR, -1)
        
        for i, (sx, sy) in enumerate(canvas_pts):
            cv2.putText(frame, str(i + 1), (sx + 10, sy - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

        if len(canvas_pts) > 1:
            for i in range(len(canvas_pts) - 1):
                cv2.line(frame, canvas_pts[i], canvas_pts[i + 1], LINE_COLOR, 2)
        if len(canvas_pts) == 4:
            cv2.line(frame, canvas_pts[3], canvas_pts[0], LINE_COLOR, 2)

        # 3. Draw HUD
        frame = draw_hud(frame, save_counts, len(clicks))
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            print("\n[QUIT]")
            cv2.destroyAllWindows()
            sys.exit(0)

        if key in (ord('u'), ord('U')):
            if clicks:
                removed = clicks.pop()
                print(f"  [UNDO] Removed point {removed}. {len(clicks)} remaining.")

        if key in (ord('n'), ord('N'), ord(' ')):
            print("  [NEXT] Moving to next image.")
            clicks.clear()
            with open(PROGRESS_FILE, "a") as f:
                f.write(os.path.abspath(path) + "\n")
            break

        if len(clicks) == 4 and key in CLASS_KEYS:
            cls = CLASS_KEYS[key]
            # clicks are already in original_image coords! No conversion needed!
            corners = np.array(clicks, dtype="float32")
            try:
                crop = extractor.warp_crop(original_image, corners)
                save_path = next_save_path(cls)
                cv2.imwrite(save_path, crop)
                save_counts[cls] = save_counts.get(cls, 0) + 1
                print(f"  [SAVED] {save_path}  ({crop.shape[1]}×{crop.shape[0]}px)")
                flash_preview(crop, cls)
            except Exception as e:
                print(f"  [ERROR] warp_crop failed: {e}")

            clicks.clear()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    ensure_seed_dirs()
    save_counts.update(count_existing_seeds())

    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python -m src.tools.seed_cropper <photo.jpg | folder/> ...")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
                paths.extend(sorted(glob.glob(os.path.join(arg, ext))))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"[WARN] Not found: {arg}")

    if not paths:
        print("[ERROR] No images found. Exiting.")
        sys.exit(1)

    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed = set(f.read().splitlines())
            
    paths = [p for p in paths if os.path.abspath(p) not in processed]
    
    if not paths:
        print("[DONE] All images in this folder have already been processed!")
        sys.exit(0)

    print(f"[START] {len(paths)} image(s) remaining in queue.")
    print(f"        Existing seeds: {save_counts}")

    window = "PanelSafe — Seed Cropper"
    # Use AUTOSIZE to prevent the OS from stretching the window
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    for path in paths:
        process_image(path, window)

    cv2.destroyAllWindows()
    print("\n[DONE] Session complete.")
    print(f"       Final counts: {save_counts}")

if __name__ == "__main__":
    main()
