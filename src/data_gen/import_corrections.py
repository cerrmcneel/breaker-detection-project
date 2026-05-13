import os
import json
import cv2
import glob
import random

# Mapping must match compositor.py
CLASS_MAP = {
    "MCB":         0,
    "RCD":         1,
    "RCD_SI":      2,
    "MAINBREAKER": 3,
    "OVERSURGE":   4,
    "OTHER":       5,
}

def import_corrections(active_learning_dir="data/active_learning", output_dir="data/dataset", val_split=0.15):
    """
    Reads exported JSON files and corresponding images from active_learning_dir.
    Randomly splits them between output_dir/train and output_dir/val based on val_split.
    """
    json_files = glob.glob(os.path.join(active_learning_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {active_learning_dir}. Use the dashboard's 'Send to Vault' button.")
        return

    processed_train = 0
    processed_val = 0
    
    for json_path in json_files:
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Look for corresponding image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            possible = os.path.join(active_learning_dir, base_name + ext)
            if os.path.exists(possible):
                img_path = possible
                break
                
        if not img_path:
            print(f"Warning: Found {json_path} but no matching image. Skipping.")
            continue
            
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_h, img_w = img.shape[:2]
        
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        
        yolo_lines = []
        for item in data:
            cls_name = item.get("class", "MCB").upper()
            box = item.get("box", [])
            if len(box) != 4: continue
                
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            bw, bh = x2 - x1, y2 - y1
            nx, ny = (x1 + bw/2.0)/img_w, (y1 + bh/2.0)/img_h
            nw, nh = bw/img_w, bh/img_h
            
            cls_id = CLASS_MAP.get(cls_name, 0)
            yolo_lines.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
            
        if yolo_lines:
            # Decide split
            is_val = random.random() < val_split
            split_dir = "val" if is_val else "train"
            
            img_out = os.path.join(output_dir, split_dir, "images")
            lbl_out = os.path.join(output_dir, split_dir, "labels")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)
            
            with open(os.path.join(lbl_out, base_name + ".txt"), 'w') as f:
                f.write("\n".join(yolo_lines))
                
            cv2.imwrite(os.path.join(img_out, os.path.basename(img_path)), img)
            
            if is_val:
                processed_val += 1
            else:
                processed_train += 1
            print(f"Imported [{split_dir.upper()}]: {base_name}")
            
    print(f"\nSuccessfully imported {processed_train} train and {processed_val} val images to {output_dir}")

if __name__ == "__main__":
    print("PanelSafe Active Learning Importer")
    print("----------------------------------")
    os.makedirs("data/active_learning", exist_ok=True)
    import_corrections()
