import os
import random
import shutil

# Paths
RAW_DIR = os.path.join("data", "images", "raw_uploads")
TRAIN_IMG = os.path.join("data", "dataset", "train", "images")
TRAIN_LBL = os.path.join("data", "dataset", "train", "labels")
VAL_IMG = os.path.join("data", "dataset", "val", "images")
VAL_LBL = os.path.join("data", "dataset", "val", "labels")

def clean_and_split(val_split=0.2, seed=42):
    print("==================================================")
    print("PanelSafe Dataset Cleaning and Leakage Prevention")
    print("==================================================")

    # 1. Clean up dummy files in raw_uploads
    print("\n[Step 1] Cleaning up corrupted/dummy placeholder files...")
    removed_raw_count = 0
    if os.path.exists(RAW_DIR):
        raw_files = os.listdir(RAW_DIR)
        for f in raw_files:
            file_path = os.path.join(RAW_DIR, f)
            if not os.path.isfile(file_path):
                continue
                
            # Check for images with size < 100 bytes or text dummy content
            is_image = f.lower().endswith(('.jpg', '.jpeg', '.png'))
            if is_image:
                size = os.path.getsize(file_path)
                is_dummy = False
                
                # Check for tiny files
                if size < 100:
                    is_dummy = True
                else:
                    # Check if file has "dummy" content
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as tf:
                            first_line = tf.readline().strip().lower()
                            if "dummy" in first_line:
                                is_dummy = True
                    except Exception:
                        pass
                
                if is_dummy:
                    print(f"  Removing dummy image from raw_uploads: {f} ({size} bytes)")
                    os.remove(file_path)
                    removed_raw_count += 1
                    
                    # Also remove corresponding text label if it exists
                    basename = os.path.splitext(f)[0]
                    txt_path = os.path.join(RAW_DIR, basename + ".txt")
                    if os.path.exists(txt_path):
                        print(f"  Removing orphaned label: {basename}.txt")
                        os.remove(txt_path)
                        
    print(f"Cleaned up {removed_raw_count} dummy/placeholder files in raw_uploads.")

    # 2. Selectively purge real-world files from train/val (to prevent data leakage)
    print("\n[Step 2] Selectively purgying old real-world files from train/val splits...")
    purged_count = 0
    for img_dir, lbl_dir in [(TRAIN_IMG, TRAIN_LBL), (VAL_IMG, VAL_LBL)]:
        if os.path.exists(img_dir):
            for f in os.listdir(img_dir):
                # Only delete files that do NOT start with 'synth_panel_'
                if not f.startswith("synth_panel_"):
                    # Remove image
                    img_path = os.path.join(img_dir, f)
                    os.remove(img_path)
                    purged_count += 1
                    
                    # Remove matching label
                    basename = os.path.splitext(f)[0]
                    lbl_path = os.path.join(lbl_dir, basename + ".txt")
                    if os.path.exists(lbl_path):
                        os.remove(lbl_path)
                        
    print(f"Purged {purged_count} legacy real-world files from train/val to prevent leakage.")

    # 3. Gather all valid real-world images and labels from raw_uploads
    print("\n[Step 3] Gathering valid annotated images from raw_uploads...")
    valid_pairs = []
    if os.path.exists(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                basename = os.path.splitext(f)[0]
                img_path = os.path.join(RAW_DIR, f)
                lbl_path = os.path.join(RAW_DIR, basename + ".txt")
                
                if os.path.exists(lbl_path):
                    valid_pairs.append((img_path, lbl_path))
                    
    print(f"Found {len(valid_pairs)} valid, annotated real-world images.")

    if not valid_pairs:
        print("No valid real-world image-label pairs found. Split process complete.")
        return

    # 4. Perform a deterministic split
    print(f"\n[Step 4] Splitting real-world images (Split ratio: {1.0 - val_split:.2f} train / {val_split:.2f} val)...")
    random.seed(seed)
    random.shuffle(valid_pairs)
    
    val_count = int(len(valid_pairs) * val_split)
    train_pairs = valid_pairs[val_count:]
    val_pairs = valid_pairs[:val_count]
    
    print(f"  Training set size: {len(train_pairs)}")
    print(f"  Validation set size: {len(val_pairs)}")

    # 5. Copy files to train and val folders
    print("\n[Step 5] Copying files to train and val directories...")
    copied_train = 0
    copied_val = 0
    
    for img_path, lbl_path in train_pairs:
        # Create output directories if missing
        os.makedirs(TRAIN_IMG, exist_ok=True)
        os.makedirs(TRAIN_LBL, exist_ok=True)
        
        shutil.copy2(img_path, os.path.join(TRAIN_IMG, os.path.basename(img_path)))
        shutil.copy2(lbl_path, os.path.join(TRAIN_LBL, os.path.basename(lbl_path)))
        copied_train += 1

    for img_path, lbl_path in val_pairs:
        # Create output directories if missing
        os.makedirs(VAL_IMG, exist_ok=True)
        os.makedirs(VAL_LBL, exist_ok=True)
        
        shutil.copy2(img_path, os.path.join(VAL_IMG, os.path.basename(img_path)))
        shutil.copy2(lbl_path, os.path.join(VAL_LBL, os.path.basename(lbl_path)))
        copied_val += 1

    print(f"Successfully copied {copied_train} to train and {copied_val} to val.")
    
    # 6. Print final validation status
    print("\n==================================================")
    print("Verification Summary:")
    print("==================================================")
    
    # Count final counts including synthetic images
    final_train_img = len([f for f in os.listdir(TRAIN_IMG) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(TRAIN_IMG) else 0
    final_val_img = len([f for f in os.listdir(VAL_IMG) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(VAL_IMG) else 0
    
    print(f"Total train images (Synthetic + Real): {final_train_img}")
    print(f"Total val images (Synthetic + Real): {final_val_img}")
    
    # Verify no overlaps
    train_set = set(os.listdir(TRAIN_IMG)) if os.path.exists(TRAIN_IMG) else set()
    val_set = set(os.listdir(VAL_IMG)) if os.path.exists(VAL_IMG) else set()
    intersection = train_set.intersection(val_set)
    if intersection:
        print(f"WARNING: Found {len(intersection)} overlapping images in train and val!")
    else:
        print("PASS: Train and Val sets are completely distinct (No leakage).")
    print("==================================================")

if __name__ == "__main__":
    clean_and_split()
