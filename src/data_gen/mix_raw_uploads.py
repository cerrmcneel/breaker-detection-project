import os
import shutil
import glob
import random

def mix_raw_uploads(raw_dir="data/images/raw_uploads", output_dir="data/dataset", val_split=0.15,
                     deterministic_seed=None):
    """
    Copies pre-labeled real-world images and their YOLO .txt labels
    from the raw_uploads folder into the train and val directories.

    deterministic_seed: if set, assigns an EXACT val_split fraction (shuffled with this
    seed) instead of a per-image random.random() coin flip. At small batch sizes a
    Bernoulli trial per image has real variance around the target ratio; a pre-shuffle +
    slice guarantees the exact count and is reproducible.
    """
    print("PanelSafe Real-World Data Mixer")
    print("-------------------------------")

    if not os.path.exists(raw_dir):
        print(f"Directory not found: {raw_dir}")
        return

    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    if not image_files:
        print(f"No images found in {raw_dir}")
        return

    val_set = None
    if deterministic_seed is not None:
        shuffled = list(image_files)
        random.Random(deterministic_seed).shuffle(shuffled)
        n_val = round(len(shuffled) * val_split)
        val_set = set(shuffled[:n_val])
        print(f"Deterministic split (seed={deterministic_seed}): "
              f"{n_val} val / {len(shuffled) - n_val} train (of {len(shuffled)} images)")

    processed_train = 0
    processed_val = 0

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(raw_dir, base_name + ".txt")

        # Only process if it has a label file and it's not a tiny/corrupted image
        if os.path.exists(txt_path) and os.path.getsize(img_path) > 100:

            # Determine split
            is_val = (img_path in val_set) if val_set is not None else (random.random() < val_split)
            split_dir = "val" if is_val else "train"
            
            img_out_dir = os.path.join(output_dir, split_dir, "images")
            lbl_out_dir = os.path.join(output_dir, split_dir, "labels")
            
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(lbl_out_dir, exist_ok=True)
            
            # Destination paths
            dest_img = os.path.join(img_out_dir, os.path.basename(img_path))
            dest_txt = os.path.join(lbl_out_dir, base_name + ".txt")
            
            # Copy files
            shutil.copy2(img_path, dest_img)
            shutil.copy2(txt_path, dest_txt)
            
            if is_val:
                processed_val += 1
            else:
                processed_train += 1

    print(f"Successfully mixed {processed_train} train and {processed_val} val images into {output_dir}")

if __name__ == "__main__":
    mix_raw_uploads()
