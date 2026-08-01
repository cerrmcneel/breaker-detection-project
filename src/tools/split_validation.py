import os
import shutil


def split_validation_set():
    raw_dir = os.path.join("data", "images", "raw_uploads")
    val_img_dir = os.path.join("data", "dataset", "val", "images")
    val_lbl_dir = os.path.join("data", "dataset", "val", "labels")

    # Create directories if they don't exist
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    print(f"Scanning {raw_dir} for real-world images...")
    
    # Loop through all files in the raw uploads folder
    for filename in os.listdir(raw_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # We found an image! Let's define the paths
            img_src_path = os.path.join(raw_dir, filename)
            
            # The label has the same base name, but with a .txt extension
            base_name = os.path.splitext(filename)[0]
            lbl_src_path = os.path.join(raw_dir, f"{base_name}.txt")

            # TODO: Write the logic below to copy the image to val_img_dir
            # HINT: use shutil.copy()
            shutil.copy(img_src_path, val_img_dir)
            if os.path.exists(lbl_src_path):
                shutil.copy(lbl_src_path, val_lbl_dir)
                print(f"  > Synced label for {base_name}")
            else:
                print('Labels do not exist.')

if __name__ == "__main__":
    split_validation_set()
