import os
import random
import shutil

src_dir = r"data\images\raw_uploads"
train_img_dir = r"data\dataset\train\images"
train_lbl_dir = r"data\dataset\train\labels"
val_img_dir = r"data\dataset\val\images"
val_lbl_dir = r"data\dataset\val\labels"

# Get all unique basenames
files = [f for f in os.listdir(src_dir) if f.endswith(".jpg")]
random.shuffle(files)

# 80/20 split
split_idx = int(len(files) * 0.8)
train_files = files[:split_idx]
val_files = files[split_idx:]

for f in train_files:
    basename = f[:-4]
    shutil.copy(os.path.join(src_dir, f), os.path.join(train_img_dir, f))
    if os.path.exists(os.path.join(src_dir, basename + ".txt")):
        shutil.copy(os.path.join(src_dir, basename + ".txt"), os.path.join(train_lbl_dir, basename + ".txt"))

for f in val_files:
    basename = f[:-4]
    shutil.copy(os.path.join(src_dir, f), os.path.join(val_img_dir, f))
    if os.path.exists(os.path.join(src_dir, basename + ".txt")):
        shutil.copy(os.path.join(src_dir, basename + ".txt"), os.path.join(val_lbl_dir, basename + ".txt"))

print(f"Copied {len(train_files)} to train and {len(val_files)} to val.")
