import os
import random
import shutil


def mix_val_to_train(num_to_move=60):
    val_images_dir = "data/dataset/val/images"
    val_labels_dir = "data/dataset/val/labels"
    train_images_dir = "data/dataset/train/images"
    train_labels_dir = "data/dataset/train/labels"
    
    # Ensure all directories exist
    for d in [val_images_dir, val_labels_dir, train_images_dir, train_labels_dir]:
        if not os.path.exists(d):
            print(f"Error: Directory {d} does not exist.")
            return

    # Get all jpgs in val set
    val_images = [f for f in os.listdir(val_images_dir) if f.endswith(".jpg")]
    
    if len(val_images) <= num_to_move:
        print(f"Not enough validation images! Found {len(val_images)}, requested {num_to_move}.")
        return

    # Randomly select 60 images
    images_to_move = random.sample(val_images, num_to_move)
    
    success_count = 0
    for img_name in images_to_move:
        # Construct names
        label_name = img_name.replace(".jpg", ".txt")
        
        # Source paths
        src_img = os.path.join(val_images_dir, img_name)
        src_lbl = os.path.join(val_labels_dir, label_name)
        
        # Dest paths
        dst_img = os.path.join(train_images_dir, img_name)
        dst_lbl = os.path.join(train_labels_dir, label_name)
        
        # Only move if BOTH exist (to prevent orphans)
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.move(src_img, dst_img)
            shutil.move(src_lbl, dst_lbl)
            success_count += 1
            print(f"Moved {img_name} and its label to train.")
        else:
            print(f"Warning: Label missing for {img_name}, skipping.")
            
    print(f"\nSuccessfully moved {success_count} image-label pairs from Validation to Training.")
    print(f"Validation set now contains {len(val_images) - success_count} images.")

if __name__ == "__main__":
    mix_val_to_train()
