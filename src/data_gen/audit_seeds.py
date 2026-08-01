import os
import shutil

import cv2


def audit_seeds(seed_dir="data/seeds"):
    """
    Iterates through the seed library. Displays each image.
    User presses 1, 2, 3, or 4 to indicate pole/module width.
    Moves the image to a new width-specific folder (e.g. MCB_1, MAINBREAKER_2).
    """
    print("PanelSafe Seed Library Auditor")
    print("------------------------------")
    print("An image will appear. Press 1, 2, 3, or 4 on your keyboard to classify its width in modules.")
    print("Press 's' to skip an image. Press 'q' or ESC to quit.")
    print("Make sure the OpenCV image window is in focus when you press a key!\n")

    if not os.path.exists(seed_dir):
        print(f"Directory not found: {seed_dir}")
        return

    # Gather all images in base classes (e.g., MCB, MAINBREAKER, but skip already audited like MCB_1)
    base_classes = ["MCB", "MAINBREAKER", "OVERSURGE", "RCD", "RCD_SI", "OTHER"]
    
    for cls in base_classes:
        cls_dir = os.path.join(seed_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
            
        print(f"\nAuditing Class: {cls} ({len(images)} images)")
        
        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Resize for better visibility if it's too small
            h, w = img.shape[:2]
            if h < 200:
                scale = 300 / h
                display_img = cv2.resize(img, (int(w * scale), 300))
            else:
                display_img = img
                
            cv2.imshow("Seed Auditor (Press 1,2,3,4 | s=skip, q=quit)", display_img)
            
            key = cv2.waitKey(0) & 0xFF
            
            # 27 is ESC, 113 is 'q'
            if key in [27, 113]: 
                print("Exiting auditor.")
                cv2.destroyAllWindows()
                return
                
            # 115 is 's'
            if key == 115:
                print(f"Skipped {img_name}")
                continue
                
            # '1'=49, '2'=50, '3'=51, '4'=52
            width_str = None
            if key == 49:
                width_str = "1"
            elif key == 50:
                width_str = "2"
            elif key == 51:
                width_str = "3"
            elif key == 52:
                width_str = "4"
                
            if width_str:
                new_cls = f"{cls}_{width_str}"
                new_dir = os.path.join(seed_dir, new_cls)
                os.makedirs(new_dir, exist_ok=True)
                
                new_path = os.path.join(new_dir, img_name)
                shutil.move(img_path, new_path)
                print(f"Moved -> {new_cls}/{img_name}")
            else:
                print(f"Invalid key (pressed {key}). Image not moved.")
                
    cv2.destroyAllWindows()
    print("\nAudit Complete! You can now update compositor.py to use these new folders.")

if __name__ == "__main__":
    audit_seeds()
