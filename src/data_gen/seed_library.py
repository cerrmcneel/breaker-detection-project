import os
import cv2
import random
import numpy as np


class SeedLibrary:
    """
    Manages seed images for synthetic panel generation.

    On load, classes with fewer than MIN_SEEDS_THRESHOLD images are
    automatically augmented to increase diversity.  This addresses
    the severe class imbalance (e.g. RCD_SI=16, OVERSURGE=30 seeds
    vs MCB=245).
    """

    MIN_SEEDS_THRESHOLD = 50   # Auto-augment classes below this count
    AUGMENT_FACTOR      = 4    # Number of augmented copies per original

    def __init__(self):
        self.seeds = {}
        self.backgrounds = []

    def load_seeds(self, seed_dir):
        for class_name in os.listdir(seed_dir):
            class_path = os.path.join(seed_dir, class_name)
            if os.path.isdir(class_path):
                images = []
                for filename in os.listdir(class_path):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                self.seeds[class_name] = images

        # Auto-augment underrepresented classes
        for cls, imgs in self.seeds.items():
            if 0 < len(imgs) < self.MIN_SEEDS_THRESHOLD:
                originals = list(imgs)  # snapshot before augmentation
                for orig in originals:
                    for _ in range(self.AUGMENT_FACTOR):
                        imgs.append(self._augment_seed(orig))

    def load_backgrounds(self, bg_dir):
        if not os.path.exists(bg_dir):
            return
        for filename in os.listdir(bg_dir):
            img_path = os.path.join(bg_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                self.backgrounds.append(img)

    def get_random_seed(self, cls, width=None):
        # Attempt to pull from width-specific folder (e.g., MCB_2)
        target_cls = f"{cls}_{width}" if width else cls
        
        if target_cls in self.seeds and self.seeds[target_cls]:
            return random.choice(self.seeds[target_cls])
            
        # Fallback 1: Try base class (if user hasn't audited yet)
        if cls in self.seeds and self.seeds[cls]:
            return random.choice(self.seeds[cls])
            
        # Fallback 2: Try ANY class starting with cls_ (if user audited but missing this specific width)
        available = [v for k, v in self.seeds.items() if k.startswith(f"{cls}_") and len(v) > 0]
        if available:
            # Flatten the list of lists and pick one
            all_available_seeds = [item for sublist in available for item in sublist]
            return random.choice(all_available_seeds)
            
        raise ValueError(f"No seeds found for class '{cls}' (tried '{target_cls}' too)")

    def get_random_background(self):
        if not self.backgrounds:
            return None
        return random.choice(self.backgrounds)

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _augment_seed(img):
        """
        Produce one augmented variant of a seed image.

        Augmentations applied (each with independent probability):
          - Horizontal flip       (50%)
          - Brightness/contrast   (60%)  ±20%
          - Gaussian blur         (30%)  kernel 3×3
          - Color temperature     (40%)  warm/cool shift

        Returns a new image; does not modify the input.
        """
        out = img.copy()

        # Horizontal flip — breakers are roughly symmetrical
        if random.random() < 0.5:
            out = cv2.flip(out, 1)

        # Brightness / contrast shift
        if random.random() < 0.6:
            alpha = 1.0 + random.uniform(-0.20, 0.20)  # contrast
            beta  = random.randint(-15, 15)             # brightness
            out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

        # Gaussian blur — simulates different camera focus
        if random.random() < 0.3:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        # Color temperature shift — warm (add red) or cool (add blue)
        if random.random() < 0.4:
            shift = random.randint(-12, 12)
            b, g, r = cv2.split(out)
            if shift > 0:
                r = np.clip(r.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            else:
                b = np.clip(b.astype(np.int16) - shift, 0, 255).astype(np.uint8)
            out = cv2.merge([b, g, r])

        return out