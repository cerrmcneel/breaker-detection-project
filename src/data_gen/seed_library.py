import os
import cv2
import random

class SeedLibrary:
    def __init__(self):
        self.seeds = {}
        self.backgrounds = []

    def load_seeds(self, seed_dir):
        for class_name in os.listdir(seed_dir):
            class_path = os.path.join(seed_dir, class_name)
            if os.path.isdir(class_path):
                images = []
                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                self.seeds[class_name] = images

    def load_backgrounds(self, bg_dir):
        if not os.path.exists(bg_dir):
            return
        for filename in os.listdir(bg_dir):
            img_path = os.path.join(bg_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                self.backgrounds.append(img)

    def get_random_seed(self, cls):
        return random.choice(self.seeds[cls])

    def get_random_background(self):
        if not self.backgrounds:
            return None
        return random.choice(self.backgrounds)