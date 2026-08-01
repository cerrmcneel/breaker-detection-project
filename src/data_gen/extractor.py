import cv2
import numpy as np


class SeedExtractor:
    def warp_crop(self, image, corners):
        """
        Transforms a tilted OBB into a straight, upright crop.
        """
        # 1. Calculate the 'Real World' width and height of the tilted box
        # Using the norm (distance) between corner points
        w = int(np.linalg.norm(corners[0] - corners[1]))
        h = int(np.linalg.norm(corners[0] - corners[3]))
        
        # 2. Define the 'Ideal World' coordinates (The result we want)
        # We map the 4 corners to a flat rectangle starting at (0,0)
        dst_pts = np.array([
            [0, 0],         # Top-left maps to 0,0
            [w - 1, 0],     # Top-right
            [w - 1, h - 1], # Bottom-right
            [0, h - 1]      # Bottom-left
        ], dtype='float32')

        # 3. Get the Transformation Matrix (The 'M' Matrix)
    
        M = cv2.getPerspectiveTransform(corners, dst_pts)

        warped = cv2.warpPerspective(image, M, (w, h))

        return warped 

