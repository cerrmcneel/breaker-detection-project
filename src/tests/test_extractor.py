import pytest
import numpy as np
import cv2
import os

def test_seed_extraction_aspect_ratio():
    """
    Test that the extractor produces an upright image 
    matching the aspect ratio of the input coordinates.
    """
    from src.data_gen.extractor import SeedExtractor
    
    # 1. Create a dummy "Tilted" 100x200 rectangle in a 500x500 image
    dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Define 4 corners of a tilted box (x, y)
    corners = np.array([
        [100, 100], # Top-left
        [200, 120], # Top-right (Vector 100, 20)
        [160, 320], # Bottom-right
        [60, 300]   # Bottom-left (Vector -40, 200)
    ], dtype=np.float32)
    
    extractor = SeedExtractor()
    
    # 2. Extract the seed
    seed = extractor.warp_crop(dummy_img, corners)
    
    # 3. Verify the aspect ratio is roughly 0.5 (Width/Height)
    h, w = seed.shape[:2]
    aspect_ratio = w / h
    
    # Allow for minor rounding errors in the warp math
    assert 0.45 < aspect_ratio < 0.55, f"Incorrect aspect ratio: {aspect_ratio}"
