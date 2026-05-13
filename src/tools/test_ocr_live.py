import cv2
import os
from src.model.ocr_reader import OCRReader

def test_ocr_on_validation_image():
    # Setup paths
    img_path = "data/dataset/val/images/SPAIN_66e65cb2-f410-4140-86c8-e11439746fc6.jpg"
    label_path = "data/dataset/val/labels/SPAIN_66e65cb2-f410-4140-86c8-e11439746fc6.txt"
    
    if not os.path.exists(img_path) or not os.path.exists(label_path):
        print("Error: Test image or label not found.")
        return

    # Load image to get dimensions
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    print(f"Loaded image: {img_path} ({w}x{h})")

    # Initialize OCR
    reader = OCRReader()

    # Read labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    print(f"Found {len(lines)} breakers to test OCR on.\n")

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5: continue
        
        cls, x_center, y_center, width, height = map(float, parts)
        
        # Convert normalized to pixel coordinates [x1, y1, x2, y2]
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        box = [x1, y1, x2, y2]
        
        # Run OCR
        text = reader.read_bounding_box(img_path, box)
        
        print(f"Breaker {i+1} (Class {int(cls)}):")
        print(f"  OCR Result: '{text}'")
        print("-" * 20)

if __name__ == "__main__":
    test_ocr_on_validation_image()
