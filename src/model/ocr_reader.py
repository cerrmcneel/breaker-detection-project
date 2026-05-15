import cv2
import numpy as np
import pytesseract
import re

# Point PyTesseract to the Winget installation path on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRReader:
    def __init__(self, lang='eng'):
        """
        Initializes the PyTesseract engine wrapper.
        """
        print("Initializing Tesseract OCR Engine...")
        self.lang = lang
        print("Tesseract OCR Engine Ready!")
        
    def read_bounding_box(self, img_path, box, margin=12):
        """
        Crops an image based on YOLO coordinates and extracts text using OCR.
        
        Args:
            img_path (str): Path to the full image.
            box (list): [x1, y1, x2, y2]
            margin (int): Extra pixel padding around the box to avoid cutting off text edges.
            
        Returns:
            str: Extracted text string.
        """
        img = cv2.imread(img_path)
        if img is None:
            return "ERROR: Image not found"
            
        # Ensure integer coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Apply padding while respecting image boundaries
        h, w = img.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        # Crop the specific breaker
        cropped_breaker = img[y1:y2, x1:x2]
        
        # --- VISION SHARPENING PIPELINE (v2 — optimised for field photos) ---

        # 1. Convert to grayscale
        gray = cv2.cvtColor(cropped_breaker, cv2.COLOR_BGR2GRAY)
        
        # 2. Dynamic Upscaling
        # Tesseract performs best when characters are ~30-50px tall.
        # We target a cropped image height of ~200px for optimal balance.
        crop_h, crop_w = gray.shape
        target_h = 200
        
        if crop_h < target_h:
            scale_factor = target_h / crop_h
            # Use Lanczos interpolation for high-quality edge preservation
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)

        # 3. CLAHE — handles uneven lighting inside panel enclosures
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 4. Deskewing — correct slight text rotation from camera angle
        gray = self._deskew(gray)
        
        # 5. Gentle Unsharp Mask (replaces the aggressive 3×3 sharpening kernel)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # 6. Otsu's Binarization (primary)
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 7. Adaptive Thresholding (fallback for difficult lighting)
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Try multiple PSM modes, starting with the most specific
        # PSM 7 = single text line (ideal for "C16" style labels)
        # PSM 13 = raw line, treats image as single character (fallback)
        # PSM 6 = uniform block (original, broadest)
        for psm, binary in [(7, binary_otsu), (7, binary_adaptive),
                            (13, binary_otsu), (6, binary_otsu)]:
            config = f'--oem 3 --psm {psm}'
            raw_text = pytesseract.image_to_string(binary, lang=self.lang, config=config)
            cleaned = self._clean_ocr_text(raw_text)
            if cleaned:
                return cleaned
        
        return ""

    @staticmethod
    def _deskew(img):
        """
        Detect dominant text angle and rotate to straighten.
        Uses minAreaRect on thresholded contours.
        """
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 10:
            return img  # Not enough content to deskew
        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect returns angles in [-90, 0); normalize
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return img  # Already straight enough
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _clean_ocr_text(self, text):
        """
        Cleans raw OCR output to return only expected electrical values.
        Looks for MCB ratings (e.g., C16) and 'SI' marker.
        Returns empty string if no valid pattern is found to prevent garbage output.
        """
        text = text.upper().strip()
        
        # Check for SI marker
        if "SI" in text:
            return "SI"
            
        # Check for B, C, or D rating (1 or 2 digits)
        # We allow an optional curve letter B, C, or D
        match = re.search(r'([B|C|D])\s?(\d{1,2})', text)
        if match:
            return f"{match.group(1)}{match.group(2)}"
            
        # Fallback: if we just find a standalone 10, 16, 20, 25, 32, 40, 63 (common amperages)
        # We'll assume it's a C-curve for now if no letter is found
        amp_match = re.search(r'\b(10|16|20|25|32|40|63)\b', text)
        if amp_match:
            return f"C{amp_match.group(1)}"
            
        return ""

if __name__ == "__main__":
    # Simple test initialization
    reader = OCRReader()
    print("OCR Module successfully loaded. Waiting for YOLO crop testing...")
