import os
import re

import cv2
import numpy as np
import pytesseract

# Point PyTesseract to the Winget installation path on Windows if it exists
if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

    # Standard IEC 60898 MCB current ratings. Anything outside this set is a
    # misread or a fragment of a model number, not a real rating.
    VALID_RATINGS = {1, 2, 3, 4, 6, 10, 13, 16, 20, 25, 32, 40, 50, 63}

    # A curve letter immediately followed by a valid rating, as a standalone token.
    # Both \b anchors matter: without the leading one, the model number "BD62"
    # yields a bogus "D62"; without the trailing one, "C1234" yields "C12".
    _RATING_RE = re.compile(r'\b([BCD])\s?(\d{1,2})\b')

    # RCD residual-current marker for a 30 mA device. Deliberately NOT \b-anchored:
    # real OCR output runs the marker into neighbouring glyphs ("IAN0,03A",
    # "30 MAL"), and a \b(...)\b form silently missed every one of those.
    #
    # Digit lookarounds instead: they keep the leading-letter tolerance while
    # refusing to match inside a longer number, so a model number like "130MA"
    # no longer yields "30MA".
    #
    # The decimal branch requires BOTH zeros. An earlier `0[.,]0?3` also matched
    # "0.3A" -- which is a 300 mA fire-protection RCD, a genuinely different
    # device -- and would have emitted a 30 mA verdict for it at 19:1 odds.
    # Not observed in the 1,060-crop sample, but wrong by construction.
    _LEAKAGE_RE = re.compile(r'(?<!\d)(?:30\s*M\s*A|0[.,]03\s*A)(?!\d)')

    # "SI" = superinmunizado (nuisance-trip-immune RCD). Must be a STANDALONE token:
    # a bare substring test matches SIEMENS, Schneider RESI9, SIMON and a long tail
    # of OCR garble (EBSIN, MNSI6V, JENSION...). Measured 2026-07-28 over 1,060 real
    # crops, the substring form was wrong 58/58 times. Word-boundary matching rejects
    # all of those while still accepting the real forms "SI", "A-SI" and "A[SI]".
    _SI_RE = re.compile(r'\bSI\b')

    def _clean_ocr_text(self, text):
        """
        Reduces raw OCR output to a single electrical verdict the HMM can consume:
        "SI", a curve+rating like "C16", "30MA", or "" for no usable signal.

        Resolves by STRENGTH of evidence, not by first match. Ordering matters
        because the verdict carries a 19:1 likelihood ratio in the HMM's emission
        probability, so a confidently-wrong verdict is far more damaging than none.
        """
        text = (text or "").upper().strip()

        si = bool(self._SI_RE.search(text))
        leakage = bool(self._LEAKAGE_RE.search(text))

        rating = None
        for match in self._RATING_RE.finditer(text):
            amps = int(match.group(2))
            if amps in self.VALID_RATINGS:
                # Emit the parsed integer, not the raw digits, so a zero-padded
                # OCR read ("C06") canonicalises to "C6" rather than leaking a
                # second spelling of the same rating downstream.
                rating = f"{match.group(1)}{amps}"
                break

        # A superinmunizado device is still an RCD, so a standalone SI token
        # corroborated by a leakage marker is the genuine RCD_SI signature.
        if si and leakage:
            return "SI"

        # Leakage outranks a curve+rating when both appear. "30mA" only ever occurs
        # on a residual-current device, whereas a C16 in the same crop is usually
        # bleed from a neighbour: RCDs are 2+ modules wide and sit at the head of a
        # rail, so their crops routinely catch the adjacent MCB's marking, while a
        # 1-module MCB crop mostly bleeds into other MCBs. Measured 2026-07-28, every
        # crop carrying both markers was a true RCD.
        if leakage:
            return "30MA"

        # A curve LETTER is the MCB-specific discriminator. A bare amperage is not:
        # RCDs carry current ratings too ("40A"), which is why the old bare-amperage
        # fallback was removed -- it fired on model numbers ("NL1-63" -> "C63") and
        # pushed true RCDs toward MCB.
        if rating:
            return rating

        # Deliberately NOT returning "SI" here. An uncorroborated two-character token
        # is not enough evidence for the rarest class in the taxonomy at 19:1 leverage,
        # and across 1,060 real crops there was not a single box where a standalone SI
        # was the only available signal -- so this costs no measurable recall.
        return ""

if __name__ == "__main__":
    # Simple test initialization
    reader = OCRReader()
    print("OCR Module successfully loaded. Waiting for YOLO crop testing...")
