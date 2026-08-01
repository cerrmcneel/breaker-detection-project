import json
import os

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from src.model.crop_classifier import BreakerCropClassifier
from src.model.heuristics import SpatialHeuristicEngine
from src.model.ocr_reader import OCRReader


class PanelSafePipeline:
    def __init__(self, config_path=None):
        if config_path is None:
            # Default path relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(project_root, "src", "model", "pipeline_config.json")
            
        self.config_path = config_path
        self.load_config()

        # Initialize devices
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize Models locally
        self.yolo_model = YOLO(self.config["yolo_model_path"])
        
        self.crop_classifier = None
        if self.config["classifier_mode"] == "two_stage":
            self.load_crop_classifier()

        # EasyOCR is heavy and only used when HMM/OCR is enabled. Lazy-load it so the
        # server still starts (degrading to no-OCR) if easyocr is absent in the env.
        self.reader = None
        self._ocr_unavailable = False

        # Local components
        self.heuristic_engine = SpatialHeuristicEngine()
        self.ocr_reader = OCRReader() # Reused for regex cleaning logic

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def load_crop_classifier(self):
        self.crop_classifier = BreakerCropClassifier()
        model_path = self.config["crop_model_path"]
        if os.path.exists(model_path):
            try:
                self.crop_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                self.crop_classifier.to(self.device)
                self.crop_classifier.eval()
            except Exception as e:
                print(f"Warning: Failed to load crop classifier weights from {model_path}: {e}")
        else:
            print(f"Warning: Crop classifier weights file not found at {model_path}. Running with uninitialized weights.")

    def _get_ocr_reader(self):
        """Lazy-init EasyOCR; return None (and stop retrying) if unavailable so
        inference degrades to no-OCR instead of failing at import/startup."""
        if self.reader is not None or self._ocr_unavailable:
            return self.reader
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
        except Exception as e:
            print(f"Warning: EasyOCR unavailable, continuing without OCR text: {e}")
            self._ocr_unavailable = True
        return self.reader

    def run_inference(self, image_path):
        """
        Runs the full configured inference pipeline on a local image file.
        """
        # 1. Run YOLO (Either 6-class detector or 1-class localizer)
        results = self.yolo_model.predict(
            image_path,
            conf=self.config["confidence_threshold"],
            imgsz=1280,
            device=self.device,
            verbose=False
        )

        predictions = []
        for r in results:
            for b in r.boxes:
                raw_box = b.xyxy[0]
                box = raw_box.tolist() if hasattr(raw_box, "tolist") else list(raw_box)
                conf = float(b.conf[0])
                pred_cls = r.names[int(b.cls[0])]
                
                predictions.append({
                    "box": box,
                    "class": pred_cls,
                    "conf": conf,
                    "ocr_text": ""
                })

        img = cv2.imread(image_path)
        if img is not None:
            h_img, w_img = img.shape[:2]
            
            # 2. Stage 2 Crop Classification (Strategy E)
            if self.config["classifier_mode"] == "two_stage" and self.crop_classifier:
                for pred in predictions:
                    # Extract coordinates safely
                    x1, y1, x2, y2 = map(int, pred["box"])
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, w_img - 1))
                    y1 = max(0, min(y1, h_img - 1))
                    x2 = max(0, min(x2, w_img - 1))
                    y2 = max(0, min(y2, h_img - 1))
                    
                    if x2 > x1 and y2 > y1:
                        crop_cv = img[y1:y2, x1:x2]
                        # Convert to PIL
                        crop_rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(crop_rgb)
                        
                        # Predict class and confidence using secondary classifier
                        new_cls, new_conf = self.crop_classifier.predict_image(pil_crop, device=self.device)
                        pred["class"] = new_cls
                        pred["conf"] = new_conf

            # 3. GPU-Native EasyOCR Reader (Strategy F Option B) - only run if HMM is enabled
            ocr_reader = self._get_ocr_reader() if self.config.get("use_hmm", True) else None
            if ocr_reader is not None:
                for pred in predictions:
                    if pred["class"] in ["MCB", "MAINBREAKER", "RCD", "RCD_SI", "OTHER"]:
                        # Crop coordinates safely
                        x1, y1, x2, y2 = map(int, pred["box"])
                        margin = 12
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(w_img, x2 + margin)
                        y2 = min(h_img, y2 + margin)
                        
                        if x2 > x1 and y2 > y1:
                            crop_cv = img[y1:y2, x1:x2]
                            crop_rgb = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB)
                            
                            # Read text using EasyOCR
                            try:
                                ocr_results = ocr_reader.readtext(crop_rgb)
                                raw_text = " ".join([res[1] for res in ocr_results])
                                # Clean text using existing regex patterns in ocr_reader
                                pred["ocr_text"] = self.ocr_reader._clean_ocr_text(raw_text)
                            except Exception as ocr_err:
                                print(f"Warning: EasyOCR error on crop: {ocr_err}")
                                pred["ocr_text"] = ""

        # 4. Refined layout via Spatial Heuristics & HMM Corrector
        refined_predictions = self.heuristic_engine.apply_logic(
            predictions, 
            image_path, 
            use_hmm=self.config["use_hmm"]
        )

        return refined_predictions
