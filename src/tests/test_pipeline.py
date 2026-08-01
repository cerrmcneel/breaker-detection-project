import json
import os
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import pytest

from src.model.pipeline import PanelSafePipeline


@pytest.fixture
def mock_config():
    return {
        "classifier_mode": "single_stage",
        "yolo_model_path": "models/best.pt",
        "crop_model_path": "models/crop_classifier.pth",
        "use_hmm": True,
        "use_button_detector": False,
        "confidence_threshold": 0.20
    }

def test_pipeline_initialization_loads_config(mock_config):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
        json.dump(mock_config, f)
        config_path = f.name

    try:
        # Patch the model loading inside pipeline to prevent loading real weights in unit test.
        # EasyOCR is NOT patched here: it is lazy-imported inside _get_ocr_reader(), so
        # __init__ never touches it (see pipeline.py).
        with patch('src.model.pipeline.YOLO'), patch('torch.load'):
            pipeline = PanelSafePipeline(config_path=config_path)
            assert pipeline.config["classifier_mode"] == "single_stage"
            assert pipeline.config["use_hmm"] is True
    finally:
        os.remove(config_path)

@patch('src.model.pipeline.YOLO')
def test_single_stage_inference_routing(mock_yolo_class, mock_config):
    # Setup mock config file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
        json.dump(mock_config, f)
        config_path = f.name

    # Mock YOLO predictions
    mock_yolo = MagicMock()
    mock_yolo_class.return_value = mock_yolo
    
    mock_box = MagicMock()
    mock_box.xyxy = [[100, 50, 180, 200]]
    mock_box.conf = [0.90]
    mock_box.cls = [0] # Class index for MCB
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "MCB"}
    mock_yolo.predict.return_value = [mock_result]

    try:
        with patch('torch.load'):
            pipeline = PanelSafePipeline(config_path=config_path)

            # Force the supported no-OCR degrade path so this routing test does not
            # depend on easyocr being installed (it is lazy-imported by design).
            pipeline._get_ocr_reader = MagicMock(return_value=None)

            # Neutralize the heuristic/HMM layer so this test isolates run_inference's
            # ROUTING. The HMM lives on SpatialHeuristicEngine (heuristics.py), not on
            # the pipeline -- it is exercised separately in test_hmm.py.
            pipeline.heuristic_engine.apply_logic = MagicMock(
                side_effect=lambda preds, *args, **kwargs: preds
            )

            # Run inference on a mock file
            import numpy as np
            mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                img_path = img_f.name
            cv2.imwrite(img_path, mock_img)
                
            try:
                results = pipeline.run_inference(img_path)
                assert len(results) == 1
                assert results[0]["class"] == "MCB"
                # Check that yolo was called
                mock_yolo.predict.assert_called_once()
            finally:
                os.remove(img_path)
    finally:
        os.remove(config_path)

@patch('src.model.pipeline.YOLO')
def test_two_stage_inference_routing(mock_yolo_class, mock_config):
    # Setup two stage config
    two_stage_config = mock_config.copy()
    two_stage_config["classifier_mode"] = "two_stage"
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
        json.dump(two_stage_config, f)
        config_path = f.name

    # Mock YOLO predictions (it returns a localized "Breaker" box)
    mock_yolo = MagicMock()
    mock_yolo_class.return_value = mock_yolo
    
    mock_box = MagicMock()
    # Mock coordinates that are inside a 100x100 mock image
    mock_box.xyxy = [[10, 10, 90, 90]]
    mock_box.conf = [0.85]
    mock_box.cls = [0]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "Breaker"}
    mock_yolo.predict.return_value = [mock_result]

    try:
        with patch('torch.load'):
            pipeline = PanelSafePipeline(config_path=config_path)

            # Setup mock crop classifier
            pipeline.crop_classifier = MagicMock()
            pipeline.crop_classifier.predict_image.return_value = ("RCD", 0.95)

            # Force the supported no-OCR degrade path (easyocr is lazy-imported).
            pipeline._get_ocr_reader = MagicMock(return_value=None)

            # Neutralize the heuristic/HMM layer so the crop-classifier overwrite is
            # what this test actually asserts on. See note in the single-stage test.
            pipeline.heuristic_engine.apply_logic = MagicMock(
                side_effect=lambda preds, *args, **kwargs: preds
            )

            # Create a mock 100x100 image
            import numpy as np
            mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                img_path = img_f.name
            cv2.imwrite(img_path, mock_img)
                
            try:
                results = pipeline.run_inference(img_path)
                assert len(results) == 1
                # Confirm the class was updated from "Breaker" to "RCD" by the crop classifier
                assert results[0]["class"] == "RCD"
                assert results[0]["conf"] == 0.95
                pipeline.crop_classifier.predict_image.assert_called_once()
            finally:
                os.remove(img_path)
    finally:
        os.remove(config_path)
