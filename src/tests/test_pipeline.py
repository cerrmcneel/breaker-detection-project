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


@patch('src.model.pipeline.YOLO')
def test_pipeline_output_dictionary_schema(mock_yolo_class, mock_config):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
        json.dump(mock_config, f)
        config_path = f.name

    mock_yolo = MagicMock()
    mock_yolo_class.return_value = mock_yolo

    mock_box = MagicMock()
    mock_box.xyxy = [[15, 25, 120, 220]]
    mock_box.conf = [0.88]
    mock_box.cls = [0]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "MCB"}
    mock_yolo.predict.return_value = [mock_result]

    try:
        with patch('torch.load'):
            pipeline = PanelSafePipeline(config_path=config_path)
            pipeline._get_ocr_reader = MagicMock(return_value=None)
            pipeline.heuristic_engine.apply_logic = MagicMock(
                side_effect=lambda preds, *args, **kwargs: preds
            )

            import numpy as np
            mock_img = np.zeros((300, 300, 3), dtype=np.uint8)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_f:
                img_path = img_f.name
            cv2.imwrite(img_path, mock_img)

            try:
                results = pipeline.run_inference(img_path)
                assert isinstance(results, list)
                assert len(results) == 1
                pred = results[0]
                assert set(pred.keys()) == {"box", "class", "conf", "ocr_text"}
                assert isinstance(pred["box"], list) and len(pred["box"]) == 4
                assert isinstance(pred["class"], str)
                assert isinstance(pred["conf"], float) and 0.0 <= pred["conf"] <= 1.0
                assert isinstance(pred["ocr_text"], str)
            finally:
                os.remove(img_path)
    finally:
        os.remove(config_path)



# --- OCR gating -------------------------------------------------------------
# OCR text reading used to be gated on `use_hmm`, so disabling the HMM silently
# disabled text reading too (every ocr_text came back empty in production).
# `use_ocr` now controls it, falling back to the old coupled value when absent.

def _pipeline_with(config, tmp_path):
    config_path = tmp_path / "pipeline_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with patch('src.model.pipeline.YOLO'), patch('torch.load'):
        return PanelSafePipeline(config_path=str(config_path))


@pytest.mark.parametrize("config,expected", [
    ({"use_ocr": True, "use_hmm": False}, True),    # the production case after the split
    ({"use_ocr": False, "use_hmm": True}, False),   # explicit opt-out wins over use_hmm
    ({"use_hmm": True}, True),                      # legacy config: old coupled behaviour
    ({"use_hmm": False}, False),                    # legacy config: old coupled behaviour
    ({}, True),                                     # neither key: same default as before
])
def test_ocr_enabled_resolution(config, expected, mock_config, tmp_path):
    pipeline = _pipeline_with({**mock_config, **config}, tmp_path)
    assert pipeline.ocr_enabled() is expected


def test_ocr_runs_when_enabled_with_hmm_off(mock_config, tmp_path):
    """The whole point of the split: OCR reads text even though the HMM is off."""
    pipeline = _pipeline_with({**mock_config, "use_ocr": True, "use_hmm": False}, tmp_path)

    reader = MagicMock()
    reader.readtext.return_value = [([[25, 25], [45, 25], [45, 35], [25, 35]], "C16", 0.9)]
    pipeline._get_ocr_reader = MagicMock(return_value=reader)
    pipeline.heuristic_engine.apply_logic = MagicMock(
        side_effect=lambda preds, *args, **kwargs: preds
    )

    import numpy as np
    mock_box = MagicMock()
    mock_box.xyxy = [[10, 10, 60, 90]]
    mock_box.conf = [0.90]
    mock_box.cls = [0]
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "MCB"}
    pipeline.yolo_model.predict.return_value = [mock_result]

    img_path = tmp_path / "panel.jpg"
    cv2.imwrite(str(img_path), np.zeros((300, 300, 3), dtype=np.uint8))

    results = pipeline.run_inference(str(img_path))

    assert reader.readtext.called, "OCR should run with use_ocr=True even when use_hmm=False"
    assert results[0]["ocr_text"] == "C16"


def test_ocr_token_attribution_filters_margin_bleed(mock_config, tmp_path):
    """Tokens whose centre lies outside the unpadded box (margin bleed) are dropped."""
    pipeline = _pipeline_with({**mock_config, "use_ocr": True, "use_hmm": False}, tmp_path)

    # Box is [20, 20, 80, 100]. Crop expands by margin 12 -> [8, 8, 92, 112].
    # Relative original box in crop: x in [12, 72], y in [12, 92].
    # Token 1: inside box (center x=40, y=50) -> "C16"
    # Token 2: in margin bleed (center x=4, y=4, which is outside [12, 72]) -> "C32"
    reader = MagicMock()
    reader.readtext.return_value = [
        ([[2, 2], [6, 2], [6, 6], [2, 6]], "C32", 0.9),
        ([[35, 45], [45, 45], [45, 55], [35, 55]], "C16", 0.9),
    ]
    pipeline._get_ocr_reader = MagicMock(return_value=reader)
    pipeline.heuristic_engine.apply_logic = MagicMock(
        side_effect=lambda preds, *args, **kwargs: preds
    )

    import numpy as np
    mock_box = MagicMock()
    mock_box.xyxy = [[20, 20, 80, 100]]
    mock_box.conf = [0.90]
    mock_box.cls = [0]
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.names = {0: "MCB"}
    pipeline.yolo_model.predict.return_value = [mock_result]

    img_path = tmp_path / "panel.jpg"
    cv2.imwrite(str(img_path), np.zeros((300, 300, 3), dtype=np.uint8))

    results = pipeline.run_inference(str(img_path))

    # C32 from margin bleed is dropped; only C16 is read
    assert results[0]["ocr_text"] == "C16"


def test_ocr_skipped_when_disabled(mock_config, tmp_path):
    pipeline = _pipeline_with({**mock_config, "use_ocr": False, "use_hmm": False}, tmp_path)
    pipeline._get_ocr_reader = MagicMock()
    pipeline.heuristic_engine.apply_logic = MagicMock(
        side_effect=lambda preds, *args, **kwargs: preds
    )
    pipeline.yolo_model.predict.return_value = []

    import numpy as np
    img_path = tmp_path / "panel.jpg"
    cv2.imwrite(str(img_path), np.zeros((300, 300, 3), dtype=np.uint8))

    pipeline.run_inference(str(img_path))
    pipeline._get_ocr_reader.assert_not_called()
