from unittest.mock import MagicMock, patch

import pytest

from src.model.export import (
    export_model,
    export_to_onnx,
    get_supported_export_args,
    main,
    parse_args,
)


def test_parse_args_defaults():
    args = parse_args(["--weights", "path/to/model.pt"])
    assert args.weights == "path/to/model.pt"
    assert args.imgsz == 640
    assert args.format == "onnx"
    assert args.dynamic is False
    assert args.half is False
    assert args.int8 is False


def test_parse_args_custom_values():
    args = parse_args(
        [
            "-w",
            "path/to/viewfinder.pt",
            "--imgsz",
            "320",
            "--format",
            "openvino",
            "--dynamic",
            "--half",
            "--int8",
        ]
    )
    assert args.weights == "path/to/viewfinder.pt"
    assert args.imgsz == 320
    assert args.format == "openvino"
    assert args.dynamic is True
    assert args.half is True
    assert args.int8 is True


def test_parse_args_no_dynamic():
    args = parse_args(["--weights", "path/to/model.pt", "--no-dynamic"])
    assert args.dynamic is False


def test_parse_args_missing_weights():
    with pytest.raises(SystemExit):
        parse_args([])


def test_main_missing_weights_file(tmp_path, capsys):
    nonexistent = tmp_path / "does_not_exist.pt"
    ret = main(["--weights", str(nonexistent)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Error: Weights file not found" in captured.err


def test_export_model_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError, match="Weights file not found"):
        export_model("nonexistent_model_weights.pt")


def test_get_supported_export_args():
    onnx_args = get_supported_export_args("onnx")
    assert "half" in onnx_args
    assert "dynamic" in onnx_args
    assert "simplify" in onnx_args
    assert "int8" not in onnx_args

    openvino_args = get_supported_export_args("openvino")
    assert "int8" in openvino_args
    assert "half" in openvino_args


@patch("src.model.export.YOLO")
def test_export_model_onnx_passes_expected_kwargs(mock_yolo_cls, tmp_path):
    dummy_weights = tmp_path / "best.pt"
    dummy_weights.write_text("dummy")

    mock_model_instance = MagicMock()
    mock_model_instance.export.return_value = str(tmp_path / "best.onnx")
    mock_yolo_cls.return_value = mock_model_instance

    res = export_model(
        weights_path=str(dummy_weights),
        format="onnx",
        imgsz=320,
        dynamic=False,
        half=True,
        int8=True,  # int8 not supported by onnx in ultralytics, should be excluded
        simplify=True,
    )

    mock_yolo_cls.assert_called_once_with(str(dummy_weights))
    mock_model_instance.export.assert_called_once()
    call_kwargs = mock_model_instance.export.call_args.kwargs
    assert call_kwargs["format"] == "onnx"
    assert call_kwargs["imgsz"] == 320
    assert call_kwargs["dynamic"] is False
    assert call_kwargs["half"] is True
    assert "int8" not in call_kwargs
    assert call_kwargs["simplify"] is True
    assert res == str(tmp_path / "best.onnx")


@patch("src.model.export.YOLO")
def test_export_model_openvino_passes_int8(mock_yolo_cls, tmp_path):
    dummy_weights = tmp_path / "best.pt"
    dummy_weights.write_text("dummy")

    mock_model_instance = MagicMock()
    mock_model_instance.export.return_value = str(tmp_path / "best_openvino_model")
    mock_yolo_cls.return_value = mock_model_instance

    export_model(
        weights_path=str(dummy_weights),
        format="openvino",
        imgsz=640,
        int8=True,
    )

    call_kwargs = mock_model_instance.export.call_args.kwargs
    assert call_kwargs["format"] == "openvino"
    assert call_kwargs["int8"] is True


@patch("src.model.export.export_model")
def test_export_to_onnx_wrapper(mock_export_model):
    mock_export_model.return_value = "exported.onnx"
    out = export_to_onnx("my_weights.pt", imgsz=320, dynamic=True)
    mock_export_model.assert_called_once_with(
        weights_path="my_weights.pt",
        format="onnx",
        imgsz=320,
        dynamic=True,
        half=False,
        simplify=True,
    )
    assert out == "exported.onnx"
