import argparse
import os
import sys
from typing import Optional, Set

from ultralytics import YOLO


def get_supported_export_args(export_format: str) -> Set[str]:
    """Queries ultralytics Exporter for supported export arguments for a given format."""
    try:
        from ultralytics.engine.exporter import export_formats

        ef = export_formats()
        for fmt, args in zip(ef.get("Argument", ()), ef.get("Arguments", ())):
            if fmt.lower() == export_format.lower():
                return set(args)
    except Exception:
        pass

    # Built-in fallbacks based on ultralytics exporter specifications
    if export_format.lower() == "onnx":
        return {"batch", "dynamic", "half", "opset", "simplify", "nms"}
    return {"batch", "dynamic", "half", "int8", "nms"}


def export_model(
    weights_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    dynamic: bool = False,
    half: bool = False,
    int8: bool = False,
    simplify: bool = True,
    **extra_kwargs,
) -> str:
    """Exports a trained YOLO model to the specified format.

    Args:
        weights_path: Path to PyTorch model weights (.pt).
        format: Export target format (default: 'onnx').
        imgsz: Image size (pixels) for inference input.
        dynamic: Whether to use dynamic input shapes/axes.
        half: Enable FP16 half precision if supported.
        int8: Enable INT8 quantization if supported.
        simplify: Simplify ONNX model via onnxsim if supported.
        **extra_kwargs: Additional arguments passed to YOLO.export().

    Returns:
        The path to the exported model file as returned by YOLO.export().

    Raises:
        FileNotFoundError: If weights_path does not exist.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Loading weights from {weights_path}...")
    model = YOLO(weights_path)

    supported_args = get_supported_export_args(format)
    export_kwargs = {
        "format": format,
        "imgsz": imgsz,
    }

    if "dynamic" in supported_args:
        export_kwargs["dynamic"] = dynamic
    elif dynamic:
        print(f"Notice: Format '{format}' does not support --dynamic. Skipping.")

    if "half" in supported_args:
        if half:
            export_kwargs["half"] = True
    elif half:
        print(f"Notice: Format '{format}' does not support --half precision. Skipping.")

    if "int8" in supported_args:
        if int8:
            export_kwargs["int8"] = True
    elif int8:
        print(f"Notice: Format '{format}' does not support --int8 precision. Skipping.")

    if "simplify" in supported_args:
        export_kwargs["simplify"] = simplify

    export_kwargs.update(extra_kwargs)

    print(f"Exporting model to {format} (imgsz={imgsz}, kwargs={export_kwargs})...")
    exported_path = model.export(**export_kwargs)
    print(f"Exported model written to: {exported_path}")
    return str(exported_path)


def export_to_onnx(
    model_path: str,
    imgsz: int = 640,
    dynamic: bool = False,
    half: bool = False,
    simplify: bool = True,
    **kwargs,
) -> str:
    """Backwards-compatible wrapper for exporting a model to ONNX."""
    return export_model(
        weights_path=model_path,
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        half=half,
        simplify=simplify,
        **kwargs,
    )


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO model weights to ONNX or other deployment formats."
    )
    parser.add_argument(
        "--weights",
        "-w",
        required=True,
        type=str,
        help="Path to YOLO PyTorch weights (.pt) file to export.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels for export (default: 640).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="Export format (default: 'onnx').",
    )
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dynamic input shape/axes (default: False).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="FP16 half-precision export (if supported by format).",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        default=False,
        help="INT8 quantization (if supported by format).",
    )
    return parser.parse_args(args)


def main(args: Optional[list] = None) -> int:
    parsed = parse_args(args)

    if not os.path.isfile(parsed.weights):
        sys.stderr.write(f"Error: Weights file not found: {parsed.weights}\n")
        return 1

    try:
        exported_path = export_model(
            weights_path=parsed.weights,
            format=parsed.format,
            imgsz=parsed.imgsz,
            dynamic=parsed.dynamic,
            half=parsed.half,
            int8=parsed.int8,
        )
        print(f"Export successful: {exported_path}")
        return 0
    except Exception as err:
        sys.stderr.write(f"Error during export: {err}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
