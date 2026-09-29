import json
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import mlflow
from mlflow.tracking import MlflowClient


def load_config(config_path: str = "src/model/pipeline_config.json") -> dict:
    """Loads current pipeline configuration dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict, config_path: str = "src/model/pipeline_config.json") -> None:
    """Saves updated configuration dictionary atomically."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def register_model_version(
    version_tag: str,
    yolo_weight_path: str,
    crop_weight_path: Optional[str] = None,
    config_path: str = "src/model/pipeline_config.json",
    run_id: Optional[str] = None,
    models_dir: Union[str, Path] = "models",
    model_name: Optional[str] = None,
) -> dict:
    """
    Registers a new model version tag in config and MLflow Model Registry.
    Copies weight files to version-tagged paths in models_dir.

    Args:
        version_tag: Release version string, e.g. "v1.1.0".
        yolo_weight_path: Path to YOLO model weights file.
        crop_weight_path: Optional path to crop classifier weights file.
        config_path: Path to pipeline config JSON file.
        run_id: Optional MLflow run ID for Model Registry tagging.
        models_dir: Destination directory for versioned weight copies (default: "models").
        model_name: Optional family/model stem name (e.g. "yolo26m"). If None,
            derived from yolo_weight_path filename stem.
    """
    config = load_config(config_path)

    # 1. Update version history pointers
    old_version = config.get("model_version", "v1.0.0")
    config["previous_version"] = old_version
    config["model_version"] = version_tag

    dest_dir = Path(models_dir)

    # 2. Store versioned weight file
    if os.path.exists(yolo_weight_path):
        dest_dir.mkdir(parents=True, exist_ok=True)
        yolo_stem = model_name or Path(yolo_weight_path).stem
        yolo_ext = Path(yolo_weight_path).suffix or ".pt"
        target_yolo = str(dest_dir / f"{yolo_stem}_{version_tag}{yolo_ext}")
        if os.path.abspath(yolo_weight_path) != os.path.abspath(target_yolo):
            shutil.copy(yolo_weight_path, target_yolo)
        config["yolo_model_path"] = target_yolo
    else:
        config["yolo_model_path"] = yolo_weight_path

    if crop_weight_path:
        if os.path.exists(crop_weight_path):
            dest_dir.mkdir(parents=True, exist_ok=True)
            crop_stem = Path(crop_weight_path).stem
            crop_ext = Path(crop_weight_path).suffix or ".pth"
            target_crop = str(dest_dir / f"{crop_stem}_{version_tag}{crop_ext}")
            if os.path.abspath(crop_weight_path) != os.path.abspath(target_crop):
                shutil.copy(crop_weight_path, target_crop)
            config["crop_model_path"] = target_crop
        else:
            config["crop_model_path"] = crop_weight_path

    # 3. Optional MLflow Model Registry Tagging
    if run_id:
        try:
            model_uri = f"runs:/{run_id}/weights/best.pt"
            mv = mlflow.register_model(model_uri, "PanelSafe-YOLO")
            client = MlflowClient()
            client.set_model_version_tag("PanelSafe-YOLO", mv.version, "release_version", version_tag)
        except Exception as err:
            print(f"Notice: MLflow registry tagging skipped or unreachable: {err}")

    save_config(config, config_path)
    return config


def rollback_model_version(
    target_version: str,
    target_yolo_path: str,
    target_crop_path: Optional[str] = None,
    config_path: str = "src/model/pipeline_config.json"
) -> dict:
    """
    Rolls back pipeline configuration to target model version and weight paths.
    """
    config = load_config(config_path)
    current_version = config.get("model_version", "v1.0.0")

    config["previous_version"] = current_version
    config["model_version"] = target_version
    config["yolo_model_path"] = target_yolo_path
    if target_crop_path:
        config["crop_model_path"] = target_crop_path

    save_config(config, config_path)
    return config
