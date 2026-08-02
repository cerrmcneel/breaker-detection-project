import json
import os
from src.tools.manage_model_version import register_model_version, rollback_model_version


def test_register_model_version(tmp_path):
    config_file = tmp_path / "pipeline_config.json"
    dummy_yolo = tmp_path / "yolo_v1.0.0.pt"
    dummy_yolo.write_text("weights")

    initial_config = {
        "model_version": "v1.0.0",
        "previous_version": None,
        "yolo_model_path": str(dummy_yolo)
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(initial_config, f)

    new_yolo = tmp_path / "new_yolo.pt"
    new_yolo.write_text("new weights")

    updated = register_model_version(
        version_tag="v1.1.0",
        yolo_weight_path=str(new_yolo),
        config_path=str(config_file)
    )

    assert updated["model_version"] == "v1.1.0"
    assert updated["previous_version"] == "v1.0.0"
    assert "v1.1.0" in updated["yolo_model_path"]


def test_rollback_model_version(tmp_path):
    config_file = tmp_path / "pipeline_config.json"
    v1_yolo = str(tmp_path / "yolo_v1.0.0.pt")
    v2_yolo = str(tmp_path / "yolo_v2.0.0.pt")

    config_data = {
        "model_version": "v2.0.0",
        "previous_version": "v1.0.0",
        "yolo_model_path": v2_yolo
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    rolled_back = rollback_model_version(
        target_version="v1.0.0",
        target_yolo_path=v1_yolo,
        config_path=str(config_file)
    )

    assert rolled_back["model_version"] == "v1.0.0"
    assert rolled_back["previous_version"] == "v2.0.0"
    assert rolled_back["yolo_model_path"] == v1_yolo
