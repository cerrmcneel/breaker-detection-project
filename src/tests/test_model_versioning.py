import json
import os
from pathlib import Path

from src.tools.manage_model_version import register_model_version, rollback_model_version


def test_register_model_version(tmp_path):
    config_file = tmp_path / "pipeline_config.json"
    dummy_yolo = tmp_path / "yolo_v1.0.0.pt"
    dummy_yolo.write_text("weights")

    initial_config = {
        "model_version": "v1.0.0",
        "previous_version": None,
        "yolo_model_path": str(dummy_yolo),
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(initial_config, f)

    new_yolo = tmp_path / "new_yolo.pt"
    new_yolo.write_text("new weights")

    dest_models_dir = tmp_path / "models"
    repo_models_before = set(os.listdir("models")) if os.path.exists("models") else None

    updated = register_model_version(
        version_tag="v1.1.0",
        yolo_weight_path=str(new_yolo),
        config_path=str(config_file),
        models_dir=dest_models_dir,
    )

    assert updated["model_version"] == "v1.1.0"
    assert updated["previous_version"] == "v1.0.0"
    assert "v1.1.0" in updated["yolo_model_path"]
    assert "new_yolo_v1.1.0.pt" in updated["yolo_model_path"]

    # Verify target file was written to tmp_path destination
    target_path = Path(updated["yolo_model_path"])
    assert target_path.exists()
    assert target_path.read_text(encoding="utf-8") == "new weights"
    assert target_path.resolve().is_relative_to(tmp_path.resolve())

    # Assert nothing was written to the repo's models/ directory. Compare
    # before/after rather than asserting a specific file is absent: the
    # pre-fix code left a models/yolo26l_v1.1.0.pt stub on every run, so that
    # file may exist on any machine that ran the old tests, whatever this
    # test does.
    repo_models_after = set(os.listdir("models")) if os.path.exists("models") else None
    assert repo_models_after == repo_models_before


def test_register_model_version_with_crop_and_custom_model_name(tmp_path):
    config_file = tmp_path / "pipeline_config.json"
    initial_config = {
        "model_version": "v1.1.0",
        "previous_version": "v1.0.0",
        "yolo_model_path": "models/best.pt",
        "crop_model_path": "models/crop_classifier.pth",
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(initial_config, f)

    yolo_src = tmp_path / "best.pt"
    yolo_src.write_text("medium weights")

    crop_src = tmp_path / "crop_classifier.pth"
    crop_src.write_text("crop weights")

    dest_models_dir = tmp_path / "custom_models"
    repo_models_before = set(os.listdir("models")) if os.path.exists("models") else None

    updated = register_model_version(
        version_tag="v1.2.0",
        yolo_weight_path=str(yolo_src),
        crop_weight_path=str(crop_src),
        config_path=str(config_file),
        models_dir=dest_models_dir,
        model_name="yolo26m",
    )

    assert updated["model_version"] == "v1.2.0"
    assert updated["previous_version"] == "v1.1.0"
    assert "yolo26m_v1.2.0.pt" in updated["yolo_model_path"]
    assert "crop_classifier_v1.2.0.pth" in updated["crop_model_path"]

    # Verify both weight copies exist under tmp_path
    yolo_dest = Path(updated["yolo_model_path"])
    crop_dest = Path(updated["crop_model_path"])
    assert yolo_dest.exists() and yolo_dest.read_text(encoding="utf-8") == "medium weights"
    assert crop_dest.exists() and crop_dest.read_text(encoding="utf-8") == "crop weights"
    assert yolo_dest.resolve().is_relative_to(dest_models_dir.resolve())
    assert crop_dest.resolve().is_relative_to(dest_models_dir.resolve())

    # Assert repo models/ was untouched
    repo_models_after = set(os.listdir("models")) if os.path.exists("models") else None
    assert repo_models_after == repo_models_before


def test_register_model_version_nonexistent_weights(tmp_path):
    config_file = tmp_path / "pipeline_config.json"
    initial_config = {
        "model_version": "v1.0.0",
        "previous_version": None,
        "yolo_model_path": "models/best.pt",
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(initial_config, f)

    dest_models_dir = tmp_path / "models"
    updated = register_model_version(
        version_tag="v1.3.0",
        yolo_weight_path="nonexistent/path/model.pt",
        config_path=str(config_file),
        models_dir=dest_models_dir,
    )

    assert updated["model_version"] == "v1.3.0"
    assert updated["yolo_model_path"] == "nonexistent/path/model.pt"
    # Destination directory shouldn't be created if weights don't exist
    assert not dest_models_dir.exists()


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
