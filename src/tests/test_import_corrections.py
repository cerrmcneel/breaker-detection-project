import json

import cv2
import numpy as np

from src.data_gen.import_corrections import import_corrections

ANNOTATIONS = [{"class": "RCD", "box": [2, 2, 10, 10]}]


def _write_pair(directory, name, sidecar):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    assert cv2.imwrite(str(directory / f"{name}.jpg"), img)
    (directory / f"{name}.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _run(tmp_path):
    source, out = tmp_path / "active_learning", tmp_path / "dataset"
    import_corrections(str(source), str(out), val_split=0.0)
    return {p.stem for p in (out / "train" / "labels").glob("*.txt")} if out.exists() else set()


def test_imports_the_bare_list_written_before_the_vault_asked_for_consent(tmp_path):
    (tmp_path / "active_learning").mkdir()
    _write_pair(tmp_path / "active_learning", "correction_old", ANNOTATIONS)

    assert _run(tmp_path) == {"correction_old"}


def test_imports_a_consented_sidecar(tmp_path):
    (tmp_path / "active_learning").mkdir()
    _write_pair(
        tmp_path / "active_learning",
        "correction_new",
        {"training_consent": True, "timestamp": "2026-10-25T10:00:00+00:00", "annotations": ANNOTATIONS},
    )

    assert _run(tmp_path) == {"correction_new"}


def test_skips_a_sidecar_that_does_not_say_yes(tmp_path):
    source = tmp_path / "active_learning"
    source.mkdir()
    _write_pair(source, "correction_missing", {"annotations": ANNOTATIONS})
    _write_pair(source, "correction_false", {"training_consent": False, "annotations": ANNOTATIONS})
    _write_pair(source, "correction_string", {"training_consent": "true", "annotations": ANNOTATIONS})

    assert _run(tmp_path) == set()
