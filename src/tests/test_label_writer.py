import os
import tempfile

import pytest

# ============================================================
#  TDD — test_label_writer.py
#  These tests are written BEFORE label_writer.py exists.
#  Run them now and they should FAIL (RED phase).
#  Then we implement label_writer.py to make them GREEN.
# ============================================================
from src.data_gen.label_writer import normalize_box, write_label_file


# -----------------------------------------------------------------
# Test 1 — The "sanity" test: a box perfectly centered on 640×640
#   Box that fills the whole image: x=0, y=0, w=640, h=640
#   Expected: x_center=0.5, y_center=0.5, width=1.0, height=1.0
# -----------------------------------------------------------------
def test_center_box_on_640x640():
    """
    A box placed dead-center on a 640×640 image should return
    x_center=0.5 and y_center=0.5.
    """
    result = normalize_box(x=160, y=160, w=320, h=320, img_width=640, img_height=640)

    assert result["x_center"] == pytest.approx(0.5)
    assert result["y_center"] == pytest.approx(0.5)
    assert result["width"]    == pytest.approx(0.5)
    assert result["height"]   == pytest.approx(0.5)


# -----------------------------------------------------------------
# Test 2 — Non-square image: 1280×720 (HD frame)
#   Verify that width and height axes are normalized independently.
# -----------------------------------------------------------------
def test_normalize_non_square_image():
    """
    On a 1280×720 image, width/height normalization must use
    the correct axis (not the same divisor for both).
    """
    result = normalize_box(x=0, y=0, w=1280, h=720, img_width=1280, img_height=720)

    assert result["x_center"] == pytest.approx(0.5)
    assert result["y_center"] == pytest.approx(0.5)
    assert result["width"]    == pytest.approx(1.0)
    assert result["height"]   == pytest.approx(1.0)


# -----------------------------------------------------------------
# Test 3 — Values from last night's Socratic session:
#   x=192, y=144, w=320, h=320 on a 640×640 image.
#   x_center = (192 + 320/2) / 640 = 352/640 = 0.55
#   y_center = (144 + 320/2) / 640 = 304/640 = 0.475
# -----------------------------------------------------------------
def test_known_values_from_session():
    """
    Values we derived by hand last session — treat this as a
    regression guard for the formula you already understand.
    """
    result = normalize_box(x=192, y=144, w=320, h=320, img_width=640, img_height=640)

    assert result["x_center"] == pytest.approx(0.55)
    assert result["y_center"] == pytest.approx(0.475)
    assert result["width"]    == pytest.approx(0.5)
    assert result["height"]   == pytest.approx(0.5)


# -----------------------------------------------------------------
# Test 4 — Output format: normalize_box must return a dict with
#   exactly these four keys (YOLO label format).
# -----------------------------------------------------------------
def test_output_is_dict_with_correct_keys():
    """
    normalize_box must return a dict — not a tuple, not a list.
    YOLO label files need named fields, not positional guessing.
    """
    result = normalize_box(x=0, y=0, w=64, h=64, img_width=640, img_height=640)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"x_center", "y_center", "width", "height"}


# =================================================================
#  Task 3.2 — write_label_file()
#
#  YOLO expects one .txt file per image.
#  Each line: <class_id> <x_center> <y_center> <width> <height>
#  All values space-separated, normalized to [0, 1].
# =================================================================

# -----------------------------------------------------------------
# Test 5 — Single object: write one breaker annotation and verify
#   the file contains exactly one line in YOLO format.
# -----------------------------------------------------------------
def test_write_single_object_label():
    """
    A label file for one MCB (class 0) in the center of a 640×640
    image should produce one line: '0 0.5 0.5 0.5 0.5'
    """
    annotations = [
        {"class_id": 0, "x": 160, "y": 160, "w": 320, "h": 320}
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "label_0001.txt")
        write_label_file(out_path, annotations, img_width=640, img_height=640)

        with open(out_path, "r") as f:
            lines = f.read().strip().splitlines()

    assert len(lines) == 1
    assert lines[0] == "0 0.5 0.5 0.5 0.5"


# -----------------------------------------------------------------
# Test 6 — Multi-object: two breakers on the same image.
#   File must have exactly 2 lines, one per object.
# -----------------------------------------------------------------
def test_write_multi_object_label():
    """
    Two annotations → two lines in the .txt file.
    Order must be preserved (first annotation = first line).
    """
    annotations = [
        {"class_id": 0, "x": 0,   "y": 0,   "w": 320, "h": 320},  # top-left
        {"class_id": 1, "x": 320, "y": 320, "w": 320, "h": 320},  # bottom-right
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "label_0002.txt")
        write_label_file(out_path, annotations, img_width=640, img_height=640)

        with open(out_path, "r") as f:
            lines = f.read().strip().splitlines()

    assert len(lines) == 2
    assert lines[0] == "0 0.25 0.25 0.5 0.5"
    assert lines[1] == "1 0.75 0.75 0.5 0.5"


# -----------------------------------------------------------------
# Test 7 — File is actually created on disk.
#   An empty annotation list must still produce a file
#   (empty files are valid in YOLO datasets).
# -----------------------------------------------------------------
def test_label_file_is_created_on_disk():
    """
    write_label_file must always create the file, even when
    the annotations list is empty.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "label_empty.txt")
        write_label_file(out_path, [], img_width=640, img_height=640)

        assert os.path.exists(out_path), "File was not created!"
