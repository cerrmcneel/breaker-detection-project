import glob
import json
import os
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import MAX_FILE_SIZE, app, validate_image_upload


def make_jpeg_bytes(width=20, height=20):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client():
    # Startup event scans real dataset dirs on disk; harmless (read-only) if they
    # exist, gracefully skipped if they don't (see get_unique_dataset_count).
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_seen_hashes():
    # seen_hashes is process-global state main.py mutates on every /upload/ call;
    # tests must not leak dedup state into each other.
    original = set(main_module.seen_hashes)
    yield
    main_module.seen_hashes = original


# --- validate_image_upload (unit-level, no HTTP) ---

def test_validate_image_upload_accepts_real_image():
    validate_image_upload(make_jpeg_bytes())  # should not raise


def test_validate_image_upload_rejects_garbage_bytes():
    with pytest.raises(HTTPException) as exc_info:
        validate_image_upload(b"this is not an image")
    assert exc_info.value.status_code == 400


# --- /predict/ ---

def test_predict_success(client):
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"predictions": [{"class": "MCB", "conf": 0.9}]}

    with patch("app.main.requests.post", return_value=fake_response) as mock_post:
        resp = client.post(
            "/predict/",
            files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["summary"]["total_components"] == 1
    mock_post.assert_called_once()


def test_predict_rejects_invalid_image(client):
    with patch("app.main.requests.post") as mock_post:
        resp = client.post(
            "/predict/",
            files={"file": ("panel.jpg", b"not an image", "image/jpeg")},
        )

    assert resp.status_code == 400
    mock_post.assert_not_called()


def test_predict_returns_503_when_inference_unreachable(client):
    with patch("app.main.requests.post", side_effect=ConnectionError("refused")):
        resp = client.post(
            "/predict/",
            files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 503


# --- /upload/ ---

def test_upload_success(client):
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"predictions": []}

    with patch("app.main.requests.post", return_value=fake_response), \
         patch("app.main.get_unique_dataset_count", return_value=(0, set())):
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"country": "ES", "rcd_test_result": "Responsive"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "duplicate" not in body  # only the early-return dedup path sets this

    # Clean up the file this endpoint actually wrote to disk.
    saved = body.get("filename")
    if saved:
        for path in glob.glob(os.path.join("data", "images", "raw_uploads", saved)):
            os.remove(path)


def test_upload_rejects_oversized_file(client):
    oversized = b"0" * (MAX_FILE_SIZE + 1)
    with patch("app.main.requests.post") as mock_post:
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", oversized, "image/jpeg")},
        )

    assert resp.status_code == 413
    mock_post.assert_not_called()


def test_upload_rejects_invalid_image(client):
    with patch("app.main.get_unique_dataset_count", return_value=(0, set())):
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", b"not an image", "image/jpeg")},
        )

    assert resp.status_code == 400


def test_upload_detects_duplicate(client):
    payload = make_jpeg_bytes()
    file_hash = __import__("hashlib").sha256(payload).hexdigest()

    with patch("app.main.get_unique_dataset_count", return_value=(0, {file_hash})):
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", payload, "image/jpeg")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["duplicate"] is True
    assert body["filename"] == "DUPLICATE"


# --- /active-learning/save ---

def test_active_learning_save_success(client):
    resp = client.post(
        "/active-learning/save",
        files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
        data={"annotations": json.dumps({"boxes": []})},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # This endpoint has no filename in its response; sweep any correction_* pair
    # written during this test run so the repo stays clean.
    for path in glob.glob(os.path.join("data", "active_learning", "correction_*")):
        os.remove(path)


def test_active_learning_save_rejects_oversized_file(client):
    oversized = b"0" * (MAX_FILE_SIZE + 1)
    resp = client.post(
        "/active-learning/save",
        files={"file": ("panel.jpg", oversized, "image/jpeg")},
        data={"annotations": "{}"},
    )
    assert resp.status_code == 413


def test_active_learning_save_rejects_invalid_image(client):
    resp = client.post(
        "/active-learning/save",
        files={"file": ("panel.jpg", b"not an image", "image/jpeg")},
        data={"annotations": "{}"},
    )
    assert resp.status_code == 400


def test_active_learning_save_rejects_malformed_annotations(client):
    resp = client.post(
        "/active-learning/save",
        files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
        data={"annotations": "{not valid json"},
    )
    assert resp.status_code == 400


def test_active_learning_save_blocks_cross_origin(client):
    resp = client.post(
        "/active-learning/save",
        files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
        data={"annotations": "{}"},
        headers={"origin": "https://evil.example.com", "host": "testserver"},
    )
    assert resp.status_code == 403


# --- degraded-inference contract (found by external review 2026-08-09) ----------------
# An empty prediction list is ambiguous: it means either "nothing detected" or
# "inference failed". Grading the second case fabricates a safety verdict, so the
# server must not produce a score at all when inference did not run.

def test_upload_does_not_fabricate_a_score_when_inference_fails(client):
    with patch("app.main.requests.post", side_effect=ConnectionError("cluster down")), \
         patch("app.main.get_unique_dataset_count", return_value=(0, set())):
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"country": "ES", "rcd_test_result": "Responsive"},
        )

    assert resp.status_code == 200          # the upload itself succeeded
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["inference_ok"] is False

    # Omitted, not nulled: a client that forgets to check inference_ok must get
    # `undefined` rather than a plausible-looking number it can render.
    assert "score" not in body
    assert "feedback" not in body

    # The image is still kept -- a failed analysis is no reason to bin training data.
    assert body["filename"] != "DUPLICATE"
    for path in glob.glob(os.path.join("data", "images", "raw_uploads", body["filename"])):
        os.remove(path)


def test_upload_still_scores_normally_when_inference_succeeds(client):
    fake = MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.return_value = {"predictions": [{"class": "MCB", "conf": 0.9, "box": [0, 0, 1, 1]}]}

    with patch("app.main.requests.post", return_value=fake), \
         patch("app.main.get_unique_dataset_count", return_value=(0, set())):
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"country": "ES", "rcd_test_result": "Responsive"},
        )

    body = resp.json()
    assert body["status"] == "success"
    assert body["inference_ok"] is True
    assert isinstance(body["score"], int)
    assert body["feedback"]

    for path in glob.glob(os.path.join("data", "images", "raw_uploads", body["filename"])):
        os.remove(path)
