import asyncio
import glob
import json
import os
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.concurrency import run_in_threadpool

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


# --- event loop must stay responsive during slow upstream calls -----------------------
# The gateway runs as a SINGLE uvicorn worker with StaticFiles mounted on the same
# app, so a blocking upstream call froze the entire site (up to 90s on a cold GPU
# backend). These tests assert the loop keeps running, not merely that the code
# "looks async".

def _slow_response(delay=0.4):
    def _send(*args, **kwargs):
        time.sleep(delay)                      # blocking, like a real slow upstream
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"predictions": []}
        return resp
    return _send


async def _count_ticks_during(coro_factory, window=0.4):
    """Run coro_factory() while a 10ms ticker runs; return how many ticks landed."""
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await coro_factory()
    finally:
        task.cancel()
    return ticks


def test_upstream_post_does_not_block_the_event_loop():
    from app.main import _post_image_to

    async def scenario():
        with patch("app.main.requests.post", side_effect=_slow_response(0.4)):
            await _post_image_to("http://upstream/predict", b"bytes", timeout=5)

    ticks = asyncio.run(_count_ticks_during(scenario))
    # ~40 ticks if the loop stayed free; ~0 if the thread blocked it.
    assert ticks > 10, f"event loop stalled during upstream call (only {ticks} ticks)"


def test_control_a_direct_blocking_call_would_stall_the_loop():
    """Proves the test above has teeth: the same wait done inline DOES stall."""
    async def scenario():
        time.sleep(0.4)          # what the code used to do, effectively

    ticks = asyncio.run(_count_ticks_during(scenario))
    assert ticks <= 2, f"expected the blocking control to stall the loop, got {ticks} ticks"


def test_dataset_scan_does_not_block_the_event_loop():
    from app.main import get_unique_dataset_count  # noqa: F401  (patched by name below)

    async def scenario():
        with patch("app.main.get_unique_dataset_count", side_effect=lambda: (time.sleep(0.4), (0, set()))[1]):
            await run_in_threadpool(main_module.get_unique_dataset_count)

    ticks = asyncio.run(_count_ticks_during(scenario))
    assert ticks > 10, f"event loop stalled during dataset scan (only {ticks} ticks)"


# --- /upload/ Azure failover (review finding #2, fixed 2026-08-09) --------------------
# The failover previously existed only on /predict/, so a GPU cluster outage dropped
# the homeowner-facing endpoint straight to the degraded no-score response even
# while Azure was healthy.

def _ok_response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_upload_fails_over_to_azure_when_primary_is_down(client):
    calls = []

    def route(url, *a, **k):
        calls.append(url)
        if url == main_module.INFERENCE_URL:
            raise ConnectionError("gpu cluster down")
        return _ok_response({"predictions": [{"class": "MCB", "conf": 0.9, "box": [0, 0, 1, 1]}]})

    with patch("app.main.requests.post", side_effect=route), \
         patch("app.main.get_unique_dataset_count", return_value=(0, set())):
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"country": "ES", "rcd_test_result": "Responsive"},
        )

    body = resp.json()
    assert calls == [main_module.INFERENCE_URL, main_module.AZURE_FALLBACK_URL]
    # Azure served it, so this is a real analysis -- score present, not degraded.
    assert body["status"] == "success"
    assert body["inference_ok"] is True
    assert body["inference_engine"] == main_module.FAILOVER_ENGINE
    assert isinstance(body["score"], int)

    for path in glob.glob(os.path.join("data", "images", "raw_uploads", body["filename"])):
        os.remove(path)


def test_upload_degrades_only_when_both_backends_fail(client):
    with patch("app.main.requests.post", side_effect=ConnectionError("everything down")), \
         patch("app.main.get_unique_dataset_count", return_value=(0, set())):
        resp = client.post(
            "/upload/",
            files={"file": ("panel.jpg", make_jpeg_bytes(), "image/jpeg")},
            data={"country": "ES"},
        )

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["inference_ok"] is False
    assert body["inference_engine"] is None
    assert "score" not in body          # still never fabricates one

    for path in glob.glob(os.path.join("data", "images", "raw_uploads", body["filename"])):
        os.remove(path)


def test_failover_is_skipped_when_the_time_budget_is_exhausted():
    """Cloudflare cuts a proxied request at ~100s; burning the budget on a doomed
    Azure attempt would hand the user a 524 instead of our own degraded response."""
    from app.main import InferenceUnavailable, _run_inference

    calls = []

    def route(url, *a, **k):
        calls.append(url)
        raise ConnectionError("primary down")

    async def scenario():
        # Pretend the primary already consumed the whole budget.
        with patch("app.main.requests.post", side_effect=route), \
             patch("app.main.INFERENCE_BUDGET", 0):
            with pytest.raises(InferenceUnavailable) as exc:
                await _run_inference(b"data", ".jpg")
            return exc.value

    err = asyncio.run(scenario())
    assert calls == [main_module.INFERENCE_URL]          # Azure never attempted
    assert "skipped" in str(err.failover_err)


def test_failover_still_attempted_when_budget_allows():
    from app.main import _run_inference

    calls = []

    def route(url, *a, **k):
        calls.append(url)
        if url == main_module.INFERENCE_URL:
            raise ConnectionError("primary down")
        return _ok_response({"predictions": []})

    async def scenario():
        with patch("app.main.requests.post", side_effect=route):
            return await _run_inference(b"data", ".jpg")

    predictions, engine = asyncio.run(scenario())
    assert calls == [main_module.INFERENCE_URL, main_module.AZURE_FALLBACK_URL]
    assert engine == main_module.FAILOVER_ENGINE
    assert predictions == []
