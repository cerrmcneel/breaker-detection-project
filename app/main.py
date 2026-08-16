import asyncio
import hashlib
import json
import logging
import os
import random
import re
import string
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INFERENCE CONFIGURATION ---
# The website now talks directly to the Windows GPU worker via the 'gpu-worker' Tailscale alias
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://gpu-worker:8088/predict")
AZURE_FALLBACK_URL = os.getenv("AZURE_FALLBACK_URL", "https://api-cloud.panelsafe.cv/predict/")

PRIMARY_ENGINE = "K3s-GPU-Cluster-Pipeline"
FAILOVER_ENGINE = "Azure-Cloud-Failover"

# The full pipeline (YOLO + OCR + HMM) is far slower than raw YOLO, and the
# primary timeout also has to cover a cold-started scale-to-zero GPU backend
# loading the model.
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "90"))
FAILOVER_TIMEOUT = int(os.getenv("FAILOVER_TIMEOUT", "30"))

# Total wall-clock budget for primary + failover combined. Cloudflare terminates
# a proxied request at ~100s with a 524, so an un-budgeted 90 + 30 could burn the
# whole allowance and hand the user a Cloudflare error page instead of our own
# degraded response. The failover gets whatever is left of this budget, and is
# skipped entirely if too little remains to be worth attempting.
INFERENCE_BUDGET = int(os.getenv("INFERENCE_BUDGET", "95"))
MIN_FAILOVER_TIMEOUT = 5

seen_hashes = set()
file_hash_cache = {}
upload_lock = asyncio.Lock()

# App instance created after get_unique_dataset_count and lifespan are defined below

def get_unique_dataset_count():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(project_root, "data", "dataset", "train", "images")
    val_dir = os.path.join(project_root, "data", "dataset", "val", "images")
    raw_dir = os.path.join(project_root, "data", "images", "raw_uploads")
    
    train_fns = set()
    val_fns = set()
    raw_fns = set()
    
    # Securely scan train images
    if os.path.exists(train_dir):
        try:
            train_fns = set(
                f for f in os.listdir(train_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and not f.startswith('synth_panel_')
            )
        except Exception as e:
            logger.warning(f"Failed to scan train images: {e}")
            
    # Securely scan val images
    if os.path.exists(val_dir):
        try:
            val_fns = set(
                f for f in os.listdir(val_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and not f.startswith('synth_panel_')
            )
        except Exception as e:
            logger.warning(f"Failed to scan val images: {e}")
            
    # Securely scan raw uploads
    if os.path.exists(raw_dir):
        try:
            raw_fns = set(
                f for f in os.listdir(raw_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and not f.startswith('synth_panel_')
            )
        except Exception as e:
            logger.warning(f"Failed to scan raw uploads: {e}")
    else:
        try:
            os.makedirs(raw_dir, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create raw uploads: {e}")
            
    # Calculate perfect photo count: train + val + any new uploads not yet in train/val
    total_count = len(train_fns) + len(val_fns) + len(raw_fns - train_fns - val_fns)
    
    # We still build the hash set for seen_hashes de-duplication
    current_hashes = set()
    for d, fns in [(train_dir, train_fns), (val_dir, val_fns), (raw_dir, raw_fns)]:
        if not os.path.exists(d):
            continue
        for filename in fns:
            file_path = os.path.join(d, filename)
            try:
                stat = os.stat(file_path)
                mtime = stat.st_mtime
                size = stat.st_size
                
                cached = file_hash_cache.get(file_path)
                if cached and cached[0] == mtime and cached[1] == size:
                    file_hash = cached[2]
                else:
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    file_hash_cache[file_path] = (mtime, size, file_hash)
                
                current_hashes.add(file_hash)
            except Exception as e:
                logger.warning(f"Failed to hash {file_path}: {e}")
                
    return total_count, current_hashes

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"PanelSafe Gateway ready. Connecting to K3s cluster at: {INFERENCE_URL}")
    
    # Initialize cache and populate seen_hashes on startup
    count, current_hashes = get_unique_dataset_count()
    global seen_hashes
    seen_hashes = current_hashes
    logger.info(f"Loaded {count} unique dataset image hashes.")
    yield

app = FastAPI(title="PanelSafe: Breaker Detection & Analysis", lifespan=lifespan)

@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Global Constants
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/images/raw_uploads")
LOG_FILE = os.path.join(os.path.dirname(UPLOAD_DIR), "upload_log.json")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}

def validate_image_upload(file_bytes: bytes) -> None:
    """
    Raises HTTPException if file_bytes is not a valid, decodable image.
    """
    img = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")


# --- BLOCKING WORK OFF THE EVENT LOOP ---
#
# This gateway is deployed as a SINGLE uvicorn worker, so anything that blocks
# the event loop blocks every other visitor -- including the static file serving,
# since StaticFiles is mounted on this same app. Upstream inference can take up
# to 90s on a cold-started GPU backend, so a synchronous requests.post() here
# meant one slow analysis froze the whole site for a minute and a half.
#
# `requests` is used rather than an async HTTP client on purpose: the only
# httpx-family package this project installs is `httpx2` (pulled in by starlette
# for TestClient), and `import httpx` is NOT available in CI or in the deployed
# image even though it happens to exist in some local envs. run_in_threadpool
# keeps the event loop free without betting on that resolving a particular way.
#
# Threads are bounded (anyio's default limiter, ~40), so this raises effective
# concurrency from 1 to ~40 rather than to infinity. That is the right ceiling
# for this traffic; revisit with a real async client if it is ever approached.


def _write_bytes(path: str, data: bytes) -> None:
    """Blocking file write, intended to be called via run_in_threadpool.

    Creates the parent directory itself rather than relying on the side effect in
    get_unique_dataset_count(), which is where /upload/ previously got its target
    directory from -- an easy thing to break by reordering calls.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _append_log_entry(log_file: str, entry: dict) -> None:
    """Blocking read-modify-write of the upload log. Callers must hold upload_lock."""
    entries = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
    entries.append(entry)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4)


async def _post_image_to(url: str, data: bytes, timeout: int, ext: str = ".jpg"):
    """POST image bytes to an upstream inference endpoint, off the event loop.

    Sends a generated filename rather than the client-supplied one -- the caller's
    filename is untrusted and nothing upstream needs it (the worker assigns its
    own UUID name).
    """
    def _send():
        response = requests.post(
            url,
            files={"file": (f"upload{ext}", data, "image/jpeg")},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    return await run_in_threadpool(_send)


class InferenceUnavailable(Exception):
    """Raised when the primary cluster AND the Azure failover both failed."""

    def __init__(self, primary_err, failover_err):
        self.primary_err = primary_err
        self.failover_err = failover_err
        super().__init__(f"Local: {primary_err}, Azure: {failover_err}")


async def _run_inference(data: bytes, ext: str):
    """Run inference on the primary cluster, falling back to Azure.

    Returns (predictions, engine_name). Raises InferenceUnavailable if both fail.

    Shared by /predict/ and /upload/ deliberately: the failover previously existed
    only on /predict/, which meant the Azure capacity did not protect the
    homeowner-facing path that actually produces the safety score.
    """
    started = time.monotonic()
    try:
        data_json = await _post_image_to(INFERENCE_URL, data, timeout=INFERENCE_TIMEOUT, ext=ext)
        return data_json.get("predictions", []), PRIMARY_ENGINE
    except Exception as primary_err:
        logger.warning(f"Primary GPU Cluster ({INFERENCE_URL}) failed: {primary_err}. Failing over to Azure Cloud!")

        remaining = INFERENCE_BUDGET - (time.monotonic() - started)
        failover_timeout = int(min(FAILOVER_TIMEOUT, remaining))
        if failover_timeout < MIN_FAILOVER_TIMEOUT:
            # The primary consumed the budget. Attempting Azure now would very
            # likely be cut off by Cloudflare mid-flight anyway, and the caller
            # can degrade far more usefully than a 524 page can.
            logger.error(f"Skipping Azure failover: only {remaining:.0f}s of the {INFERENCE_BUDGET}s budget left.")
            raise InferenceUnavailable(
                primary_err,
                f"skipped, {remaining:.0f}s of {INFERENCE_BUDGET}s budget remaining",
            ) from primary_err

        try:
            data_json = await _post_image_to(AZURE_FALLBACK_URL, data, timeout=failover_timeout, ext=ext)
            predictions = data_json.get("predictions", data_json.get("panel_layout", []))
            logger.info("Successfully processed inference via Azure Cloud Failover!")
            return predictions, FAILOVER_ENGINE
        except Exception as failover_err:
            logger.error(f"Both Primary and Azure Cloud Failover failed. Local: {primary_err}, Azure: {failover_err}")
            raise InferenceUnavailable(primary_err, failover_err) from failover_err

# --- PYDANTIC RESPONSE SCHEMAS ---


class DetectionBox(BaseModel):
    box: Optional[List[float]] = Field(default_factory=list, description="[x1, y1, x2, y2] bounding box coordinates")
    class_name: str = Field(..., alias="class", description="Detected component class")
    conf: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    ocr_text: Optional[str] = Field(default="", description="Extracted label text")

    model_config = ConfigDict(populate_by_name=True)


class PanelSummary(BaseModel):
    total_components: int
    inference_engine: str


class PredictionResponse(BaseModel):
    status: str
    panel_layout: List[DetectionBox]
    summary: PanelSummary


@app.post("/predict/", response_model=PredictionResponse)
async def predict_panel(file: UploadFile = File(...)):
    # The client-supplied filename is untrusted (it can contain path separators),
    # so only its extension is used, and only to label the forwarded upload.
    safe_ext = os.path.splitext(file.filename or "")[1].lower()
    if safe_ext not in ALLOWED_EXTENSIONS:
        safe_ext = ".jpg"
    try:
        contents = await file.read()
        validate_image_upload(contents)

        # No temp file: the bytes are already in memory, so the previous
        # write-to-disk-then-reopen-twice round-trip bought nothing and added
        # three blocking filesystem operations to every request.
        logger.info("Forwarding image to K3s GPU Worker pipeline...")
        try:
            # engine is reported back rather than hardcoded, so audit logs stop
            # attributing failover traffic to the primary cluster.
            refined_predictions, engine = await _run_inference(contents, safe_ext)
        except InferenceUnavailable as err:
            raise HTTPException(
                status_code=503,
                detail=f"All inference endpoints unreachable. {err}",
            ) from err

        return {
            "status": "success",
            "panel_layout": refined_predictions,
            "summary": {
                "total_components": len(refined_predictions),
                "inference_engine": engine
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def grade_panel_layout(predictions, rcd_test_result, country="Unknown"):
    score = 100
    feedback_es = []
    feedback_en = []
    
    # RCD Test Result Grading
    if rcd_test_result == "Unresponsive":
        score -= 40
        feedback_es.append("❌ <strong>PELIGRO CRÍTICO:</strong> El botón de prueba del diferencial (RCD) no disparó el interruptor. Esto significa que su instalación NO tiene protección activa contra descargas eléctricas. Por favor, llame a un electricista autorizado de inmediato para reemplazarlo.")
        feedback_en.append("❌ <strong>CRITICAL DANGER:</strong> The differential (RCD) test button did not trip. This means your installation has NO active protection against electrical shocks. Please call a licensed electrician immediately to replace it.")
    elif rcd_test_result == "Slow":
        score -= 15
        feedback_es.append("⚠️ <strong>ADVERTENCIA:</strong> El diferencial (RCD) tardó en dispararse. Se recomienda que un electricista lo inspeccione, ya que la degradación mecánica puede impedir que actúe a tiempo.")
        feedback_en.append("⚠️ <strong>WARNING:</strong> The RCD (differential) responded slowly. It is recommended to have it inspected, as mechanical degradation can prevent timely protection.")
    elif rcd_test_result == "Not Tested":
        score -= 5
        feedback_es.append("⚠️ <strong>Diferencial no probado:</strong> Le recomendamos presionar el botón 'T' de su interruptor diferencial mensualmente para asegurarse de que funciona activamente.")
        feedback_en.append("⚠️ <strong>RCD Not Tested:</strong> We recommend pressing the 'T' test button on your differential (RCD) monthly to ensure it actively trips.")
    else: # Responsive
        feedback_es.append("✅ <strong>Diferencial funcionando:</strong> El botón de prueba disparó el RCD al instante. ¡Excelente hábito de seguridad mensual!")
        feedback_en.append("✅ <strong>RCD Working:</strong> The test button tripped the RCD instantly. Great job maintaining your monthly safety checks!")

    # Count components
    rcd_count = sum(1 for p in predictions if p.get("class") in ["RCD", "RCD_SI"])
    mcb_count = sum(1 for p in predictions if p.get("class") == "MCB")
    main_breaker_count = sum(1 for p in predictions if p.get("class") == "MAINBREAKER")
    oversurge_count = sum(1 for p in predictions if p.get("class") == "OVERSURGE")

    # Presence checks
    if rcd_count == 0:
        score -= 25
        feedback_es.append("⚠️ <strong>No se detectó Diferencial (RCD):</strong> No pudimos identificar un diferencial en la foto. Verifique si su cuadro tiene un interruptor con botón de test (marcado con 'T').")
        feedback_en.append("⚠️ <strong>No RCD detected:</strong> We couldn't identify a differential (RCD) in the photo. Please verify if your panel has a breaker with a test button (marked 'T').")
    
    if mcb_count == 0:
        score -= 10
        feedback_es.append("⚠️ <strong>No se detectaron térmicos (MCBs):</strong> No se identificaron interruptores individuales en el cuadro. Asegúrese de que la foto muestre claramente los interruptores.")
        feedback_en.append("⚠️ <strong>No MCBs detected:</strong> We couldn't identify individual circuit breakers (MCBs) in the panel. Ensure the photo clearly shows all breakers.")

    # Spanish market rules (if Spain or Spain-like panel layout)
    is_spain = country.upper() in ["ES", "SPAIN"] or (main_breaker_count > 0 or oversurge_count > 0)
    
    if is_spain:
        if main_breaker_count == 0:
            score -= 10
            feedback_es.append("ℹ️ <strong>Sin IGA (Interruptor General):</strong> No se detectó un interruptor de corte general. Los cuadros en España deben contar con un IGA para cortar la corriente total de la vivienda.")
            feedback_en.append("ℹ️ <strong>No Main Breaker (IGA) detected:</strong> We did not identify a dedicated General automatic breaker. Spanish standards require an IGA to cut power to the entire house.")
        
        if oversurge_count == 0:
            score -= 10
            feedback_es.append("ℹ️ <strong>Sin protector contra sobretensiones:</strong> No detectamos un protector de sobretensiones. La normativa española actual (REBT) exige su instalación para proteger los electrodomésticos contra picos de tensión.")
            feedback_en.append("ℹ️ <strong>No Surge Protector (Sobretensiones) detected:</strong> Modern Spanish regulations require a surge protector to shield appliances from voltage spikes.")

    # Bound score
    score = max(0, min(100, score))
    
    # Combine lists into HTML lines based on language
    report = f"""
<div class="lang-en-report">
  <strong>Automated Safety Score: {score}/100</strong><br>
  <ul style="margin: 8px 0; padding-left: 20px;">
    {"".join(f"<li style='margin-bottom: 5px;'>{item}</li>" for item in feedback_en)}
  </ul>
</div>
<hr style="margin: 10px 0; border: none; border-top: 1px dashed rgba(145,55,175,0.2);">
<div class="lang-es-report">
  <strong>Puntuación de Seguridad Automática: {score}/100</strong><br>
  <ul style="margin: 8px 0; padding-left: 20px;">
    {"".join(f"<li style='margin-bottom: 5px;'>{item}</li>" for item in feedback_es)}
  </ul>
</div>
<br>
<span style="font-size: 11px; opacity: 0.7; font-style: italic;">🤖 Auto-generated by PanelSafe AI Model</span>
"""
    return score, report

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), country: str = Form(default="Unknown"), rcd_test_result: str = Form(default="Not Tested")):
    try:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")
        validate_image_upload(file_bytes)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        # Check against currently scanned hashes for up-to-date de-duplication.
        # This walks three directories and SHA-256s every image, so it goes to a
        # worker thread rather than stalling the event loop on disk I/O.
        _, current_hashes = await run_in_threadpool(get_unique_dataset_count)
        global seen_hashes
        seen_hashes = current_hashes
        
        if file_hash in seen_hashes:
            return {"status": "success", "duplicate": True, "filename": "DUPLICATE", "tracking_id": "ALREADY-UPLOADED"}

        # Sanitize country before it becomes part of a filesystem path (traversal guard).
        safe_country = re.sub(r"[^A-Za-z0-9_-]", "", country).upper() or "UNKNOWN"
        safe_ext = os.path.splitext(file.filename or "")[1].lower()
        if safe_ext not in ALLOWED_EXTENSIONS:
            safe_ext = ".jpg"
        unique_filename = f"{safe_country}_{uuid.uuid4()}{safe_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        await run_in_threadpool(_write_bytes, file_path, file_bytes)

        tracking_id = "BKR-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

        # Run model inference. Distinguish "inference failed" from "genuinely nothing
        # detected" — a silent empty-on-error is exactly how a model outage masquerades
        # as "no components found" with no signal to anyone (it hid a stale-model bug
        # for days). Surface the failure instead of swallowing it.
        predictions = []
        inference_ok = False
        inference_detail = "ok"
        inference_engine = None
        try:
            # Same primary + Azure failover path as /predict/. This endpoint is the
            # one that produces the homeowner-facing safety score, so it is the one
            # that most needs the fallback -- without it, any GPU cluster outage
            # dropped straight to the degraded no-score response even while Azure
            # was healthy.
            predictions, inference_engine = await _run_inference(file_bytes, safe_ext)
            inference_ok = True
        except InferenceUnavailable as inf_err:
            inference_detail = str(inf_err)
            logger.error(f"ALL INFERENCE ENDPOINTS FAILED: {inf_err}")
        except Exception as inf_err:
            inference_detail = str(inf_err)
            logger.error(f"INFERENCE CALL FAILED ({INFERENCE_URL}): {inf_err}")

        # Only grade when inference actually ran. An empty `predictions` list is
        # ambiguous -- it means EITHER "the panel genuinely has no detectable
        # components" OR "the inference call failed" -- and grade_panel_layout
        # cannot tell the difference. Scoring the failure case fabricates a
        # verdict: a fully compliant panel comes back as 45/100 with "No RCD
        # detected" and "No MCBs detected", contradicting the user's own RCD test
        # answer in the same report. On a safety product that is worse than no
        # answer, because a homeowner may act on it.
        #
        # The score/feedback keys are OMITTED rather than nulled on failure, so a
        # client that forgets to check `inference_ok` cannot silently render a
        # fabricated number -- it gets `undefined` and has to handle it.
        if inference_ok:
            auto_score, auto_feedback = grade_panel_layout(predictions, rcd_test_result, country)
        else:
            auto_score, auto_feedback = None, None

        # Log metadata. The image itself is still kept either way -- a failed
        # analysis is no reason to discard training data the user just gave us.
        entry = {
            "timestamp": datetime.now().isoformat(),
            "original_filename": file.filename,
            "saved_filename": unique_filename,
            "country": country,
            "hash": file_hash,
            "rcd_test_result": rcd_test_result,
            "tracking_id": tracking_id,
            "inference_ok": inference_ok,
            "inference_engine": inference_engine,
            "manual_score": auto_score,
            "manual_feedback": auto_feedback
        }
        # Serialize the dedup-set mutation and the log read-modify-write so concurrent
        # uploads cannot clobber upload_log.json or drop a seen-hash entry.
        # The lock is held across the threadpool call so the read-modify-write of
        # upload_log.json stays atomic with respect to other requests in this
        # process. NOTE: asyncio.Lock is process-local -- see the roadmap; running
        # multiple uvicorn workers would reintroduce the clobbering it prevents.
        async with upload_lock:
            seen_hashes.add(file_hash)
            await run_in_threadpool(_append_log_entry, LOG_FILE, entry)

        payload = {
            "status": "success" if inference_ok else "degraded",
            "filename": unique_filename,
            "tracking_id": tracking_id,
            "inference_ok": inference_ok,
            "inference_detail": inference_detail,
            "inference_engine": inference_engine,
        }
        if inference_ok:
            payload["score"] = auto_score
            payload["feedback"] = auto_feedback
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/count/")
async def get_count():
    count, current_hashes = await run_in_threadpool(get_unique_dataset_count)
    global seen_hashes
    seen_hashes = current_hashes
    return {"count": count}

@app.post("/active-learning/save")
async def save_active_learning(request: Request, file: UploadFile = File(...), annotations: str = Form(...)):
    try:
        # Lightweight same-origin check: this endpoint is anonymous/no-login by design
        # (any visitor reviewing their own analysis can submit a correction), so this is
        # NOT a real auth boundary -- Origin/Referer are attacker-controllable. It only
        # raises the bar against casual cross-site/scripted abuse. Checked only when the
        # header is present, so requests that legitimately omit it (e.g. through a proxy
        # or tunnel that strips headers) aren't blocked.
        request_host = request.headers.get("host")
        origin_header = request.headers.get("origin") or request.headers.get("referer")
        if origin_header and request_host:
            origin_host = urllib.parse.urlparse(origin_header).netloc
            if origin_host and origin_host != request_host:
                raise HTTPException(status_code=403, detail="Cross-origin request blocked.")

        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")
        validate_image_upload(file_bytes)

        # Build the on-disk name from a UUID + sanitized extension only; never trust
        # the client-supplied filename (it can contain path separators -> traversal).
        safe_ext = os.path.splitext(file.filename or "")[1].lower()
        if safe_ext not in ALLOWED_EXTENSIONS:
            safe_ext = ".jpg"

        # Validate annotations is well-formed JSON before persisting it.
        try:
            parsed_annotations = json.loads(annotations)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="annotations must be valid JSON.")

        active_learning_dir = "data/active_learning"
        base_name = f"correction_{int(datetime.now().timestamp())}_{uuid.uuid4()}"

        def _persist_correction():
            os.makedirs(active_learning_dir, exist_ok=True)
            with open(os.path.join(active_learning_dir, f"{base_name}{safe_ext}"), "wb") as f:
                f.write(file_bytes)
            with open(os.path.join(active_learning_dir, f"{base_name}.json"), "w", encoding="utf-8") as f:
                json.dump(parsed_annotations, f)

        await run_in_threadpool(_persist_correction)
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/unifilar", StaticFiles(directory=os.path.join("panel-safe-unifilar", "dist"), html=True), name="unifilar")
app.mount("/", StaticFiles(directory=os.path.join("app", "frontend"), html=True), name="frontend")
