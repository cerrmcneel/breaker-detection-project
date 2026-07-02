import os
import uuid
import re
import json
import logging
import hashlib
import asyncio
import requests
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import argparse
import random
import string
import cv2
import numpy as np
from src.model.heuristics import SpatialHeuristicEngine
from src.model.ocr_reader import OCRReader

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- INFERENCE CONFIGURATION ---
# The website now talks directly to the Windows GPU worker via the 'gpu-worker' Tailscale alias
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://gpu-worker:8088/predict")

seen_hashes = set()
file_hash_cache = {}
upload_lock = asyncio.Lock()

app = FastAPI(title="PanelSafe: Breaker Detection & Analysis")

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

# FastAPI Gateway acts as a lightweight router to the GPU cluster

@app.on_event("startup")
async def startup_event():
    logger.info(f"PanelSafe Gateway ready. Connecting to K3s cluster at: {INFERENCE_URL}")
    
    # Initialize cache and populate seen_hashes on startup
    count, current_hashes = get_unique_dataset_count()
    global seen_hashes
    seen_hashes = current_hashes
    logger.info(f"Loaded {count} unique dataset image hashes.")

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
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}

@app.post("/predict/")
async def predict_panel(file: UploadFile = File(...)):
    # Build the temp name from a UUID + sanitized extension only; never trust the
    # client-supplied filename (it can contain path separators -> traversal).
    safe_ext = os.path.splitext(file.filename or "")[1].lower()
    if safe_ext not in ALLOWED_EXTENSIONS:
        safe_ext = ".jpg"
    temp_filename = f"temp_{uuid.uuid4()}{safe_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Forward the image to K3s GPU Worker cluster for pipeline prediction
        logger.info("Forwarding image to K3s GPU Worker pipeline...")
        try:
            with open(temp_path, "rb") as f:
                response = requests.post(INFERENCE_URL, files={"file": f}, timeout=45)
                response.raise_for_status()
                cluster_data = response.json()
                refined_predictions = cluster_data.get("predictions", [])
        except Exception as cluster_err:
            logger.error(f"K3s Cluster Connection Failed: {cluster_err}")
            raise HTTPException(status_code=503, detail=f"GPU Inference Cluster is unreachable at {INFERENCE_URL}")

        os.remove(temp_path)
        return {
            "status": "success",
            "panel_layout": refined_predictions,
            "summary": {
                "total_components": len(refined_predictions),
                "inference_engine": "K3s-GPU-Cluster-Pipeline"
            }
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
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
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Invalid file type.")
        
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")
            
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        # Check against currently scanned hashes for up-to-date de-duplication
        _, current_hashes = get_unique_dataset_count()
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

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        tracking_id = "BKR-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

        # Run model inference. Distinguish "inference failed" from "genuinely nothing
        # detected" — a silent empty-on-error is exactly how a model outage masquerades
        # as "no components found" with no signal to anyone (it hid a stale-model bug
        # for days). Surface the failure instead of swallowing it.
        predictions = []
        inference_ok = False
        inference_detail = "ok"
        try:
            files = {"file": (file.filename, file_bytes, file.content_type)}
            # The full pipeline (YOLO + OCR + HMM) is far slower than raw YOLO; the old
            # 5s ceiling caused silent timeouts. Give real headroom.
            response = requests.post(INFERENCE_URL, files=files, timeout=30)
            response.raise_for_status()
            cluster_data = response.json()
            predictions = cluster_data.get("predictions", [])
            inference_ok = True
        except Exception as inf_err:
            inference_detail = str(inf_err)
            logger.error(f"INFERENCE CALL FAILED ({INFERENCE_URL}): {inf_err}")

        # Grade the panel
        auto_score, auto_feedback = grade_panel_layout(predictions, rcd_test_result, country)

        # Log metadata
        entry = {
            "timestamp": datetime.now().isoformat(),
            "original_filename": file.filename,
            "saved_filename": unique_filename,
            "country": country,
            "hash": file_hash,
            "rcd_test_result": rcd_test_result,
            "tracking_id": tracking_id,
            "manual_score": auto_score,
            "manual_feedback": auto_feedback
        }
        # Serialize the dedup-set mutation and the log read-modify-write so concurrent
        # uploads cannot clobber upload_log.json or drop a seen-hash entry.
        async with upload_lock:
            seen_hashes.add(file_hash)
            entries = []
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "r") as f: entries = json.load(f)
            entries.append(entry)
            with open(LOG_FILE, "w") as f: json.dump(entries, f, indent=4)

        return {"status": "success", "filename": unique_filename, "tracking_id": tracking_id, "score": auto_score, "feedback": auto_feedback, "inference_ok": inference_ok, "inference_detail": inference_detail}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/count/")
async def get_count():
    count, current_hashes = get_unique_dataset_count()
    global seen_hashes
    seen_hashes = current_hashes
    return {"count": count}

@app.post("/active-learning/save")
async def save_active_learning(file: UploadFile = File(...), annotations: str = Form(...)):
    try:
        active_learning_dir = "data/active_learning"
        os.makedirs(active_learning_dir, exist_ok=True)
        base_name = f"correction_{int(datetime.now().timestamp())}_{file.filename}"
        with open(os.path.join(active_learning_dir, base_name), "wb") as f:
            f.write(await file.read())
        with open(os.path.join(active_learning_dir, f"{os.path.splitext(base_name)[0]}.json"), "w") as f:
            f.write(annotations)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/unifilar", StaticFiles(directory=os.path.join("panel-safe-unifilar", "dist"), html=True), name="unifilar")
app.mount("/", StaticFiles(directory=os.path.join("app", "frontend"), html=True), name="frontend")
