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
# The website now talks directly to the Windows GPU worker via hostNetwork
INFERENCE_URL = os.getenv("INFERENCE_URL", "http://192.168.1.147:8080/predict")

seen_hashes = set()
upload_lock = asyncio.Lock()

app = FastAPI(title="PanelSafe: Breaker Detection & Analysis")

# Initialize Pipeline Components
heuristic_engine = SpatialHeuristicEngine()
ocr_reader = OCRReader()

@app.on_event("startup")
async def startup_event():
    logger.info(f"PanelSafe Gateway ready. Connecting to K3s cluster at: {INFERENCE_URL}")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/images/raw_uploads")
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        return
        
    for filename in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    seen_hashes.add(file_hash)
            except: pass

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
    temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # 1. Offload Heavy Lifting to K3s GPU Cluster
        logger.info(f"Forwarding image to K3s Cluster for GPU inference...")
        try:
            with open(temp_path, "rb") as f:
                response = requests.post(INFERENCE_URL, files={"file": f}, timeout=45)
                response.raise_for_status()
                cluster_data = response.json()
                raw_predictions = cluster_data.get("predictions", [])
        except Exception as cluster_err:
            logger.error(f"K3s Cluster Connection Failed: {cluster_err}")
            raise HTTPException(status_code=503, detail=f"GPU Inference Cluster is unreachable at {INFERENCE_URL}")

        # 2. Apply Spatial Heuristics locally
        refined_predictions = heuristic_engine.apply_logic(raw_predictions, temp_path)

        # 3. Apply OCR locally
        for pred in refined_predictions:
            if pred["class"] in ["MCB", "MAINBREAKER", "RCD", "RCD_SI"]:
                ocr_result = ocr_reader.read_bounding_box(temp_path, pred["box"])
                pred["ocr_text"] = ocr_result

        os.remove(temp_path)
        return {
            "status": "success",
            "panel_layout": refined_predictions,
            "summary": {
                "total_components": len(refined_predictions),
                "inference_engine": "K3s-GPU-Cluster"
            }
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), country: str = Form(default="Unknown"), rcd_test_result: str = Form(default="Not Tested")):
    try:
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Invalid file type.")
        
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")
            
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if file_hash in seen_hashes:
            return {"status": "success", "duplicate": True, "filename": "DUPLICATE", "tracking_id": "ALREADY-UPLOADED"}

        unique_filename = f"{country.upper()}_{uuid.uuid4()}{os.path.splitext(file.filename)[1].lower()}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_bytes)
            
        tracking_id = "BKR-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        seen_hashes.add(file_hash)
        
        # Log metadata
        entry = {"timestamp": datetime.now().isoformat(), "original_filename": file.filename, "saved_filename": unique_filename, "country": country, "hash": file_hash, "rcd_test_result": rcd_test_result, "tracking_id": tracking_id}
        entries = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f: entries = json.load(f)
        entries.append(entry)
        with open(LOG_FILE, "w") as f: json.dump(entries, f, indent=4)
            
        return {"status": "success", "filename": unique_filename, "tracking_id": tracking_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/count/")
async def get_count():
    return {"count": len(seen_hashes)}

@app.get("/score/{tracking_id}")
async def get_score(tracking_id: str):
    if not os.path.exists(LOG_FILE): raise HTTPException(status_code=404)
    with open(LOG_FILE, "r") as f: entries = json.load(f)
    for entry in reversed(entries):
        if entry.get("tracking_id") == tracking_id:
            if entry.get("manual_score") is not None:
                return {"status": "scored", "score": entry["manual_score"], "feedback": entry.get("manual_feedback", "")}
            return {"status": "pending"}
    raise HTTPException(status_code=404)

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

app.mount("/", StaticFiles(directory=os.path.join("app", "frontend"), html=True), name="frontend")
