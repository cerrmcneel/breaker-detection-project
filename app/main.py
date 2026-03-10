import os
import uuid
import re
import json
import logging
import hashlib
import asyncio
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import argparse
import random
import string

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seen_hashes = set()
# Lock to prevent race conditions when multiple identical images upload at the exact same millisecond
upload_lock = asyncio.Lock()

app = FastAPI(title="Breaker Detection Data Collection Beta")

def generate_tracking_id():
    """Generate a random 5-character alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    return "BKR-" + "".join(random.choices(chars, k=5))

# Disable caching globally for the beta phase to ensure frontend updates immediately propagate
@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Use UPLOAD_DIR from environment (Docker maps this to the NAS), fallback to local for dev
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/images/raw_uploads")
# We keep the log file in the same base directory as the images
LOG_FILE = os.path.join(os.path.dirname(UPLOAD_DIR), "upload_log.json")

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Security Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up: Scanning existing files for deduplication hashes...")
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
            except Exception as e:
                logger.error(f"Error reading {file_path} for hash: {e}")
                
    logger.info(f"Startup complete: Discovered {len(seen_hashes)} unique hashes on disk.")

# Helper function to log metadata
def log_metadata(original_filename: str, saved_filename: str, country: str, file_hash: str = "", rcd_test_result: str = "Not Tested", tracking_id: str = ""):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "country": country,
        "hash": file_hash,
        "rcd_test_result": rcd_test_result,
        "tracking_id": tracking_id,
        "manual_score": None,
        "manual_feedback": ""
    }
    
    # Simple append to a JSON list in a file (not efficient for huge scale, but fine for beta)
    entries = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                entries = json.load(f)
        except json.JSONDecodeError:
            entries = [] # Start fresh if corrupt
            
    entries.append(entry)
    
    with open(LOG_FILE, "w") as f:
        json.dump(entries, f, indent=4)

@app.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    country: str = Form(default="Unknown"),
    rcd_test_result: str = Form(default="Not Tested")
):
    try:
        # 1. Input Validation for Country
        country = str(country)[:50] # Enforce max length of 50 chars
        if country != "Unknown" and not re.match(r"^[a-zA-Z\s\-]+$", country):
            raise HTTPException(status_code=400, detail="Invalid country format.")

        # 2. File Validation: MIME type
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, and GIF are allowed.")

        # 3. File Validation: Extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if not file_extension or file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file extension. Only .jpg, .jpeg, .png, and .gif are allowed.")
            
        # Generate unique filename with country prefix
        safe_country = "".join(c for c in country if c.isalnum() or c.isspace() or c == "-").strip().upper()
        if not safe_country:
            safe_country = "UNKNOWN"
            
        unique_filename = f"{safe_country}_{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 4. Save the file with streaming and size limit to prevent memory exhaustion
        file_size = 0
        file_hash_obj = hashlib.sha256()
        
        # Read the file to get the hash but DO NOT save it yet
        # Store in memory temporarily since these are <10MB images
        file_bytes = await file.read()
        file_size = len(file_bytes)
        
        if file_size > MAX_FILE_SIZE:
             raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
             
        file_hash_obj.update(file_bytes)
        file_hash = file_hash_obj.hexdigest()
        
        # Concurrency safety lock
        async with upload_lock:
            # Deduplication check FIRST before writing anything to disk
            if file_hash in seen_hashes:
                logger.info(f"Discarded duplicate {file.filename} (hash: {file_hash})")
                return JSONResponse(content={"message": "Duplicate discarded", "filename": file.filename, "duplicate": True}, status_code=200)
    
            seen_hashes.add(file_hash)
            
            # Generate tracking ID
            tracking_id = generate_tracking_id()

            # Now safe to write the file
            with open(file_path, "wb") as f:
                f.write(file_bytes)
    
            # Log metadata
            log_metadata(file.filename, unique_filename, country, file_hash, rcd_test_result, tracking_id)
            
            logger.info(f"Saved {file.filename} as {unique_filename} with tracking ID {tracking_id}")
            
            return JSONResponse(
                content={
                    "message": "Upload successful", 
                    "filename": unique_filename, 
                    "duplicate": False,
                    "tracking_id": tracking_id
                }, 
                status_code=200
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/count/")
async def get_upload_count():
    try:
        # Filter out anything that isn't a file (like subdirectories if any exist)
        count = len([f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))])
        return {"count": count}
    except Exception as e:
        logger.error(f"Error getting file count: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve file count.")

@app.get("/score/{tracking_id}")
async def get_score(tracking_id: str):
    """Retrieve the manual safety score for a given tracking ID."""
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="Log file not found.")
        
    try:
        with open(LOG_FILE, "r") as f:
            entries = json.load(f)
            
        # Search backwards since newer entries are appended
        for entry in reversed(entries):
            if entry.get("tracking_id") == tracking_id:
                if entry.get("manual_score") is not None:
                    return {
                        "status": "scored",
                        "score": entry.get("manual_score"),
                        "feedback": entry.get("manual_feedback", "")
                    }
                else:
                    return {"status": "pending"}
                    
        raise HTTPException(status_code=404, detail="Tracking ID not found.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error reading log data.")
    except Exception as e:
        logger.error(f"Error checking score for {tracking_id}: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while checking the score.")

from pydantic import BaseModel

class AdminVerifyRequest(BaseModel):
    password: str

@app.post("/verify-admin/")
async def verify_admin(req: AdminVerifyRequest):
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        # Failsafe if env var isn't configured
        raise HTTPException(status_code=500, detail="Admin password not configured on server.")
        
    if not req.password:
        raise HTTPException(status_code=400, detail="Password is required.")
        
    if req.password == admin_password:
        return {"verified": True}
    else:
        raise HTTPException(status_code=401, detail="Incorrect password.")

# Mount frontend directory for static files
# Create app/frontend if it doesn't exist to avoid startup errors
FRONTEND_PATH = os.path.join("app", "frontend")
os.makedirs(os.path.join("app", "frontend"), exist_ok=True)
app.mount("/", StaticFiles(directory=os.path.join("app", "frontend"), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow access from local network
    uvicorn.run(app, host="0.0.0.0", port=8000)
