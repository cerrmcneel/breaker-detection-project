import os
import uuid
import re
import json
import logging
import hashlib
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seen_hashes = set()

app = FastAPI(title="Breaker Detection Data Collection Beta")

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

# Pre-load existing hashes
if os.path.exists(LOG_FILE):
    try:
        with open(LOG_FILE, "r") as f:
            _entries = json.load(f)
            for _e in _entries:
                if "hash" in _e:
                    seen_hashes.add(_e["hash"])
    except json.JSONDecodeError:
        pass

# Helper function to log metadata
def log_metadata(original_filename: str, saved_filename: str, country: str, file_hash: str = ""):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "original_filename": original_filename,
        "saved_filename": saved_filename,
        "country": country,
        "hash": file_hash
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
    country: str = Form(default="Unknown")
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
            
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 4. Save the file with streaming and size limit to prevent memory exhaustion
        file_size = 0
        file_hash_obj = hashlib.sha256()
        with open(file_path, "wb") as f:
            # Read 1MB at a time
            while chunk := await file.read(1024 * 1024):
                file_size += len(chunk)
                file_hash_obj.update(chunk)
                if file_size > MAX_FILE_SIZE:
                    f.close()
                    os.remove(file_path) # Clean up partial file on failure
                    raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
                f.write(chunk)
            
        file_hash = file_hash_obj.hexdigest()
        
        # Deduplication check
        if file_hash in seen_hashes:
            os.remove(file_path)
            logger.info(f"Discarded duplicate {file.filename} (hash: {file_hash})")
            return JSONResponse(content={"message": "Duplicate discarded", "filename": unique_filename, "duplicate": True}, status_code=200)

        seen_hashes.add(file_hash)

        # Log metadata
        log_metadata(file.filename, unique_filename, country, file_hash)
        
        logger.info(f"Saved {file.filename} as {unique_filename}")
        
        return JSONResponse(content={"message": "Upload successful", "filename": unique_filename, "duplicate": False}, status_code=200)

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

@app.post("/verify-admin/")
async def verify_admin(password: str = Form(...)):
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        # Failsafe if env var isn't configured
        raise HTTPException(status_code=500, detail="Admin password not configured on server.")
        
    if password == admin_password:
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
