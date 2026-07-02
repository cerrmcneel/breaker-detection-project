import os
import json
import uuid
import hashlib
import subprocess
import cv2
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from src.model.pipeline import PanelSafePipeline

print("--- PanelSafe GPU Inference Pipeline Server ---")
pipeline = PanelSafePipeline()


def _build_health():
    """Report exactly what is being served (model md5, classes, git commit) so
    deployment drift is visible from a single `GET /`. The stale-model outage that
    motivated this hid for days precisely because the server exposed no identity."""
    model_path = pipeline.config.get("yolo_model_path", "models/best.pt")
    health = {
        "service": "src.model.inference_server",
        "status": "active",
        "model_path": model_path,
        "use_hmm": pipeline.config.get("use_hmm"),
        "classifier_mode": pipeline.config.get("classifier_mode"),
        "device": pipeline.device,
    }
    try:
        with open(model_path, "rb") as f:
            health["model_md5"] = hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        health["model_md5"] = f"unavailable ({e})"
    try:
        health["classes"] = pipeline.yolo_model.names
    except Exception:
        pass
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        health["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    return health


HEALTH = _build_health()
print(f"--- SERVING FINGERPRINT: {json.dumps(HEALTH)} ---")


class InferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(HEALTH).encode())

    def do_POST(self):
        if self.path == "/predict":
            temp_path = None
            try:
                content_length = int(self.headers['Content-Length'])
                file_data = self.rfile.read(content_length)
                
                # Parse boundary if multipart
                if b"boundary=" in self.headers.get('Content-Type', '').encode():
                    parts = file_data.split(b"\r\n\r\n")
                    if len(parts) > 1:
                        file_data = parts[1].split(b"\r\n--")[0]

                # Save bytes to a temp file since YOLO / OpenCV pipeline expects a path
                temp_filename = f"inference_temp_{uuid.uuid4()}.jpg"
                # Save locally relative to execution directory
                temp_path = os.path.join("data", "images", "raw_uploads", temp_filename)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(file_data)
                
                # Run the full local pipeline (YOLO + Crop Classifier + EasyOCR + HMM)
                refined_predictions = pipeline.run_inference(temp_path)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Return response
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "success", 
                    "predictions": refined_predictions
                }).encode())
                
            except Exception as e:
                print(f"Error during GPU pipeline inference: {e}")
                if temp_path and os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())

def run_server():
    port = int(os.getenv("INFERENCE_PORT", 8089))
    server = ThreadingHTTPServer(("0.0.0.0", port), InferenceHandler)
    print(f"Serving GPU Pipeline on port {port}...")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
