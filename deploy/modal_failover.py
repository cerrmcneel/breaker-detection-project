"""
Cloud failover deployment for PanelSafe, on Modal (https://modal.com).

WHY MODAL RATHER THAN A JOB-QUEUE SERVERLESS PROVIDER (e.g. RunPod Serverless):
`app/main.py` already calls out to INFERENCE_URL as a plain HTTP POST with a
multipart file (see app/main.py:33, :162, :304) and expects a plain JSON
response -- the exact same contract `src/model/inference_server.py` serves
today over Tailscale. Modal's `@modal.asgi_app()` exposes a real HTTPS
endpoint that speaks that same plain-HTTP contract with zero code changes to
either app/main.py or the inference logic. A job-queue-style serverless API
(submit job -> poll/receive result) would require rewriting the calling code
to speak a different protocol during failover -- avoided here on purpose.

TWO INDEPENDENT FUNCTIONS, BOTH SCALE-TO-ZERO (pay only while a request is
being served, ~0 cost idle):
  - `gateway_asgi`   -- CPU only. Runs the *exact* existing FastAPI app
                        (app.main:app) unchanged, using the same
                        requirements.txt as the production Docker image.
  - `inference_asgi` -- GPU (T4). Runs PanelSafePipeline directly in-process
                        (no Tailscale hop needed in the cloud -- gateway and
                        inference can share a container network here, but are
                        kept as separate scale-to-zero functions so the
                        expensive GPU only spins up for actual predictions).

DEPLOY ORDER MATTERS:
  1. `modal deploy deploy/modal_failover.py` deploys BOTH functions. Modal
     prints each function's URL at the end, in the pattern:
       https://<your-modal-workspace>--panelsafe-failover-inference-asgi.modal.run
       https://<your-modal-workspace>--panelsafe-failover-gateway-asgi.modal.run
  2. Copy the INFERENCE URL from step 1 into the `INFERENCE_URL_OVERRIDE`
     Modal Secret (see setup instructions below) so the gateway forwards to
     it instead of the homelab. Because both are defined in this same file,
     re-running `modal deploy` after setting the secret wires them together
     automatically -- no manual URL surgery needed after the first deploy.

ONE-TIME SETUP (run these yourself -- needs your own Modal account/billing):
  pip install modal
  modal setup                              # opens browser, authenticates your account
  modal secret create panelsafe-failover INFERENCE_URL_OVERRIDE=<inference URL from step 1>
  modal deploy deploy/modal_failover.py
"""
import os
import modal

app = modal.App("panelsafe-failover")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Inference image: GPU-enabled torch, matches src/model/pipeline.py's needs ---
inference_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "tesseract-ocr")
    .pip_install(
        "fastapi", "python-multipart", "opencv-python", "numpy",
        "ultralytics", "easyocr", "pytesseract", "torch",  # GPU-enabled default index, unlike requirements.txt's CPU pin
    )
    .add_local_dir(PROJECT_ROOT, remote_path="/code", ignore=[".git", ".venv", "runs", "data/dataset"])
)

# --- Gateway image: identical dependency set to the production Dockerfile ---
gateway_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "tesseract-ocr")
    .pip_install_from_requirements(os.path.join(PROJECT_ROOT, "requirements.txt"))
    .add_local_dir(PROJECT_ROOT, remote_path="/code", ignore=[".git", ".venv", "runs", "data/dataset"])
)


# Flip to 1 during periods where an error can't be afforded (active job search,
# a sales demo, a portfolio review) -- keeps one GPU container warm at all times
# so failover has zero cold-start delay, at a small constant cost. Flip back to
# 0 afterward for genuine $0-idle pay-per-use. Redeploy after changing this.
INFERENCE_MIN_CONTAINERS = 0


@app.function(image=inference_image, gpu="T4", scaledown_window=120, timeout=90, min_containers=INFERENCE_MIN_CONTAINERS)
@modal.asgi_app()
def inference_asgi():
    """Serves the exact same /predict + / contract as inference_server.py,
    over a real HTTPS endpoint instead of Tailscale."""
    import sys
    import uuid
    sys.path.insert(0, "/code")
    os.chdir("/code")

    from fastapi import FastAPI, UploadFile, File
    from src.model.pipeline import PanelSafePipeline

    web_app = FastAPI()
    pipeline = PanelSafePipeline()

    @web_app.get("/")
    def health():
        return {
            "service": "modal-failover-inference",
            "status": "active",
            "classifier_mode": pipeline.config.get("classifier_mode"),
            "use_hmm": pipeline.config.get("use_hmm"),
        }

    @web_app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        contents = await file.read()
        temp_path = f"/tmp/{uuid.uuid4()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)
        try:
            predictions = pipeline.run_inference(temp_path)
            return {"status": "success", "predictions": predictions}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return web_app


@app.function(
    image=gateway_image,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("panelsafe-failover")],
)
@modal.asgi_app()
def gateway_asgi():
    """Runs the unmodified production FastAPI app, pointed at the cloud
    inference function instead of the homelab GPU worker."""
    import sys
    sys.path.insert(0, "/code")
    os.chdir("/code")

    # app/main.py reads INFERENCE_URL straight from the environment (app/main.py:33)
    override = os.environ.get("INFERENCE_URL_OVERRIDE")
    if override:
        os.environ["INFERENCE_URL"] = override.rstrip("/") + "/predict"

    from app.main import app as fastapi_app
    return fastapi_app
