"""
Local dev launcher for the full PanelSafe gateway (app/main.py), pointed at
the verified-working Modal failover inference endpoint instead of the
homelab's Tailscale-only GPU worker (which isn't reachable from arbitrary
dev environments). Lets you preview the current frontend code end-to-end,
including real AI predictions, without needing homelab/Tailscale access.

Run: python run_local_server.py
"""
import os

os.environ.setdefault(
    "INFERENCE_URL",
    "https://ericmcneel--panelsafe-failover-inference-asgi.modal.run/predict",
)

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
