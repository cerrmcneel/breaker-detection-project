#!/usr/bin/env bash
# Versioned launcher for the PanelSafe GPU inference server.
#
# WHY THIS EXISTS: prod was found running a hand-pasted `python3 -c "..."` server that
# had drifted from the repo and was serving a stale model nobody could see (it exposed
# no identity). Always launch the maintained, config-driven server from here so that
# "what's deployed" == "what's in the repo", and auto-restart it if it dies.
#
# Usage:
#   INFERENCE_PORT=8088 ./scripts/start_inference.sh
#   # if the torch-enabled interpreter isn't `python` on PATH, point at it:
#   PYTHON=/path/to/torch/python INFERENCE_PORT=8088 ./scripts/start_inference.sh
#
# Verify what's actually serving (model md5 + classes + git commit):
#   curl -s http://localhost:${INFERENCE_PORT:-8088}/ | python -m json.tool
set -u
cd "$(dirname "$0")/.."                     # repo root
export INFERENCE_PORT="${INFERENCE_PORT:-8088}"
PY="${PYTHON:-python}"
echo "[start_inference] repo=$(pwd) port=$INFERENCE_PORT python=$PY"
while true; do
  "$PY" -m src.model.inference_server
  code=$?
  echo "[start_inference] server exited (code $code). Restarting in 3s..." >&2
  sleep 3
done
