# Persistent Windows launcher for the PanelSafe GPU inference server.
#
# WHY: replaces the hand-pasted, drift-prone inline server. Runs the maintained,
# config-driven server (loads models/best.pt via pipeline_config) and auto-restarts
# it if it exits. Register as a Scheduled Task so it survives logoff/reboot:
#
#   # (admin PowerShell) register to auto-start at logon, hidden, highest privileges:
#   schtasks /Create /TN "PanelSafeInference" /SC ONLOGON /RL HIGHEST /F /TR `
#     "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\ironhack\labs\breaker-detection-project\scripts\start_inference.ps1"
#   schtasks /Run /TN "PanelSafeInference"      # start it now (stop any manual server first)
#
# Verify what's serving:  curl.exe -s http://localhost:8088/
$ErrorActionPreference = "Continue"
Set-Location "C:\ironhack\labs\breaker-detection-project"
if (-not $env:INFERENCE_PORT) { $env:INFERENCE_PORT = "8088" }
while ($true) {
    Write-Host "[start_inference] launching src.model.inference_server on :$($env:INFERENCE_PORT)"
    python -m src.model.inference_server
    Write-Host "[start_inference] server exited (code $LASTEXITCODE). Restarting in 3s..."
    Start-Sleep -Seconds 3
}
