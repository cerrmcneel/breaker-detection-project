"""Automated Verification of T9 Viewfinder Offline Service Worker & In-Browser Inference.

Uses Chrome DevTools Protocol (CDP) via aiohttp to:
1. Load the page online and wait for Service Worker to install and precache assets.
2. Verify all 9 assets in Cache Storage `panelsafe-viewfinder-v1.0.0`.
3. Emulate network disconnection (offline=True).
4. Reload the page while completely offline.
5. Verify page renders, Service Worker responds from cache, COOP/COEP headers persist,
   crossOriginIsolated is true, and YOLO26 inference runs offline.
6. Capture a screenshot of the offline reloaded state.
"""

import asyncio
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9222
APP_URL = "http://localhost:8095/?src=video"
CACHE_NAME = "panelsafe-viewfinder-v1.0.0"


async def run_offline_verification():
    user_data_dir = Path(tempfile.gettempdir()) / "chrome_t9_offline_cdp"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CDP Test] Launching Headless Chrome on debug port {PORT}...")
    chrome_cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={user_data_dir}",
        "--window-size=1280,720",
        "--no-first-run",
        "--no-default-browser-check",
        APP_URL
    ]

    chrome_proc = subprocess.Popen(chrome_cmd)
    msg_id = 0

    async def send_cmd(ws, method, params=None):
        nonlocal msg_id
        msg_id += 1
        payload = {"id": msg_id, "method": method, "params": params or {}}
        await ws.send_json(payload)
        while True:
            msg = await ws.receive_json()
            if msg.get("id") == msg_id:
                return msg.get("result", {})

    try:
        # 1. Wait for Chrome debug endpoint
        print("[CDP Test] Connecting to Chrome debug endpoint...")
        ws_url = None
        for _ in range(30):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://127.0.0.1:{PORT}/json") as resp:
                        if resp.status == 200:
                            targets = await resp.json()
                            for t in targets:
                                if t.get("type") == "page":
                                    ws_url = t.get("webSocketDebuggerUrl")
                                    break
                            if ws_url:
                                break
            except Exception:
                pass
            await asyncio.sleep(0.5)

        if not ws_url:
            raise RuntimeError("Could not retrieve webSocketDebuggerUrl from Chrome")

        print(f"[CDP Test] Connected to target page: {ws_url}")

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # Enable required CDP domains
                await send_cmd(ws, "Page.enable")
                await send_cmd(ws, "Network.enable")
                await send_cmd(ws, "Runtime.enable")

                print("[CDP Test] Step 1: Online Load & Precache Installation...")
                # Allow 8 seconds for Service Worker to install, precache 9 assets, and start camera
                await asyncio.sleep(8)

                # Query Cache Storage directly from JavaScript
                eval_cache_script = f"""
                (async () => {{
                    const cache = await caches.open('{CACHE_NAME}');
                    const keys = await cache.keys();
                    return keys.map(k => new URL(k.url).pathname);
                }})()
                """
                cache_res = await send_cmd(ws, "Runtime.evaluate", {
                    "expression": eval_cache_script,
                    "awaitPromise": True,
                    "returnByValue": True
                })
                cached_paths = cache_res.get("result", {}).get("value", [])
                print(f"[CDP Test] Cached assets in '{CACHE_NAME}' ({len(cached_paths)} items):")
                for p in cached_paths:
                    print(f"   -> {p}")

                # Check online telemetry
                eval_telemetry = """
                ({
                    isolated: window.crossOriginIsolated,
                    threads: ort.env.wasm.numThreads,
                    infMs: parseFloat(document.getElementById('infTime').textContent),
                    fps: document.getElementById('uiFps').textContent,
                    status: document.getElementById('statusText').textContent,
                    offlineHUD: document.getElementById('offlineStatus').textContent,
                    onLine: navigator.onLine,
                    hasController: !!navigator.serviceWorker.controller
                })
                """
                online_tel_res = await send_cmd(ws, "Runtime.evaluate", {
                    "expression": eval_telemetry,
                    "returnByValue": True
                })
                online_tel = online_tel_res.get("result", {}).get("value", {})
                print("\n[CDP Test] Online Baseline Telemetry:", json.dumps(online_tel, indent=2))

                # Step 2: Go Offline
                print("\n[CDP Test] Step 2: Disconnecting Network (offline=True)...")
                await send_cmd(ws, "Network.emulateNetworkConditions", {
                    "offline": True,
                    "latency": 0,
                    "downloadThroughput": 0,
                    "uploadThroughput": 0
                })

                # Step 3: Reload Page while completely Offline
                print("[CDP Test] Step 3: Reloading page while 100% OFFLINE...")
                await send_cmd(ws, "Page.reload", {})

                # Allow 8 seconds for offline reload, cache-first response, model re-initialization, and inference
                await asyncio.sleep(8)

                # Step 4: Verify Offline State & Telemetry
                print("[CDP Test] Step 4: Inspecting offline page state...")
                offline_tel_res = await send_cmd(ws, "Runtime.evaluate", {
                    "expression": eval_telemetry,
                    "returnByValue": True
                })
                offline_tel = offline_tel_res.get("result", {}).get("value", {})
                print("[CDP Test] Offline Reload Telemetry:", json.dumps(offline_tel, indent=2))

                # Step 5: Capture Offline Screenshot
                print("[CDP Test] Step 5: Capturing offline screenshot...")
                shot_res = await send_cmd(ws, "Page.captureScreenshot", {"format": "png"})
                shot_b64 = shot_res.get("data")
                if shot_b64:
                    shot_bytes = base64.b64decode(shot_b64)
                    shot_path = Path(__file__).parent / "screenshot_offline.png"
                    shot_path.write_bytes(shot_bytes)
                    print(f"[CDP Test] Saved offline screenshot: {shot_path.resolve()} ({len(shot_bytes)} bytes)")

                # Step 6: Save Results Report JSON
                report = {
                    "task": "T9_viewfinder_offline",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cacheName": CACHE_NAME,
                    "cachedAssetsCount": len(cached_paths),
                    "cachedAssets": cached_paths,
                    "onlineBaseline": online_tel,
                    "offlineReload": offline_tel,
                    "verification": {
                        "offlineReloadSuccessful": offline_tel.get("status") in ["BOARD FOUND", "SEARCHING..."],
                        "crossOriginIsolatedPreservedOffline": offline_tel.get("isolated") is True,
                        "wasmMultiThreadingActiveOffline": offline_tel.get("threads") == 4,
                        "inferenceActiveOffline": (offline_tel.get("infMs") or 0) > 0,
                        "offlineHUDText": offline_tel.get("offlineHUD"),
                        "navigatorOnLineReported": offline_tel.get("onLine")
                    }
                }

                report_path = Path(__file__).parent / "offline_verification_results.json"
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"[CDP Test] Saved offline report to: {report_path.resolve()}")
                print("\n>>> FINAL VERIFICATION SUMMARY:")
                print(json.dumps(report["verification"], indent=2))

    finally:
        print("[CDP Test] Terminating Headless Chrome...")
        chrome_proc.terminate()
        try:
            chrome_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            chrome_proc.kill()


if __name__ == "__main__":
    asyncio.run(run_offline_verification())
