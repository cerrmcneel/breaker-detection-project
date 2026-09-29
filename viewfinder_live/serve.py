import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class ViewfinderLiveHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler adding Cross-Origin Isolation headers for multi-threaded WASM

    and registering correct MIME types for WebAssembly and ONNX assets.
    """

    extensions_map = SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({
        ".wasm": "application/wasm",
        ".mjs": "application/javascript",
        ".js": "application/javascript",
        ".onnx": "application/octet-stream",
        ".json": "application/json",
        ".mp4": "video/mp4",
    })

    def end_headers(self):
        # Enable SharedArrayBuffer for multi-threaded ONNX Runtime Web WASM
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Service-Worker-Allowed", "/")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    # The POST routes below are Gemini's self-test hooks: they write files into the
    # served folder. They stay OFF unless --dev is given, because once Tailscale
    # Serve proxies this server, every device on the tailnet can reach it.
    dev_endpoints = False

    # /api/mobile_test used to build filenames from the request's "orientation"
    # field, so a crafted value could write outside this folder. Only these
    # values are accepted now.
    ALLOWED_ORIENTATIONS = {"portrait", "landscape"}

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if not self.dev_endpoints:
            self.send_response(404)
            self.end_headers()
            return
        if self.path == "/api/benchmark":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            with open("benchmark_results.json", "wb") as f:
                f.write(body)
            print("\n>>> LIVE VIEWFINDER BENCHMARK TELEMETRY RECEIVED:")
            print(body.decode("utf-8"))
            print("<<<\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif self.path == "/api/mobile_test":
            import base64
            import json

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))

            orientation = data.get("orientation")
            if orientation not in self.ALLOWED_ORIENTATIONS:
                self.send_response(400)
                self.end_headers()
                return
            # If screenshot image data provided, decode and save to file
            if "screenshot" in data and data["screenshot"].startswith("data:image/"):
                header, b64data = data["screenshot"].split(",", 1)
                img_bytes = base64.b64decode(b64data)
                screenshot_filename = f"screenshot_{orientation}.png"
                with open(screenshot_filename, "wb") as img_f:
                    img_f.write(img_bytes)
                print(f"[Mobile Test] Saved screenshot: {screenshot_filename} ({len(img_bytes)} bytes)")
                del data["screenshot"]  # Don't clutter JSON with large base64 string

            report_file = f"mobile_test_{orientation}.json"
            with open(report_file, "w", encoding="utf-8") as rf:
                json.dump(data, rf, indent=2)

            print(f"\n>>> MOBILE LAYOUT TELEMETRY RECEIVED ({orientation.upper()}):")
            print(json.dumps(data, indent=2))
            print("<<<\n")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif self.path == "/api/offline_test":
            import json

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode("utf-8"))

            report_file = "offline_test_report.json"
            with open(report_file, "w", encoding="utf-8") as rf:
                json.dump(data, rf, indent=2)

            print("\n>>> OFFLINE VIEWFINDER TELEMETRY RECEIVED:")
            print(json.dumps(data, indent=2))
            print("<<<\n")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def run_server(host: str = "127.0.0.1", port: int = 8095, directory: str = ".", dev: bool = False):
    os.chdir(directory)
    ViewfinderLiveHTTPRequestHandler.dev_endpoints = dev
    server_address = (host, port)
    httpd = HTTPServer(server_address, ViewfinderLiveHTTPRequestHandler)
    print("\n=======================================================")
    print("PanelSafe Live Viewfinder HTTP Server Running")
    print(f"URL: http://{host}:{port}/")
    print("Headers: COOP (same-origin), COEP (require-corp) enabled")
    print(f"Self-test POST endpoints: {'ENABLED (--dev)' if dev else 'disabled'}")
    print(f"Serving directory: {Path(directory).resolve()}")
    print("Press Ctrl+C to stop.")
    print("=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Serve the Live Viewfinder page.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8095, help="Port to serve on (default: 8095).")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory to serve static files from.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable the self-test POST endpoints (/api/*). Never use with Tailscale Serve on.",
    )
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, directory=args.dir, dev=args.dev)


if __name__ == "__main__":
    main()
