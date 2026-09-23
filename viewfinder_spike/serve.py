import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class ViewfinderHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler adding Cross-Origin Isolation headers for multi-threaded WASM."""

    def end_headers(self):
        # Enable SharedArrayBuffer for multi-threaded ONNX Runtime Web WASM
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/benchmark":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            with open("benchmark_results.json", "wb") as f:
                f.write(body)
            print("\n>>> IN-BROWSER BENCHMARK TELEMETRY RECEIVED:")
            print(body.decode("utf-8"))
            print("<<<\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = 8092, directory: str = "."):
    os.chdir(directory)
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, ViewfinderHTTPRequestHandler)
    print("\n=======================================================")
    print("PanelSafe Viewfinder Spike HTTP Server Running")
    print(f"URL: http://127.0.0.1:{port}/")
    print("Headers: COOP (same-origin), COEP (require-corp) enabled")
    print(f"Serving directory: {Path(directory).resolve()}")
    print("Press Ctrl+C to stop.")
    print("=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Serve the Viewfinder ONNX Spike static page.")
    parser.add_argument("--port", type=int, default=8092, help="Port to serve on (default: 8092).")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory to serve static files from.",
    )
    args = parser.parse_args()
    run_server(port=args.port, directory=args.dir)


if __name__ == "__main__":
    main()
