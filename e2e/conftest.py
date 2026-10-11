"""Fixtures for the front-end browser tests.

The pages are static HTML, so a plain file server is enough: no gateway, no
inference, nothing written to data/. Calls the page makes to API routes
(`/count/`, `/upload/`) simply 404 here, which is fine for layout tests.
"""
import functools
import http.server
import pathlib
import threading

import pytest

FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "app" / "frontend"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - signature set by the base class
        pass


@pytest.fixture(scope="session")
def base_url():
    """Serve app/frontend on a free port; pytest-playwright's `page.goto("/")` uses this."""
    handler = functools.partial(_QuietHandler, directory=str(FRONTEND_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
