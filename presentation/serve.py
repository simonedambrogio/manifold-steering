#!/usr/bin/env python3
"""Serve the presentation with caching disabled.

Browsers aggressively cache the static files (script.js, style.css, and the
multi-MB figure iframes), which makes edits appear not to take effect. Serving
with no-store headers means a normal refresh always fetches the latest files.

    python3 presentation/serve.py [port]   # default 8077
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    root = Path(__file__).resolve().parent
    handler = partial(NoCacheHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("", port), handler)
    print(f"serving {root} at http://localhost:{port}/ (no-cache)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
