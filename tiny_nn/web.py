"""A local drawing interface; inference uses exactly the saved Python model."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np

from . import LABELS
from .dataset import render_shape


def make_handler(model, report_path=Path("artifacts/reports/test/report.html")):
    html = Path(__file__).with_name("lab.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value, status=200, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request = urlparse(self.path)
            if request.path == "/":
                return self.respond(html, content_type="text/html")
            if request.path == "/favicon.ico":
                return self.respond(b"", status=204, content_type="image/x-icon")
            if request.path == "/api/info":
                return self.respond({**model.stats(), "labels": LABELS, "report_available": report_path.is_file()})
            if request.path == "/report" and report_path.is_file():
                return self.respond(report_path.read_bytes(), content_type="text/html")
            if request.path == "/api/example":
                query = parse_qs(request.query)
                label = query.get("shape", ["cross"])[0]
                try:
                    seed = int(query.get("seed", ["42"])[0])
                    pixels, _ = render_shape(label, np.random.default_rng(seed))
                    return self.respond({"pixels": (pixels.astype(float) / 255).tolist(), "expected": label})
                except (ValueError, TypeError) as error:
                    return self.respond({"error": str(error)}, status=400)
            return self.respond({"error": "Not found"}, status=404)

        def do_POST(self):
            if self.path != "/api/predict":
                return self.respond({"error": "Not found"}, status=404)
            if self.headers.get_content_type() != "application/json":
                return self.respond({"error": "Send pixels as application/json"}, status=415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    return self.respond({"error": "Expected a small JSON image"}, status=413)
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or "pixels" not in payload:
                    raise ValueError("Provide a pixels array")
                pixels = np.asarray(payload["pixels"], dtype=np.float32)
                if pixels.shape != (16, 16):
                    raise ValueError("Provide exactly 16 rows of 16 pixels")
                return self.respond(model.predict(pixels))
            except (ValueError, TypeError, OverflowError) as error:
                return self.respond({"error": str(error)}, status=400)

    return Handler


def serve(model, port=8765, open_browser=False):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(model))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Tiny NN drawing lab: {url}", flush=True)
    print("Draw an isolated cross, hollow square, hollow circle, or horizontal/vertical line. Ctrl-C stops the server.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
