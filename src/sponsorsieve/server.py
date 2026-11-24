"""Tiny HTTP API (standard library only), for podcast ad removers and other tools.

POST /v1/detect   {"segments": [{start, end, text, id?}], "threshold": 0.5, "self_promo": false}
             ->   {"ads": [{start, end, score, kind, segment_ids?}], "segments": [[p_content, p_ad, p_self], ...]}
                  segment_ids is present when the input segments carry an "id" (any JSON value).
GET  /health  ->  {"ok": true, "version": "..."}
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__

MAX_BODY = 20 * 1024 * 1024


def serve(detector, host: str = "127.0.0.1", port: int = 8765) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "version": __version__})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/detect":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BODY:
                return self._send(413, {"error": "body missing or too large"})
            try:
                req = json.loads(self.rfile.read(n))
                segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": str(s["text"]),
                         **({"id": s["id"]} if "id" in s else {})} for s in req["segments"]]
                threshold, self_promo = float(req.get("threshold", 0.5)), bool(req.get("self_promo", False))
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            probs = detector.segment_probs(segs)
            ads = detector.detect(segs, threshold=threshold, include_self_promo=self_promo, probs=probs)
            self._send(200, {"ads": ads, "segments": [[round(p, 4) for p in row] for row in probs]})

        def log_message(self, *a):
            pass

    print(f"sponsorsieve {__version__} listening on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
