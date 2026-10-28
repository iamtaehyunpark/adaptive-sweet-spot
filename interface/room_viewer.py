"""Local web server for the 3D room viewer and speaker placement (Milestone 3.2b).

Serves interface/web/room_viewer.{html,js} plus the room data written by
tools/room_setup.py, and saves speaker positions back to config/room.json.
Runs only on 127.0.0.1.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

WEB_DIR = Path(__file__).resolve().parent / "web"


def _clean_speakers(raw: list[dict], template: list[dict]) -> list[dict]:
    """Accept only known speakers and numeric positions from the page."""
    by_name = {s["name"]: s for s in raw if isinstance(s, dict) and "name" in s}
    out = []
    for t in template:
        s = by_name.get(t["name"], t)
        pos = [round(float(v), 3) for v in s.get("position", t["position"])][:3]
        out.append({"name": t["name"], "channel": t["channel"], "lfe": t.get("lfe", False),
                    "position": pos, "placed": bool(s.get("placed", t.get("placed", False)))})
    return out


def serve(room_file: Path, cloud_xyz: Path, cloud_rgb: Path, defaults: list[dict],
          port: int = 8765, open_browser: bool = True) -> dict:
    """Run the viewer until the page sends "finish" (or Ctrl-C). Returns the saved room."""
    room = json.loads(room_file.read_text())
    if not room.get("speakers"):
        room["speakers"] = defaults
    done = threading.Event()
    lock = threading.Lock()

    def save(speakers):
        with lock:
            room["speakers"] = _clean_speakers(speakers, defaults)
            room_file.write_text(json.dumps(room, indent=2))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            routes = {
                "/": (WEB_DIR / "room_viewer.html", "text/html; charset=utf-8"),
                "/room_viewer.js": (WEB_DIR / "room_viewer.js", "text/javascript; charset=utf-8"),
                "/api/cloud.xyz": (cloud_xyz, "application/octet-stream"),
                "/api/cloud.rgb": (cloud_rgb, "application/octet-stream"),
            }
            if path in routes:
                file, ctype = routes[path]
                self._send(file.read_bytes(), ctype)
            elif path == "/api/room":
                with lock:
                    self._send(json.dumps(room).encode(), "application/json")
            elif path == "/api/defaults":
                self._send(json.dumps(defaults).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path not in ("/api/save", "/api/finish"):
                self._send(b"not found", "text/plain", 404)
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                save(data["speakers"])
            except (ValueError, KeyError, TypeError) as e:
                self._send(json.dumps({"ok": False, "error": str(e)}).encode(), "application/json", 400)
                return
            self._send(b'{"ok": true}', "application/json")
            if path == "/api/finish":
                done.set()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"room viewer: {url}  (finish in the page, or Ctrl-C here)")
    if open_browser:
        webbrowser.open(url)
    try:
        done.wait()
    except KeyboardInterrupt:
        print("stopped; last saved positions kept")
    server.shutdown()
    return room
