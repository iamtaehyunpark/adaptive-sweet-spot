"""Live app (Milestone 3.2): tracking + 3D room + setup + DSP stream in one browser page.

* LiveTracker  camera -> BlazeFace pipeline -> listener offset in the room frame;
               publishes every detection on the UDP bridge (interface/bridge.py),
               keeps an annotated JPEG for the page, and runs the room setup
               (countdown -> 1080p capture -> depth -> calibration/scan) on request
               while tracking continues.
* serve        local web server (127.0.0.1) for interface/web/app.{html,js}:
               room data, live events (SSE), camera preview (MJPEG), speaker saving.
"""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from config import CONFIG_DIR, ROOM_FILE, speaker_position

from . import setup as room_setup
from .bridge import BridgePublisher

WEB_DIR = Path(__file__).resolve().parent / "web"
CAMERA_FILE = CONFIG_DIR / "camera.json"  # chosen camera + mode + controls (per machine)
ROOT = room_setup.ROOT
SETUP_COUNTDOWN_S = 5.0
SETUP_WARMUP_S = 1.5   # after switching to the setup resolution, let exposure settle
SETUP_CAPTURE_S = 2.0
SETUP_CALIB_S = 2.0    # live detections in the last seconds of the countdown calibrate the face width
PREVIEW_WIDTH = 640
PREVIEW_FPS = 15.0
EVENTS_HZ = 20.0


def rate(times) -> float:
    return (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0


class LiveTracker(threading.Thread):
    def __init__(self, cfg: dict, camera_index: int | None = None, publish: bool = True):
        super().__init__(daemon=True, name="LiveTracker")
        from vision.pipeline import VisionPipeline

        self.cfg = cfg
        self.vcfg = cfg["vision"]
        settings = {"index": self.vcfg["camera_index"], "width": self.vcfg["width"],
                    "height": self.vcfg["height"], "fps": self.vcfg["fps"], "controls": {}}
        if CAMERA_FILE.exists():
            settings.update(json.loads(CAMERA_FILE.read_text()))
        if camera_index is not None:
            settings["index"] = camera_index
        self.cam_settings = settings
        self.cam = None
        self.actual_size = (settings["width"], settings["height"])
        self.capabilities: dict | None = None  # {"modes": [[w, h], ...], "controls": {...}} after a probe
        self.cameras: list[dict] = []           # camera picker entries with snapshots
        self.camera_busy = ""
        self.camera_message = ""
        self._commands: deque = deque()
        self.pipe = VisionPipeline(self.vcfg, ROOT)
        bridge = cfg["bridge"]
        self.bridge_addr = (bridge["udp_host"], bridge["udp_port"])
        self.publisher = BridgePublisher(*self.bridge_addr) if publish else None
        self.mapper = None
        self.lock = threading.Lock()
        self.preview = threading.Condition()
        self.jpeg: bytes | None = None
        self.jpeg_seq = 0
        self.room_version = 0
        self.setup_state = {"stage": "idle", "message": "", "log": []}
        self._setup_requested = False
        self._last = None  # last listener offset
        self._state = {"tracking": False}
        self.running = True

    # ---------------------------------------------------------------- API
    def snapshot(self) -> dict:
        with self.lock:
            return {**self._state, "setup": dict(self.setup_state), "room_version": self.room_version,
                    "camera": self.camera_state(),
                    "bridge": {"host": self.bridge_addr[0], "port": self.bridge_addr[1],
                               "sent": self.publisher.sent if self.publisher else None}}

    def request_setup(self) -> bool:
        with self.lock:
            if self.setup_state["stage"] in ("idle", "done", "error"):
                self._setup_requested = True
                self.setup_state = {"stage": "countdown", "message": "Sit at your seat and face the camera",
                                    "log": []}
                return True
        return False

    def stop(self) -> None:
        self.running = False

    def camera_command(self, action: str, **kw) -> None:
        """Queue a camera action; it runs on the tracking thread (which owns the camera)."""
        self._commands.append((action, kw))

    def camera_state(self) -> dict:
        room = room_setup.load_room()
        used = (room or {}).get("camera")
        matches = None
        if used:
            cur = self.actual_size
            matches = (used["index"] == self.cam_settings["index"]
                       and abs(used["aspect"] - cur[0] / cur[1]) < 0.01
                       and used.get("controls", {}).get("zoom") == self.cam_settings["controls"].get("zoom"))
        return {"settings": self.cam_settings, "actual": list(self.actual_size), "capabilities": self.capabilities,
                "busy": self.camera_busy, "message": self.camera_message, "setup_camera": used,
                "matches_setup": matches}

    # ---------------------------------------------------------------- internals
    def _set_setup(self, **kw) -> None:
        with self.lock:
            self.setup_state.update(kw)

    def _log(self, msg: str) -> None:
        print(f"setup: {msg}")
        with self.lock:
            self.setup_state["log"] = (self.setup_state.get("log", []) + [msg])[-20:]

    def _mapper_for(self, w: int, h: int):
        from vision.spatial import Calibration, CenterMapper

        if self.mapper is None or self.mapper.cx != w / 2:
            fc = next(s for s in self.cfg["speakers"] if s["name"] == "FC")
            fallback = float(np.hypot(*speaker_position(fc)[:2]))
            self.mapper = CenterMapper(w, h, self.vcfg["hfov_deg"], fallback,
                                       Calibration.load(room_setup.CAL_FILE), ipd_m=self.vcfg["ipd_m"])
        return self.mapper

    def _publish_preview(self, frame, face) -> None:
        import cv2

        from vision.overlay import draw_face

        scale = PREVIEW_WIDTH / frame.shape[1]
        small = cv2.resize(frame, (PREVIEW_WIDTH, int(frame.shape[0] * scale)))
        if face is not None:
            face_small = type(face)(bbox=face.bbox * scale, keypoints=face.keypoints * scale, score=face.score,
                                    frame_size=face.frame_size)
            draw_face(small, face_small)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.preview:
                self.jpeg = buf.tobytes()
                self.jpeg_seq += 1
                self.preview.notify_all()

    # ---------------------------------------------------------------- camera control
    def _open(self, settings: dict):
        from vision.camera import Camera

        cam = Camera(settings["index"], settings["width"], settings["height"], settings["fps"],
                     controls=settings.get("controls"))
        return cam

    def _switch(self, settings: dict) -> bool:
        """Reopen the camera with new settings; falls back to the previous ones on failure."""
        previous = dict(self.cam_settings)
        if self.cam is not None:
            self.cam.close()
        try:
            self.cam = self._open(settings)
        except RuntimeError as e:
            self.camera_message = f"{e} - back to camera {previous['index']}"
            self.cam = self._open(previous)
            settings = previous
            ok = False
        else:
            ok = True
        if settings["index"] != previous["index"]:
            self.capabilities = None
        self.cam_settings = settings
        self.actual_size = (self.cam.width, self.cam.height)
        CAMERA_FILE.write_text(json.dumps(settings, indent=2))
        self.mapper = None
        self.pipe.reset()
        self._last = None
        return ok

    def _snapshot_b64(self, frame) -> str:
        import base64

        import cv2

        small = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))
        return base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()).decode()

    def _run_command(self, action: str, kw: dict, frame) -> None:
        from vision.camera import list_cameras, widest_mode

        self.camera_message = ""
        if action == "open":
            if isinstance(kw.get("index"), str):  # network camera: remember the URL for the picker
                urls = self.cam_settings.get("network", [])
                if kw["index"] not in urls:
                    self.cam_settings["network"] = urls + [kw["index"]]
            settings = {**self.cam_settings, **{k: v for k, v in kw.items() if v is not None}}
            settings["controls"] = {**self.cam_settings.get("controls", {}), **kw.get("controls", {})} \
                if settings.get("index") == self.cam_settings["index"] else kw.get("controls", {})
            self.camera_busy = "switching camera"
            if self._switch(settings):
                self.camera_message = (f"camera {self.cam_settings['index']}: {self.actual_size[0]}x{self.actual_size[1]}"
                                       + ("" if list(self.actual_size) == [settings["width"], settings["height"]]
                                          else f" (asked for {settings['width']}x{settings['height']})"))
        elif action == "probe":
            self.camera_busy = "probing modes and controls"
            controls = self.cam.probe_controls()
            modes = self.cam.probe_modes()
            self._switch(self.cam_settings)  # probing changed the mode
            self.capabilities = {"modes": [list(m) for m in modes], "controls": controls}
            self.camera_message = (f"{len(modes)} modes; adjustable: {', '.join(controls) or 'none'}"
                                   + ("" if controls else " (phone cameras: choose the lens in the phone app)"))
        elif action == "widest":
            self.camera_busy = "finding the widest view"
            if self.capabilities is None:
                controls = self.cam.probe_controls()
                modes = self.cam.probe_modes()
                self.capabilities = {"modes": [list(m) for m in modes], "controls": controls}
            caps = self.capabilities
            settings = dict(self.cam_settings)
            changes = []
            if "zoom" in caps["controls"]:
                settings["controls"] = {**settings.get("controls", {}), "zoom": caps["controls"]["zoom"]["min"]}
                changes.append(f"zoom {caps['controls']['zoom']['min']:g} (minimum)")
            mode = widest_mode([tuple(m) for m in caps["modes"]])
            if mode:
                settings["width"], settings["height"] = mode
                changes.append(f"{mode[0]}x{mode[1]} (full-sensor aspect)")
            self._switch(settings)
            self.camera_message = ("widest view: " + ", ".join(changes) if changes else
                                   "this camera exposes no zoom or extra modes - on a phone, pick the "
                                   "ultra-wide lens in its camera app")
        elif action == "control":
            name, value = kw["name"], float(kw["value"])
            actual = self.cam.set_control(name, value)
            self.cam_settings["controls"] = {**self.cam_settings.get("controls", {}), name: actual}
            CAMERA_FILE.write_text(json.dumps(self.cam_settings, indent=2))
            self.camera_message = f"{name} = {actual:g}"
        elif action == "list":
            from vision.camera import snapshot

            self.camera_busy = "looking for cameras"
            active = self.cam_settings["index"]
            found = list_cameras(skip={active}, snapshots=True)
            for url in self.cam_settings.get("network", []):
                if url != active and (shot := snapshot(url)):
                    found.append(shot)
            entries = [{"index": self.cam_settings["index"], "name": f"camera {self.cam_settings['index']}",
                        "width": frame.shape[1], "height": frame.shape[0], "active": True,
                        "snapshot": self._snapshot_b64(frame)}]
            entries += [{"index": info.index, "name": info.name, "width": info.width, "height": info.height,
                         "active": False, "snapshot": self._snapshot_b64(f)} for info, f in found]
            self.cameras = sorted(entries, key=lambda e: (isinstance(e["index"], str), str(e["index"])))
            self.camera_message = f"{len(entries)} camera(s) found"
        self.camera_busy = ""

    def _run_build(self, capture: Path, live_faces: list) -> None:
        try:
            self._set_setup(stage="processing", message="Running the depth model and building the scan...")
            room_setup.build(capture, self.vcfg, log=self._log, live_faces=live_faces)
            # remember which camera view the setup used (the scan/calibration only fit that view)
            room = room_setup.load_room()
            w, h = self.actual_size
            room["camera"] = {"index": self.cam_settings["index"], "size": [w, h], "aspect": w / h,
                              "controls": self.cam_settings.get("controls", {})}
            ROOM_FILE.write_text(json.dumps(room, indent=2))
            self.mapper = None  # reload the new calibration on the next frame
            self.pipe.reset()
            with self.lock:
                self.room_version += 1
            self._set_setup(stage="done", message="Setup complete")
        except Exception as e:  # report any failure to the page
            self._log(f"error: {e}")
            self._set_setup(stage="error", message=str(e).splitlines()[0][:200])

    def run(self) -> None:
        import cv2

        from vision.camera import setup_mode

        self.cam = self._open(self.cam_settings)
        self.actual_size = (self.cam.width, self.cam.height)
        print(f"camera {self.cam_settings['index']}: {self.cam.width}x{self.cam.height}")
        frame = None
        frame_times, det_times = deque(maxlen=60), deque(maxlen=30)
        last_preview = 0.0
        countdown_until = capture_from = capture_until = None
        capture_dir = None
        saved = 0
        calib_faces: list = []

        while self.running:
            busy = countdown_until is not None or capture_until is not None
            if self._commands and frame is not None and not busy:
                action, kw = self._commands.popleft()
                try:
                    self._run_command(action, kw, frame)
                except Exception as e:  # keep tracking whatever the driver does
                    self.camera_busy, self.camera_message = "", f"{action} failed: {e}"
                continue
            item = self.cam.read()
            if item is None:
                time.sleep(0.05)
                continue
            ts, frame = item
            frame_times.append(ts)

            # ---- setup capture state machine (tracking continues outside the capture window)
            if self._setup_requested:
                self._setup_requested = False
                countdown_until = ts + SETUP_COUNTDOWN_S
                calib_faces = []
            if countdown_until is not None:
                left = countdown_until - ts
                if left > 0:
                    self._set_setup(message=f"Sit at your seat and face the camera... {left:.0f}", remaining=left)
                else:
                    countdown_until = None
                    modes = [tuple(m) for m in self.capabilities["modes"]] if self.capabilities else None
                    size = setup_mode(modes, self.actual_size)
                    self.cam.close()
                    self.cam = self._open({**self.cam_settings, "width": size[0], "height": size[1], "fps": 30})
                    capture_dir = room_setup.CAPTURES / ("setup-" + time.strftime("%Y%m%d-%H%M%S"))
                    (capture_dir / "frames").mkdir(parents=True, exist_ok=True)
                    capture_from = time.perf_counter() + SETUP_WARMUP_S
                    capture_until = capture_from + SETUP_CAPTURE_S
                    saved = 0
                    self._set_setup(stage="capturing", message="Hold still - capturing", remaining=0)
                    continue
            if capture_until is not None:
                if ts >= capture_from:
                    cv2.imwrite(str(capture_dir / "frames" / f"{saved:03d}.png"), frame)
                    saved += 1
                if ts >= capture_until:
                    capture_from = capture_until = None
                    shot = (frame.shape[1], frame.shape[0])
                    self.cam.close()
                    self.cam = self._open(self.cam_settings)
                    self.mapper = None
                    self.pipe.reset()
                    self._log(f"captured {saved} frames at {shot[0]}x{shot[1]} -> {capture_dir}")
                    threading.Thread(target=self._run_build, args=(capture_dir, calib_faces), daemon=True).start()
                if time.perf_counter() - last_preview > 1 / PREVIEW_FPS:
                    last_preview = time.perf_counter()
                    self._publish_preview(frame, None)
                continue

            # ---- tracking
            h, w = frame.shape[:2]
            mapper = self._mapper_for(w, h)
            face, processed = self.pipe.process(frame, ts)
            if processed and face is not None and countdown_until is not None \
                    and countdown_until - ts < SETUP_CALIB_S and self.pipe.tracker.mode == "track":
                calib_faces.append(face)
            if processed:
                det_times.append(ts)
                offset = mapper.map(face) if face is not None else None
                if offset is not None:
                    self._last = offset
                last = self._last
                wall_t = time.time() - (time.perf_counter() - ts)
                if self.publisher is not None and last is not None:
                    self.publisher.send(wall_t, offset is not None, last.x, last.y, last.z,
                                        -last.yaw_deg, last.score)
                state = {"tracking": offset is not None, "latency_ms": (time.perf_counter() - ts) * 1000.0,
                         "infer_ms": self.pipe.tracker.infer_ms, "mode": self.pipe.tracker.mode}
                if last is not None:
                    # image yaw: + = nose toward image right = listener turned to their left
                    state.update(x=last.x, y=last.y, z=last.z, yaw_deg=-last.yaw_deg, distance=last.distance,
                                 score=last.score)
                with self.lock:
                    self._state.update(state)
            with self.lock:
                self._state.update(camera_fps=rate(frame_times), detect_fps=rate(det_times),
                                   calibrated=mapper.cal.hfov_deg is not None,
                                   target_fps=self.pipe.rate.target_fps if self.pipe.rate else None)
            if time.perf_counter() - last_preview > 1 / PREVIEW_FPS:
                last_preview = time.perf_counter()
                self._publish_preview(frame, face)

        self.cam.close()
        self.pipe.close()
        if self.publisher:
            self.publisher.close()


# -------------------------------------------------------------------- web server

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


def serve(cfg: dict, tracker: LiveTracker | None, port: int = 8765, open_browser: bool = True) -> None:
    """Serve the app until Ctrl-C."""
    lock = threading.Lock()

    def defaults() -> list[dict]:
        return room_setup.default_speakers(room_setup.load_room(), cfg["speakers"])

    def room_payload() -> dict:
        room = room_setup.load_room()
        data = dict(room) if room else {"ready": False}
        data.setdefault("ready", room is not None and room_setup.CLOUD_XYZ.exists())
        if not data.get("speakers"):
            data["speakers"] = defaults()
        data["live"] = tracker is not None
        return data

    def save_speakers(speakers: list[dict]) -> None:
        with lock:
            room = room_setup.load_room() or {}
            room["speakers"] = _clean_speakers(speakers, defaults())
            ROOM_FILE.write_text(json.dumps(room, indent=2))

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

        def _file(self, path: Path, ctype: str):
            if path.exists():
                self._send(path.read_bytes(), ctype)
            else:
                self._send(b"", ctype, 404 if ctype.startswith("image") else 200)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._file(WEB_DIR / "app.html", "text/html; charset=utf-8")
            elif path == "/app.js":
                self._file(WEB_DIR / "app.js", "text/javascript; charset=utf-8")
            elif path == "/api/room":
                self._send(json.dumps(room_payload()).encode(), "application/json")
            elif path == "/api/defaults":
                self._send(json.dumps(defaults()).encode(), "application/json")
            elif path == "/api/cloud.xyz":
                self._file(room_setup.CLOUD_XYZ, "application/octet-stream")
            elif path == "/api/cloud.rgb":
                self._file(room_setup.CLOUD_RGB, "application/octet-stream")
            elif path == "/api/map.png":
                self._file(room_setup.MAP_FILE, "image/png")
            elif path == "/api/events":
                self._events()
            elif path == "/api/video.mjpg":
                self._video()
            elif path == "/api/suggest":
                self._suggest()
            elif path == "/api/cameras":
                self._send(json.dumps({"cameras": tracker.cameras if tracker else []}).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def _suggest(self):
            from .placement import centering_analysis, suggest_layout

            room = room_setup.load_room()
            if room is None or not room_setup.CLOUD_XYZ.exists():
                self._send(b'{"error": "run the room setup first"}', "application/json", 409)
                return
            points = np.fromfile(room_setup.CLOUD_XYZ, np.float32).reshape(-1, 3)
            result = suggest_layout(room, points, cfg["speakers"])
            current = room.get("speakers") or defaults()
            result["current_analysis"] = centering_analysis(current)
            self._send(json.dumps(result).encode(), "application/json")

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    state = tracker.snapshot() if tracker else {"live": False}
                    self.wfile.write(f"data: {json.dumps(state)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(1 / EVENTS_HZ)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _video(self):
            if tracker is None:
                self._send(b"no camera", "text/plain", 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = 0
            try:
                while True:
                    with tracker.preview:
                        tracker.preview.wait_for(lambda: tracker.jpeg_seq > seq, timeout=2.0)
                        jpeg, seq = tracker.jpeg, tracker.jpeg_seq
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
                data = json.loads(body) if body else {}
                if path == "/api/save":
                    save_speakers(data["speakers"])
                    self._send(b'{"ok": true}', "application/json")
                elif path == "/api/camera":
                    if tracker is None:
                        self._send(b'{"ok": false, "error": "no camera"}', "application/json", 409)
                        return
                    action = data.get("action")
                    if action not in ("open", "probe", "widest", "control", "list"):
                        raise ValueError(f"unknown camera action {action!r}")
                    kw = {k: data[k] for k in ("index", "width", "height", "fps", "controls", "name", "value")
                          if k in data}
                    for k in ("index", "width", "height", "fps"):
                        if k in kw and not (k == "index" and isinstance(kw[k], str) and "://" in kw[k]):
                            kw[k] = int(kw[k])
                    tracker.camera_command(action, **kw)
                    self._send(b'{"ok": true}', "application/json")
                elif path == "/api/setup":
                    ok = tracker is not None and tracker.request_setup()
                    self._send(json.dumps({"ok": ok}).encode(), "application/json", 200 if ok else 409)
                else:
                    self._send(b"not found", "text/plain", 404)
            except (ValueError, KeyError, TypeError) as e:
                self._send(json.dumps({"ok": False, "error": str(e)}).encode(), "application/json", 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"app: {url}  (Ctrl-C to quit)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
