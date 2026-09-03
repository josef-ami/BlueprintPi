"""
Standalone calibration dashboard.

Run this INSTEAD of main.py — one process at a time can hold the camera and
the lidar. It opens both sensors itself, runs its own mini fusion loop using
the SAME imported detection and fusion code the robot uses, and serves a web
UI on port 8080.

    python dashboard.py        then browse to http://<pi-hostname>:8080

Sliders preview live against the current frame; nothing is written to
config.json until you press Save. camera.py reads config.json at startup, so
restart main.py to pick up saved values.
"""

import json
import math
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from worldstate import SharedState
import sensors.camera as camera
from sensors.lidar import LidarThread, sector_min
from main import fuse                      # reuse the robot's own fusion

PORT = 8080
JPEG_QUALITY = 70
RADAR_MAX_MM = 3000

app = Flask(__name__)

# ---- live (unsaved) tuning values, edited by the sliders -------------------
_live_lock = threading.Lock()
_live = camera.load_config()               # seed from the saved config

# ---- latest raw frame, owned by the capture thread ------------------------
_frame_lock = threading.Lock()
_latest_frame = None                       # RGB ndarray or None

shared = SharedState()
lidar_thread = None
lidar_ok = False


def live_cfg():
    with _live_lock:
        return json.loads(json.dumps(_live))   # deep copy, cheap at this size


# --------------------------------------------------------------------------
# capture thread — keeps the newest raw frame available to every stream
# --------------------------------------------------------------------------

class CaptureThread(threading.Thread):
    def __init__(self):
        super().__init__(name="DashCapture", daemon=True)
        self._stop = threading.Event()
        self.cam = None
        self.error = None

    def run(self):
        global _latest_frame
        try:
            self.cam = camera.open_camera()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[capture] camera failed: {self.error}")
            return
        while not self._stop.is_set():
            try:
                swap = live_cfg().get("swap_rb", False)
                frame = camera.grab_rgb(self.cam, swap)
                with _frame_lock:
                    _latest_frame = frame
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            time.sleep(0.033)
        try:
            self.cam.stop()
        except Exception:
            pass

    def stop(self):
        self._stop.set()


capture = CaptureThread()


def get_frame():
    with _frame_lock:
        return None if _latest_frame is None else _latest_frame.copy()


# --------------------------------------------------------------------------
# view rendering — raw / mask / overlay / blobs
# --------------------------------------------------------------------------

BOX_BGR = {"RED": (55, 39, 238), "GREEN": (44, 214, 68), "MAGENTA": (255, 0, 255)}


def render(view, colour):
    """Return a BGR image for the requested view, or None if no frame yet."""
    frame = get_frame()
    if frame is None:
        return None
    cfg = live_cfg()
    hsv_cfg = cfg["hsv"]

    if view == "raw":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if view == "mask":
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = camera.build_mask(hsv_img, hsv_cfg.get(colour, []))
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if view == "overlay":
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = camera.build_mask(hsv_img, hsv_cfg.get(colour, []))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        tint = np.zeros_like(bgr)
        tint[:] = BOX_BGR.get(colour, (0, 255, 255))
        keep = cv2.bitwise_and(tint, tint, mask=mask)
        dim = (bgr * 0.35).astype(np.uint8)
        return np.where(mask[:, :, None] > 0,
                        cv2.addWeighted(bgr, 0.6, keep, 0.4, 0), dim)

    # blobs: every colour boxed, with its computed bearing
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    blobs = camera.detect_blobs(frame, hsv_cfg, cfg["min_blob_area"])
    for b in blobs:
        col = BOX_BGR.get(b["colour"], (255, 255, 255))
        cv2.rectangle(bgr, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]), col, 2)
        bearing = camera.px_to_bearing(b["cx"], camera.FRAME_W, cfg["hfov_deg"])
        cv2.putText(bgr, f"{bearing:+.1f}deg", (b["x"], max(14, b["y"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return bgr


def mjpeg(view, colour):
    while True:
        img = render(view, colour)
        if img is None:
            time.sleep(0.1)
            continue
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.05)


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream/<view>")
def stream(view):
    colour = request.args.get("colour", "RED")
    if view not in ("raw", "mask", "overlay", "blobs"):
        return "unknown view", 404
    return Response(mjpeg(view, colour),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({"live": live_cfg(), "saved": camera.load_config()})


@app.route("/api/live", methods=["POST"])
def set_live():
    """Update the in-memory tuning values the streams use. Not persisted."""
    global _live
    incoming = request.get_json(force=True)
    with _live_lock:
        _live = incoming
    return jsonify({"ok": True})


@app.route("/api/save", methods=["POST"])
def save():
    """Persist current live values to config.json."""
    cfg = live_cfg()
    try:
        camera.save_config(cfg)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"ok": True, "saved_at": time.time()})


@app.route("/api/worldstate")
def worldstate():
    """Run the robot's own fusion on the current frame + latest lidar scan."""
    cfg = live_cfg()
    frame = get_frame()
    cam_result = None
    if frame is not None:
        blobs = camera.detect_blobs(frame, cfg["hsv"], cfg["min_blob_area"])
        obstacles = camera.blobs_to_obstacles(blobs, camera.FRAME_W,
                                              cfg["hfov_deg"])
        from worldstate import CameraResult
        cam_result = CameraResult(timestamp=time.time(), obstacles=obstacles)
        shared.set_camera(cam_result)

    _, lidar_result = shared.snapshot()
    fused = fuse(cam_result, lidar_result)      # the robot's own fuse()

    ranges = []
    front = None
    if lidar_result is not None:
        ranges = [None if math.isinf(d) else round(d)
                  for d in lidar_result.ranges]
        f = sector_min(lidar_result.ranges, 0, 15)
        front = None if math.isinf(f) else round(f)

    return jsonify({
        "camera_ok": frame is not None,
        "camera_error": capture.error,
        "lidar_ok": lidar_result is not None,
        "front_mm": front,
        "ranges": ranges,
        "obstacles": [
            {"color": o.color,
             "bearing_deg": round(o.bearing_deg, 1),
             "distance_mm": (None if math.isinf(o.distance_mm)
                             else round(o.distance_mm)),
             "confidence": round(o.confidence, 2)}
            for o in fused
        ],
        "radar_max_mm": RADAR_MAX_MM,
    })


# --------------------------------------------------------------------------

def main():
    global lidar_thread, lidar_ok
    capture.start()
    try:
        lidar_thread = LidarThread(shared)
        lidar_thread.start()
        lidar_ok = True
    except Exception as e:
        print(f"[dashboard] lidar unavailable, HSV tuning still works: {e}")

    print(f"Dashboard on http://0.0.0.0:{PORT}  (stop main.py first)")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True,
                debug=False, use_reloader=False)
    finally:
        capture.stop()
        if lidar_thread is not None:
            lidar_thread.stop()
            lidar_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
