"""
Standalone calibration dashboard — camera-first architecture.

Run this INSTEAD of main.py (one process may hold the camera + lidar at a
time). On every poll it re-runs the ROBOT'S OWN functions with the live
slider values:

  camera : detect_blobs -> blobs_to_obstacles   (colour + bearing via atan)
  fuse   : attach a lidar distance to each camera obstacle by bearing

Nothing here reimplements that logic — it imports it, so what you tune is
exactly what the robot runs. Sliders preview live; Save writes config.json;
restart main.py to apply.
"""

import json
import math
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from worldstate import SharedState, CameraResult
import sensors.camera as camera
from sensors.lidar import LidarThread, sector_min, select_range, pick_bearing
from main import fuse

PORT = 8080
JPEG_QUALITY = 70
RADAR_MAX_MM = 3000

app = Flask(__name__)

_live_lock = threading.Lock()
_live = camera.load_config()

_frame_lock = threading.Lock()
_latest_frame = None

shared = SharedState()
lidar_thread = None


def live_cfg():
    with _live_lock:
        return json.loads(json.dumps(_live))


# --------------------------------------------------------------------------
# capture thread
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
# video views
# --------------------------------------------------------------------------

BOX_BGR = {"RED": (55, 39, 238), "GREEN": (44, 214, 68), "MAGENTA": (255, 0, 255)}


def render(view, colour):
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

    # blobs: boxes with the atan-computed bearing
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    blobs = camera.detect_blobs(frame, hsv_cfg, cfg["min_blob_area"])
    for b in blobs:
        col = BOX_BGR.get(b["colour"], (255, 255, 255))
        cv2.rectangle(bgr, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]), col, 2)
        bearing = camera.px_to_bearing(b["cx"], camera.FRAME_W, cfg["hfov_deg"],
                                       cy=b["cy"],
                                       offset_deg=cfg.get("camera_offset_deg", 0.0))
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
    global _live
    with _live_lock:
        _live = request.get_json(force=True)
    return jsonify({"ok": True})


@app.route("/api/save", methods=["POST"])
def save():
    cfg = live_cfg()
    try:
        camera.save_config(cfg)
    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"ok": True, "saved_at": time.time()})


@app.route("/api/lidar_raw")
def lidar_raw():
    """Raw ranges + quality — sensor-health view, no camera involved."""
    _, lidar_result = shared.snapshot()
    if lidar_result is None:
        return jsonify({"ok": False, "ranges": [], "qualities": [], "stats": None})
    ranges = lidar_result.ranges
    quals = getattr(lidar_result, "qualities", [0] * 360)
    valid = [d for d in ranges if not math.isinf(d)]
    return jsonify({
        "ok": True,
        "ranges": [None if math.isinf(d) else round(d) for d in ranges],
        "qualities": list(quals),
        "stats": {
            "valid_count": len(valid),
            "total": len(ranges),
            "min_mm": round(min(valid)) if valid else None,
            "max_mm": round(max(valid)) if valid else None,
            "age_s": round(time.time() - lidar_result.timestamp, 2),
        },
        "radar_max_mm": RADAR_MAX_MM,
    })


BEARING_TARGETS = {"deg0": 0, "deg90": 90, "deg270": 270}   # robot frame: fwd / left / right


@app.route("/api/bearings")
def bearings():
    """Highest-confidence non-inf return near each of 0/90/270 deg, within an
    adjustable +/- tolerance. Sensor-health view, no camera involved. `tol` is
    integer degrees (query arg); ranges are already robot-frame (mount offset
    folded in at ingestion), so the targets need no further correction."""
    try:
        tol = max(0, int(request.args.get("tol", 2)))
    except (TypeError, ValueError):
        tol = 2
    _, lidar_result = shared.snapshot()
    if lidar_result is None:
        return jsonify({"ok": False, "tol": tol, "picks": {}})
    ranges = lidar_result.ranges
    quals = getattr(lidar_result, "qualities", [0] * 360)
    picks = {}
    for key, tgt in BEARING_TARGETS.items():
        p = pick_bearing(ranges, quals, tgt, tol)
        picks[key] = (None if p is None
                      else {"deg": p[0], "mm": round(p[1]), "q": p[2]})
    return jsonify({"ok": True, "tol": tol, "picks": picks})


@app.route("/api/worldstate")
def worldstate():
    """
    Live fused view: detect camera obstacles with current HSV/HFOV, then
    attach a lidar distance to each with the robot's select_range — called
    directly here (not via fuse) so the live fusion sliders preview. The
    selection logic is identical to what the robot runs; only the parameter
    source differs (live config vs. import-time config).
    """
    cfg = live_cfg()
    fcfg = cfg.get("fusion", {})
    match_deg = fcfg.get("bearing_match_deg", 8)
    floor_mm = fcfg.get("range_floor_mm", 0.0)
    gap_mm = fcfg.get("gap_split_mm", 0.0)

    frame = get_frame()
    obstacles = []
    if frame is not None:
        blobs = camera.detect_blobs(frame, cfg["hsv"], cfg["min_blob_area"])
        obstacles = camera.blobs_to_obstacles(blobs, camera.FRAME_W,
                                              cfg["hfov_deg"],
                                              offset_deg=cfg.get("camera_offset_deg", 0.0))

    _, lidar_result = shared.snapshot()
    ranges_out, front = [], None
    windows = []
    if lidar_result is not None:
        ranges = lidar_result.ranges
        quals = getattr(lidar_result, "qualities", [0] * 360)
        for obs in obstacles:                       # live-parameter selection
            center = int(round(obs.bearing_deg)) % 360
            obs.distance_mm = select_range(ranges, center, match_deg,
                                           floor_mm=floor_mm, gap_split_mm=gap_mm)
            windows.append(_window_diag(ranges, quals, center, match_deg,
                                        floor_mm, gap_mm, obs))
        ranges_out = [None if math.isinf(d) else round(d) for d in ranges]
        f = sector_min(ranges, 0, 15)
        front = None if math.isinf(f) else round(f)

    return jsonify({
        "camera_ok": frame is not None,
        "camera_error": capture.error,
        "lidar_ok": lidar_result is not None,
        "front_mm": front,
        "ranges": ranges_out,
        "obstacles": [
            {"color": o.color,
             "bearing_deg": round(o.bearing_deg, 1),
             "distance_mm": (None if math.isinf(o.distance_mm)
                             else round(o.distance_mm)),
             "confidence": round(o.confidence, 2)}
            for o in obstacles
        ],
        "windows": windows,
        "radar_max_mm": RADAR_MAX_MM,
    })


def _window_diag(ranges, quals, center, half_width, floor_mm, gap_mm, obs):
    """
    Decompose the select_range decision for one obstacle's bearing window, so
    the UI can show WHY a distance was chosen — and specifically whether the
    wall was returned because the pillar was absent, mis-aligned, floored out,
    or not gap-split from the wall.

    Each point: {off (deg from centre), deg, mm, q, passed_floor, kept}
      kept = in the near cluster select_range actually took its min from.
    """
    pts = []
    for offset in range(-half_width, half_width + 1):
        deg = (center + offset) % 360
        d = ranges[deg]
        if math.isinf(d):
            continue
        pts.append({"off": offset, "deg": deg, "mm": round(d),
                    "q": quals[deg], "passed_floor": d >= floor_mm, "kept": False})

    # replicate select_range's kept-set exactly
    survivors = sorted([p for p in pts if p["passed_floor"]], key=lambda p: p["mm"])
    near = survivors
    if gap_mm > 0 and survivors:
        cut = len(survivors)
        for i in range(1, len(survivors)):
            if survivors[i]["mm"] - survivors[i - 1]["mm"] >= gap_mm:
                cut = i
                break
        near = survivors[:cut]
    near_mm = {p["mm"] for p in near}
    for p in pts:
        p["kept"] = p["passed_floor"] and p["mm"] in near_mm and (
            not near or p["mm"] <= max(near_mm))

    chosen = None if math.isinf(obs.distance_mm) else round(obs.distance_mm)
    floored = [p for p in pts if not p["passed_floor"]]
    passed = [p for p in pts if p["passed_floor"]]
    verdict = "ok"
    if not pts:
        verdict = "empty"                    # nothing in window at all
    elif chosen is None:
        verdict = "all_floored" if floored else "empty"
    else:
        # "far_only": the camera says an obstacle is here, but every floored-
        # in return sits in a single cluster with NO near member — so there's
        # no distinct pillar return, just the wall. Detect by: no internal gap
        # (gap-split found no pillar/wall split) AND the nearest survivor is
        # beyond a pillar-plausible distance.
        pm = sorted(p["mm"] for p in passed)
        has_internal_gap = (any(pm[i] - pm[i - 1] >= gap_mm
                                for i in range(1, len(pm)))
                            if (gap_mm > 0 and len(pm) > 1) else False)
        FAR_HINT_MM = 1000
        if pm and not has_internal_gap and pm[0] >= FAR_HINT_MM:
            verdict = "far_only"

    return {
        "color": obs.color,
        "bearing_deg": round(obs.bearing_deg, 1),
        "center_deg": center,
        "chosen_mm": chosen,
        "points": pts,
        "n_floored": len(floored),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------

def main():
    global lidar_thread
    capture.start()
    try:
        lidar_thread = LidarThread(shared)
        lidar_thread.start()
    except Exception as e:
        print(f"[dashboard] lidar unavailable, camera tuning still works: {e}")

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
