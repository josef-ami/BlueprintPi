"""
Standalone calibration dashboard — lidar-first architecture.

Run this INSTEAD of main.py (one process may hold the camera + lidar at a
time). It opens both sensors, and on every poll re-runs the ROBOT'S OWN
functions with the live slider values:

  camera : detect_blobs -> blobs_to_detections     (colour + bearing)
  lidar  : filter_points -> cluster_points          (discrete obstacles)
  fuse   : stamp camera colour onto lidar obstacles

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

from worldstate import SharedState, CameraResult, LidarResult
import sensors.camera as camera
from sensors.lidar import (LidarThread, filter_points, cluster_points,
                           sector_min)
from main import fuse

PORT = 8080
JPEG_QUALITY = 70
RADAR_MAX_MM = 3000

app = Flask(__name__)

_live_lock = threading.Lock()
_live = camera.load_config()          # full config: hsv, hfov, ..., lidar block

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
# diagnostic distributions for the two threshold graphs
# (diagnostic-only: lives here, not in the robot's lidar.py)
# --------------------------------------------------------------------------

HIST_BIN_MM = 20
HIST_N_BINS = 25          # 0..500 mm, matching the slider ranges


def _histogram(values, bin_mm=HIST_BIN_MM, n_bins=HIST_N_BINS):
    counts = [0] * n_bins
    for v in values:
        b = int(v // bin_mm)
        b = 0 if b < 0 else (n_bins - 1 if b >= n_bins else b)
        counts[b] += 1
    return counts


def _nearest_neighbour_dists(points):
    """For each point, distance to its closest other point (mm). O(n^2), n<=360."""
    out = []
    for i, p in enumerate(points):
        best = float("inf")
        for j, o in enumerate(points):
            if i == j:
                continue
            dd = math.hypot(p[3] - o[3], p[4] - o[4])
            if dd < best:
                best = dd
        if not math.isinf(best):
            out.append(best)
    return out


def _consecutive_gaps(points):
    """Gaps (mm) between angularly-adjacent points — what clustering walks."""
    if len(points) < 2:
        return []
    pts = sorted(points, key=lambda p: p[0])
    gaps = [math.hypot(a[3] - b[3], a[4] - b[4]) for a, b in zip(pts, pts[1:])]
    gaps.append(math.hypot(pts[0][3] - pts[-1][3], pts[0][4] - pts[-1][4]))  # wrap
    return gaps


# --------------------------------------------------------------------------
# video views (unchanged — still needed for HSV tuning)
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

    # blobs: colour detections with their bearing (no distance in this model)
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
    """Raw ranges + quality — sensor-health view, no filtering, no camera."""
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


@app.route("/api/worldstate")
def worldstate():
    """
    The live fused view. Re-runs the robot's filter -> cluster -> fuse chain
    with the CURRENT slider values, so tuning previews exactly what the robot
    would compute.
    """
    cfg = live_cfg()
    lidar_cfg = cfg.get("lidar", {})

    # --- camera: colour detections with live HSV / HFOV ---
    frame = get_frame()
    cam_result = None
    detections = []
    if frame is not None:
        blobs = camera.detect_blobs(frame, cfg["hsv"], cfg["min_blob_area"])
        detections = camera.blobs_to_detections(blobs, camera.FRAME_W,
                                                cfg["hfov_deg"])
        cam_result = CameraResult(timestamp=time.time(), detections=detections)

    # --- lidar: re-filter + re-cluster the raw scan with live values ---
    _, lidar_result = shared.snapshot()
    ranges_out, survivors_out, front = [], [], None
    obstacles = []
    neighbor_hist = gap_hist = None
    if lidar_result is not None:
        ranges = lidar_result.ranges
        quals = getattr(lidar_result, "qualities", [0] * 360)

        # quality-passed points = isolation disabled; this is the SET the
        # isolation filter examines, so nearest-neighbour distances computed
        # on it tell you where to put neighbour_dist_mm.
        q_cfg = dict(lidar_cfg); q_cfg["min_neighbours"] = 0
        quality_passed = filter_points(ranges, quals, q_cfg)

        survivors = filter_points(ranges, quals, lidar_cfg)   # real fn, live cfg
        obstacles = cluster_points(survivors, lidar_cfg)       # real fn, live cfg

        live_lr = LidarResult(timestamp=time.time(), ranges=ranges,
                              qualities=list(quals), obstacles=obstacles)
        obstacles = fuse(cam_result, live_lr)

        ranges_out = [None if math.isinf(d) else round(d) for d in ranges]
        survivors_out = [[p[0], round(p[1]), p[2]] for p in survivors]  # deg,mm,q
        f = sector_min(ranges, 0, 15)
        front = None if math.isinf(f) else round(f)

        # distributions for the two tuning graphs
        neighbor_hist = {
            "counts": _histogram(_nearest_neighbour_dists(quality_passed)),
            "bin_mm": HIST_BIN_MM, "max_mm": HIST_BIN_MM * HIST_N_BINS,
            "threshold": lidar_cfg.get("neighbour_dist_mm", 150),
        }
        gap_hist = {
            "counts": _histogram(_consecutive_gaps(survivors)),
            "bin_mm": HIST_BIN_MM, "max_mm": HIST_BIN_MM * HIST_N_BINS,
            "threshold": lidar_cfg.get("cluster_gap_mm", 120),
        }

    return jsonify({
        "camera_ok": frame is not None,
        "camera_error": capture.error,
        "lidar_ok": lidar_result is not None,
        "front_mm": front,
        "ranges": ranges_out,
        "survivors": survivors_out,
        "obstacles": [
            {"color": o.color,
             "bearing_deg": round(o.bearing_deg, 1),
             "distance_mm": round(o.distance_mm),
             "width_deg": round(o.width_deg, 1),
             "point_count": o.point_count}
            for o in obstacles
        ],
        "detections": [
            {"color": d.color, "bearing_deg": round(d.bearing_deg, 1)}
            for d in detections
        ],
        "neighbor_hist": neighbor_hist,
        "gap_hist": gap_hist,
        "radar_max_mm": RADAR_MAX_MM,
    })


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
