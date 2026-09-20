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

OBSTACLE RUN TAB
The second tab runs obstacle_lap.py's loop (class ObstacleLap — the same code
the CLI runs) inside this process, on the camera and lidar the dashboard
already owns, and shows everything it sees, decides and prints. A run reads
config.json from disk when you press Start, exactly as the CLI does: unsaved
slider changes do not reach it. The STM32 is driven through PERCEPT byte 15:
    Stop car      CMD STOP, held until TELEM shows STOPPED
    Rerun         CMD RERUN, held until TELEM shows BOOT (FINISH/STOPPED only)
    End session   STOP first, then the loop stops and the port is released
    Reboot STM32  STOP, end the session, CMD REBOOT x6, then watch the USB
                  device drop and come back
The open-round stream toggle and a run cannot hold /dev/ttyACM0 together.
"""

import collections
import io
import json
import math
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import serial

from worldstate import SharedState, CameraResult, Obstacle
import sensors.camera as camera
from sensors.lidar import LidarThread, sector_min, select_range, pick_bearing
from main import fuse
from openRound import (UART_PORT, UART_BAUD, SEND_HZ, STALE_S,
                       load_tol as load_saved_tol, read_three, pack_frame,
                       lidar_live)
from main import fuse_with
import obstacle_lap
from obstacle_lap import ObstacleLap
from control import percept_link as pl
from control.solver import AVOID_NONE, AVOID_TRACK

PORT = 8080
JPEG_QUALITY = 70
RADAR_MAX_MM = 3000

app = Flask(__name__)

_live_lock = threading.Lock()
_live = camera.load_config()

_frame_lock = threading.Lock()
_latest_frame = None      # RAW camera frame (no R/B swap); readers apply their own
_frame_seq = 0            # +1 per captured frame
_frame_t = 0.0            # time.time() of the capture

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
        global _latest_frame, _frame_seq, _frame_t
        try:
            self.cam = camera.open_camera()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[capture] camera failed: {self.error}")
            return
        while not self._stop.is_set():
            try:
                # Raw frame: the calibration views apply the LIVE swap_rb, a run
                # applies the swap_rb it read from config.json at Start.
                frame = camera.grab_rgb(self.cam, False)
                with _frame_lock:
                    _latest_frame = frame
                    _frame_seq += 1
                    _frame_t = time.time()
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


def _swap(frame, swap_rb):
    return np.ascontiguousarray(frame[:, :, ::-1]) if swap_rb else frame


def get_frame():
    """Latest frame with the LIVE (calibration) swap_rb applied."""
    with _frame_lock:
        f = None if _latest_frame is None else _latest_frame.copy()
    return None if f is None else _swap(f, live_cfg().get("swap_rb", False))


def get_frame_raw():
    """(raw frame, seq, capture time) — no copy; treat the array as read-only."""
    with _frame_lock:
        return _latest_frame, _frame_seq, _frame_t


# --------------------------------------------------------------------------
# STM32 stream (Open-round wire feed, toggled from the dashboard UI)
# --------------------------------------------------------------------------
#
# Reuses openRound.py's read_three/pack_frame/lidar_live against the SAME
# LidarThread instance this dashboard already owns (no second lidar
# connection). Unlike standalone openRound.py, the bearing tolerance is read
# from the LIVE (unsaved) dashboard config every send, per current
# instructions — no restart needed to pick up a slider change. If this
# thread is running, do not also run openRound.py standalone: both would
# try to open UART_PORT.

class StreamThread(threading.Thread):
    def __init__(self, lidar_thread):
        super().__init__(name="STM32Stream", daemon=True)
        self.lidar = lidar_thread
        self._stop = threading.Event()
        self.error = None
        self.ser = None
        # polled by /api/stream/status
        self.live = False
        self.f = self.l = self.r = None
        self.rev_per_s = 0
        self.queue_depth = 0

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.ser = serial.Serial(UART_PORT, UART_BAUD, timeout=0)
        except serial.SerialException as e:
            self.error = f"cannot open {UART_PORT}: {e}"
            print(f"[dashboard/stream] {self.error}")
            return

        period = 1.0 / SEND_HZ
        last_log = time.monotonic()
        last_rev = 0
        was_live = False
        rx_buf = b""

        print(f"[dashboard/stream] streaming to {UART_PORT} at {SEND_HZ} Hz "
              f"(live tol from dashboard config)")
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                live = lidar_live(self.lidar, now)
                self.live = live

                if live:
                    tol = live_cfg().get("lidar", {}).get(
                        "bearing_tol_deg", load_saved_tol())
                    f, l, r = read_three(self.lidar._ranges,
                                         self.lidar._quals, tol)
                    self.f, self.l, self.r = f, l, r
                    self.ser.write(pack_frame(f, l, r, self.lidar.rev))
                if live != was_live:
                    print("[dashboard/stream] lidar LIVE - streaming" if live
                          else "[dashboard/stream] lidar STALE - silent")
                    was_live = live

                n = self.ser.in_waiting
                if n:
                    rx_buf += self.ser.read(n)
                    *lines, rx_buf = rx_buf.split(b"\n")
                    for line in lines:
                        print("[stm32] " +
                             line.decode("ascii", "replace").rstrip())
                    if len(rx_buf) > 512:
                        rx_buf = b""

                if now - last_log >= 1.0:
                    self.rev_per_s = self.lidar.rev - last_rev
                    self.queue_depth = self.lidar.queue_depth()
                    last_rev = self.lidar.rev
                    last_log = now

                time.sleep(period)
        except serial.SerialException as e:
            self.error = f"serial error: {e} (STM32 unplugged/reset?)"
            print(f"[dashboard/stream] {self.error}")
        finally:
            self.live = False
            try:
                self.ser.close()
            except Exception:
                pass
            print("[dashboard/stream] stopped")


stream_thread = None


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

    if view == "floor":
        # The floor filter's own view, shown even when the filter is switched
        # off so it can be tuned before it is trusted.
        rejected, ctx_out = [], {}
        blobs = camera.detect_blobs(frame, hsv_cfg, cfg["min_blob_area"],
                                    floor=dict(cfg.get("floor", {}), enabled=True),
                                    rejected_out=rejected, context_out=ctx_out)
        return draw_floor_view(frame, ctx_out["floor"], blobs, rejected)

    # blobs: boxes with the atan-computed bearing; rejected ones dashed grey
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rejected = []
    blobs = camera.detect_blobs(frame, hsv_cfg, cfg["min_blob_area"],
                                floor=cfg.get("floor", {}), rejected_out=rejected)
    for b in blobs:
        col = BOX_BGR.get(b["colour"], (255, 255, 255))
        cv2.rectangle(bgr, (b["x"], b["y"]),
                      (b["x"] + b["w"], b["y"] + b["h"]), col, 2)
        bearing = camera.px_to_bearing(b["cx"], camera.FRAME_W, cfg["hfov_deg"],
                                       cy=b["cy"],
                                       offset_deg=cfg.get("camera_offset_deg", 0.0))
        cv2.putText(bgr, f"{bearing:+.1f}deg", (b["x"], max(14, b["y"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    draw_rejected(bgr, rejected)
    return bgr


# ---- floor-filter drawing, shared by the calibration and run views ----

REJECT_BGR = (150, 150, 150)
SHORT_REASON = {camera.REASON_NO_FLOOR: "not on floor",
                camera.REASON_NOT_LINKED: "wall between"}


def dashed_rect(img, x0, y0, x1, y1, col, thick=1, dash=6):
    for xa in range(x0, x1, dash * 2):
        cv2.line(img, (xa, y0), (min(xa + dash, x1), y0), col, thick)
        cv2.line(img, (xa, y1), (min(xa + dash, x1), y1), col, thick)
    for ya in range(y0, y1, dash * 2):
        cv2.line(img, (x0, ya), (x0, min(ya + dash, y1)), col, thick)
        cv2.line(img, (x1, ya), (x1, min(ya + dash, y1)), col, thick)


def outlined_text(img, text, org, col, scale=0.48):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def draw_rejected(bgr, rejected):
    """Blobs the floor filter threw out: dashed grey box + short reason."""
    for b in rejected:
        dashed_rect(bgr, b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"],
                    REJECT_BGR, 1)
        outlined_text(bgr, f"{b['colour'][0]} {SHORT_REASON.get(b['reason'], b['reason'])}",
                      (b["x"], max(14, b["y"] - 6)), REJECT_BGR, 0.45)


def draw_floor_view(frame_rgb, ctx, blobs, rejected):
    """Blue = white mat linked to the floor in front of the car; purple = white
    but cut off from it (e.g. beyond a wall); each blob's contact strip in
    green (pass) or red (fail); the ignored bottom rows hatched."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out = (bgr * 0.4).astype(np.uint8)
    floor = ctx.floor > 0
    linked = ctx.linked_floor() > 0
    out[linked] = (0.45 * bgr[linked] + 0.55 * np.array((235, 170, 60))).astype(np.uint8)
    cut = floor & ~linked
    out[cut] = (0.45 * bgr[cut] + 0.55 * np.array((170, 60, 170))).astype(np.uint8)
    if ctx.bottom < ctx.h:
        for xa in range(-ctx.h, ctx.w, 12):
            cv2.line(out, (xa, ctx.h), (xa + (ctx.h - ctx.bottom), ctx.bottom),
                     (90, 90, 90), 1)
        cv2.line(out, (0, ctx.bottom), (ctx.w, ctx.bottom), (200, 200, 200), 1)
    for b, ok in [(b, True) for b in blobs] + [(b, False) for b in rejected]:
        x0, y0, x1, y1 = ctx.strip(b["x"], b["y"], b["w"], b["h"])
        col = BOX_BGR.get(b["colour"], (255, 255, 255)) if ok else REJECT_BGR
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), col, 2 if ok else 1)
        if y1 > y0:
            cv2.rectangle(out, (x0, y0), (x1, y1), (60, 200, 60) if ok else (60, 60, 230), 2)
        frac = b.get("white_frac")
        label = (b.get("floor", "on floor") if ok else SHORT_REASON.get(b["reason"], b["reason"]))
        if frac is not None:
            label += f" {frac:.0%}"
        outlined_text(out, label, (b["x"], max(14, b["y"] - 6)), (235, 235, 235), 0.45)
    return out


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
    if view not in ("raw", "mask", "overlay", "blobs", "floor"):
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


@app.route("/api/stream/start", methods=["POST"])
def stream_start():
    global stream_thread
    if lidar_thread is None or not lidar_thread.is_alive():
        return jsonify({"ok": False,
                        "error": "lidar not running, cannot start stream"}), 400
    busy = run_mgr.port_busy(UART_PORT)
    if busy:
        return jsonify({"ok": False, "error": busy}), 409
    if stream_thread is not None and stream_thread.is_alive():
        return jsonify({"ok": False, "error": "already streaming"}), 400
    stream_thread = StreamThread(lidar_thread)
    stream_thread.start()
    time.sleep(0.1)  # let an immediate open failure surface
    if stream_thread.error:
        return jsonify({"ok": False, "error": stream_thread.error}), 500
    return jsonify({"ok": True})


@app.route("/api/stream/stop", methods=["POST"])
def stream_stop():
    global stream_thread
    if stream_thread is not None:
        stream_thread.stop()
        stream_thread.join(timeout=2.0)
    return jsonify({"ok": True})


@app.route("/api/stream/status")
def stream_status():
    if stream_thread is None or not stream_thread.is_alive():
        return jsonify({"running": False,
                        "error": stream_thread.error if stream_thread else None})
    return jsonify({
        "running": True,
        "live": stream_thread.live,
        "error": stream_thread.error,
        "f": stream_thread.f, "l": stream_thread.l, "r": stream_thread.r,
        "rev_per_s": stream_thread.rev_per_s,
        "queue_depth": stream_thread.queue_depth,
        "port": UART_PORT,
        "send_hz": SEND_HZ,
    })


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
        blobs = camera.detect_blobs(frame, cfg["hsv"], cfg["min_blob_area"],
                                    floor=cfg.get("floor", {}))
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


# ==========================================================================
# OBSTACLE RUN — obstacle_lap.py's loop, in this process, on these sensors
# ==========================================================================
#
# Nothing below re-implements the run: ObstacleLap (obstacle_lap.py) is the
# loop the CLI runs, CameraThread's detect_blobs -> blobs_to_obstacles is the
# camera stage, fuse_with is main.fuse. What IS different from the CLI, and
# why:
#   * frames come from this dashboard's CaptureThread (one camera owner);
#     VisionWorker runs CameraThread's exact pass on them and publishes the
#     same CameraResult into SharedState.
#   * lidar is this dashboard's LidarThread (one lidar owner).
#   * the loop runs with quiet=True: the status line reaches the console once
#     a second (like --quiet); the page shows the full-rate values as fields.

RUN_LOG_LINES = 4000
PREVIEW_IDLE_S = 3.0        # idle preview stops when the run tab stops asking
RUN_STREAM_FPS = 15
RUN_STOP_WAIT_S = 1.0       # End session / Reboot: wait this long for STOPPED
REBOOT_FRAMES = 2 * pl.CMD_CONFIRM_FRAMES
MASK_COLOURS = ("RED", "GREEN")
TILE_W, TILE_H = 214, 160     # three 4:3 tiles stacked beside the 640x480 view
FINISH_STATES = (pl.ST_FINISH, pl.ST_STOPPED)

_real_stdout = sys.stdout


class RunLog:
    """
    The run tab's console. emit() is the run's print(): the line goes to the
    ring the page reads AND to the real stdout (the journal). While a session
    runs, _StdoutTee also feeds in everything else this process prints — the
    [LidarThread] / [capture] lines a CLI terminal would show too.
    """

    def __init__(self, maxlen=RUN_LOG_LINES):
        self._lock = threading.Lock()
        self._lines = collections.deque(maxlen=maxlen)
        self._next = 1
        self.capture = False

    def add(self, text):
        with self._lock:
            self._lines.append((self._next, time.time(), text))
            self._next += 1

    def emit(self, text):
        text = str(text)
        for line in text.split("\n"):
            self.add(line)
        try:
            _real_stdout.write(text + "\n")
            _real_stdout.flush()
        except Exception:
            pass

    def since(self, idx, limit=600):
        with self._lock:
            out = [l for l in self._lines if l[0] > idx]
            last = self._next - 1
        return out[-limit:], last


runlog = RunLog()


class _StdoutTee(io.TextIOBase):
    """sys.stdout replacement: writes through, and while runlog.capture is on
    hands complete lines to runlog. Lines are assembled per thread because
    print() writes a line in several pieces."""

    def __init__(self, real):
        self._real = real
        self._tl = threading.local()

    def writable(self):
        return True

    def write(self, s):
        self._real.write(s)
        if runlog.capture:
            buf = getattr(self._tl, "buf", "") + s
            *lines, buf = buf.split("\n")
            self._tl.buf = buf
            for line in lines:
                runlog.add(line)
        return len(s)

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")


# ---------------------------------------------------------------- camera ---

def run_vision_cfg(cfg):
    """The camera parameters CameraThread takes from config.json."""
    return {"hsv": cfg["hsv"], "hfov": cfg["hfov_deg"],
            "min_area": cfg["min_blob_area"], "swap_rb": cfg.get("swap_rb", False),
            "offset": cfg.get("camera_offset_deg", 0.0),
            "floor": cfg.get("floor", {})}


def vision_pass(frame_raw, vcfg):
    """One CameraThread pass (detect_blobs with the floor filter ->
    blobs_to_obstacles), also keeping the pillar-only masks, the floor mask
    and the blobs the filter rejected."""
    frame = _swap(frame_raw, vcfg["swap_rb"])
    masks, rejected = {}, []
    blobs = camera.detect_blobs(frame, vcfg["hsv"], vcfg["min_area"], masks_out=masks,
                                floor=vcfg["floor"], rejected_out=rejected)
    obstacles = camera.blobs_to_obstacles(blobs, camera.FRAME_W, vcfg["hfov"],
                                          offset_deg=vcfg["offset"])
    return frame, blobs, masks, obstacles, rejected


class VisionWorker(threading.Thread):
    """
    Session mode (publish_to set): CameraThread on the dashboard's frames —
    config fixed at Start, one CameraResult into SharedState per new frame.
    Preview mode (no session): same pass, re-reading config.json whenever the
    file changes, and only while the run tab is being looked at.
    """

    def __init__(self, cfg=None, publish_to=None, name="RunVision"):
        super().__init__(name=name, daemon=True)
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.publish_to = publish_to
        self.fixed = cfg is not None
        self.vcfg = run_vision_cfg(cfg) if cfg is not None else None
        self._mtime = None
        self.latest = None
        self.wanted_until = 0.0
        self.error = None
        self.fps = 0.0

    def stop(self):
        self._halt.set()

    def get(self):
        with self._lock:
            return self.latest

    def _refresh_cfg(self):
        try:
            m = os.path.getmtime(camera.CONFIG_PATH)
            if m != self._mtime:
                self.vcfg = run_vision_cfg(camera.load_config())
                self._mtime = m
        except Exception as e:
            self.error = f"config.json: {type(e).__name__}: {e}"

    def run(self):
        last_seq, n, t_fps, t_cfg = -1, 0, time.time(), 0.0
        while not self._halt.is_set():
            now = time.time()
            if self.publish_to is None and now > self.wanted_until:
                time.sleep(0.1)
                continue
            if not self.fixed and now - t_cfg > 1.0:
                self._refresh_cfg()
                t_cfg = now
            raw, seq, _ = get_frame_raw()
            if raw is None or seq == last_seq or self.vcfg is None:
                time.sleep(0.005)
                continue
            last_seq = seq
            try:
                frame, blobs, masks, obstacles, rejected = vision_pass(raw, self.vcfg)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                time.sleep(0.1)
                continue
            res = CameraResult(timestamp=time.time(), obstacles=obstacles, ok=True)
            if self.publish_to is not None:
                self.publish_to.set_camera(res)
            with self._lock:
                self.latest = {"frame": frame, "blobs": blobs, "masks": masks,
                               "rejected": rejected, "floor": self.vcfg["floor"],
                               "cam_res": res, "seq": seq, "t": res.timestamp}
            n += 1
            if time.time() - t_fps >= 1.0:
                self.fps = n / (time.time() - t_fps)
                n, t_fps = 0, time.time()


preview_vision = VisionWorker(name="RunPreview")


# --------------------------------------------------------------- session ---

def _serial_hint(port, e):
    import glob
    found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return (f"[lap] cannot open {port}: {e}\n"
            f"      serial devices present: {', '.join(found) or 'none'}")


class RunSession:
    """One Start .. End session: a PerceptLink, a VisionWorker, an ObstacleLap
    and the thread running it. Mirrors obstacle_lap.main(), minus the camera
    and lidar threads, which belong to the dashboard."""

    def __init__(self, dry, no_avoid, port):
        self.cfg = camera.load_config()          # read at Start, like the CLI
        self.dry, self.no_avoid, self.port = dry, no_avoid, port
        self.link = pl.PerceptLink(port, obstacle_lap.UART_BAUD,
                                   on_log=lambda line: runlog.emit("[stm32] " + line))
        self.vision = VisionWorker(self.cfg, publish_to=shared, name="RunVision")
        self.lap = ObstacleLap(self.cfg, shared, lidar_thread, link=self.link,
                               dry=dry, no_avoid=no_avoid, quiet=True, port=port,
                               emit=runlog.emit)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="ObstacleLap",
                                       daemon=True)
        self.started_at = None
        self.ended_at = None
        self.ended_by = None         # "end" | "reboot"
        self.error = None

    def open(self):
        if not self.dry:
            try:
                self.link.open()
            except Exception as e:
                raise RuntimeError(_serial_hint(self.port, e))
            self.link.start()
        shared.set_camera(None)
        self.vision.start()
        self.started_at = time.time()
        runlog.capture = True
        self.lap.banner()
        self.thread.start()

    def _loop(self):
        try:
            self.lap.run(self.stop_event)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            runlog.emit("[lap] loop crashed: " + self.error)
            for line in traceback.format_exc().rstrip().split("\n"):
                runlog.emit("    " + line)

    def alive(self):
        return self.thread.is_alive()

    def send_direct(self, cmd, n):
        """Command-only frames straight from this thread (the loop is stopped
        or dead, so nothing else is writing)."""
        rev = lidar_thread.rev if lidar_thread is not None else 0
        for _ in range(n):
            self.link.send(left_mm=None, front_mm=None, right_mm=None, rev=rev,
                           lidar_ok=False, cam_ok=False, hello=False,
                           action=AVOID_NONE, green=False,
                           target_heading_deg=0.0, leg_mm=0, cmd=cmd)
            time.sleep(0.02)

    def stop_car_and_wait(self, timeout=RUN_STOP_WAIT_S):
        """Hold STOP until TELEM shows STOPPED/FINISH. True if acknowledged."""
        if self.dry:
            return True
        tel = self.link.telemetry()
        if tel.fresh() and tel.state in FINISH_STATES:
            return True
        t_req = time.time()
        if self.alive():
            self.lap.request(pl.CMD_STOP)
        else:
            runlog.emit("[lap] sending STOP")
            self.send_direct(pl.CMD_STOP, 2 * pl.CMD_CONFIRM_FRAMES)
        while time.time() - t_req < timeout:
            tel = self.link.telemetry()
            if tel.fresh() and tel.stamp > t_req and tel.state in FINISH_STATES:
                return True
            time.sleep(0.02)
        return False

    def stop_loop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def close(self, reason="end"):
        """The CLI's finally block, minus the sensors the dashboard keeps."""
        self.ended_by = reason
        runlog.emit("")
        runlog.emit("[lap] stopping...")
        self.stop_loop()
        if not self.dry:
            self.link.stop()
            self.link.close()
        self.vision.stop()
        self.vision.join(timeout=1.0)
        runlog.emit("[lap] done.")
        self.ended_at = time.time()
        runlog.capture = False


def _node_id(path):
    """Identity of a device node, or None if it is not there. A re-enumerated
    /dev/ttyACM0 is a new node even if it reappears under the same name."""
    try:
        st = os.stat(path)
        return (st.st_ino, st.st_rdev, st.st_ctime_ns)
    except OSError:
        return None


class RunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.session = None          # the active session
        self.last = None             # the most recent session, active or ended
        self.busy = None             # "ending" | "rebooting" | None
        self.reboot = {"phase": "idle", "message": "", "t": 0.0}

    # ---- guards ----

    def port_busy(self, port):
        s = self.session
        if s is not None and not s.dry and s.port == port:
            return f"an obstacle-run session holds {port}; end it first"
        if self.busy == "rebooting":
            return "an STM32 reboot is in progress"
        return None

    @staticmethod
    def _stream_holds(port):
        return (stream_thread is not None and stream_thread.is_alive()
                and port == UART_PORT)

    # ---- actions ----

    def start(self, dry, no_avoid, port):
        with self._lock:
            if self.session is not None:
                return False, "a session is already running"
            if self.busy:
                return False, f"busy ({self.busy})"
            if lidar_thread is None:
                return False, "the lidar thread failed to start; see the dashboard log"
            if not dry and self._stream_holds(port):
                return False, (f"the open-round stream holds {port}; stop it on the "
                               f"calibration tab first")
            try:
                sess = RunSession(dry, no_avoid, port)
                sess.open()
            except Exception as e:
                msg = str(e)
                runlog.emit(msg)
                return False, msg
            self.session = sess
            self.last = sess
            return True, None

    def stop_car(self):
        s = self.session
        if s is None:
            return False, "no session is running"
        if s.dry:
            return False, "--dry session: there is no STM32 to stop"
        if s.alive():
            s.lap.request(pl.CMD_STOP)
        else:
            threading.Thread(target=s.stop_car_and_wait, daemon=True).start()
        return True, None

    def rerun(self):
        s = self.session
        if s is None:
            return False, "no session is running"
        if s.dry:
            return False, "--dry session: there is no STM32 to rerun"
        tel = s.link.telemetry()
        if not (tel.fresh() and tel.state in FINISH_STATES):
            where = tel.state_name if tel.fresh() else "not reporting"
            return False, f"RERUN only works from FINISH or STOPPED (STM32: {where})"
        if not s.alive():
            return False, "the loop is not running; end the session and start again"
        s.lap.request(pl.CMD_RERUN)
        return True, None

    def end(self):
        with self._lock:
            s = self.session
            if s is None:
                return False, "no session is running"
            if self.busy:
                return False, f"busy ({self.busy})"
            self.busy = "ending"
        try:
            if not s.dry and not s.stop_car_and_wait():
                runlog.emit("[lap] WARNING: STOP was not acknowledged before the port "
                            "closed - the car may still be moving")
            s.close()
        finally:
            with self._lock:
                self.session = None
                self.busy = None
        return True, None

    def start_reboot(self, port):
        with self._lock:
            if self.busy:
                return False, f"busy ({self.busy})"
            s = self.session
            port = s.port if (s is not None and not s.dry) else port
            if self._stream_holds(port):
                return False, (f"the open-round stream holds {port} (and runs the "
                               f"open-round firmware); stop it first")
            self.busy = "rebooting"
            self.reboot = {"phase": "starting", "message": "", "t": time.time(),
                           "port": port}
        threading.Thread(target=self._reboot, args=(s, port), name="Reboot",
                         daemon=True).start()
        return True, None

    def _phase(self, phase, msg):
        self.reboot.update(phase=phase, message=msg, t=time.time())
        runlog.emit(f"[reboot] {msg}")

    def _reboot(self, s, port):
        try:
            before = _node_id(port)
            if before is None:
                self._phase("failed", f"{port} does not exist - is the STM32 plugged in?")
                return
            if s is not None and not s.dry:
                self._phase("stopping", "stopping the car, then ending the session")
                if not s.stop_car_and_wait():
                    runlog.emit("[reboot] STOP not acknowledged - rebooting anyway "
                                "(REBOOT cuts the motor first)")
                s.stop_loop()
                self._phase("sending", f"sending REBOOT x{REBOOT_FRAMES} on {port}")
                s.send_direct(pl.CMD_REBOOT, REBOOT_FRAMES)
                s.close("reboot")                 # releases the port now
            else:
                if s is not None:                 # a --dry session holds no port
                    s.close("reboot")
                self._phase("sending", f"sending REBOOT x{REBOOT_FRAMES} on {port}")
                pl.send_command_burst(port, pl.CMD_REBOOT, frames=REBOOT_FRAMES)
            with self._lock:
                if self.session is s:
                    self.session = None

            self._phase("waiting", f"waiting for the STM32 to drop off USB ({port})")
            t0 = time.time()
            while time.time() - t0 < 3.0:
                if _node_id(port) != before:
                    break
                time.sleep(0.02)
            else:
                self._phase("failed", f"{port} never dropped: the firmware did not take "
                                      f"REBOOT (older firmware without CMD 3? wrong port?)")
                return
            self._phase("waiting", f"{port} dropped - waiting for it to come back")
            while time.time() - t0 < 12.0:
                nid = _node_id(port)
                if nid is not None and nid != before:
                    self._phase("done", f"STM32 back on {port} after {time.time() - t0:.1f} s. "
                                        f"It zeroes its yaw now: keep the car still, then Start.")
                    return
                time.sleep(0.05)
            self._phase("failed", f"{port} did not come back within 12 s - check USB and power")
        except Exception as e:
            self._phase("failed", f"reboot failed: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                if self.session is s:
                    self.session = None
                self.busy = None


run_mgr = RunManager()


# ------------------------------------------------------------ state JSON ---

def _j(v, nd=None):
    """JSON-safe number: inf/nan -> None, optional rounding."""
    if v is None:
        return None
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, nd) if nd is not None else v
    return v


def _dc_json(obj):
    if obj is None:
        return None
    out = {}
    for k, v in vars(obj).items():
        out[k] = _j(v, 2) if isinstance(v, float) else v
    out["state_name"] = obj.state_name
    out["age_s"] = round(time.time() - obj.stamp, 3) if obj.stamp else None
    return out


def _obs_json(o):
    return {"color": o.color, "bearing_deg": round(o.bearing_deg, 1),
            "distance_mm": None if math.isinf(o.distance_mm) else round(o.distance_mm),
            "confidence": round(o.confidence, 2)}


def _rejected_json(vis):
    prod = vis.get() if vis else None
    if prod is None:
        return []
    return [{"colour": b["colour"], "reason": b["reason"],
             "white_frac": None if b.get("white_frac") is None else round(b["white_frac"], 2)}
            for b in prod.get("rejected", [])]


def _pick_json(p):
    return None if p is None else {"deg": p[0], "mm": round(p[1]), "q": p[2]}


def _fused_copies(cam_res, lidar_res, fusion):
    copies = [Obstacle(o.color, o.bearing_deg, float("inf"), o.confidence)
              for o in (cam_res.obstacles if cam_res else [])]
    return fuse_with(CameraResult(timestamp=0.0, obstacles=copies), lidar_res, *fusion)


def _saved_fusion(cfg):
    f = cfg.get("fusion", {})
    return (f.get("bearing_match_deg", 8), f.get("range_floor_mm", 0.0),
            f.get("gap_split_mm", 0.0))


_saved_cache = {"mtime": None, "cfg": None}


def saved_cfg():
    """config.json as on disk (what the next Start will use), cached by mtime."""
    try:
        m = os.path.getmtime(camera.CONFIG_PATH)
        if m != _saved_cache["mtime"]:
            _saved_cache.update(mtime=m, cfg=camera.load_config())
    except Exception:
        pass
    return _saved_cache["cfg"] or live_cfg()


def run_state(since):
    s = run_mgr.session
    last = run_mgr.last
    now = time.time()
    if s is None:
        preview_vision.wanted_until = now + PREVIEW_IDLE_S

    # which session to describe: the active one, else the last one (frozen)
    view = s if s is not None else last
    tick = view.lap.last if view is not None else None

    # ---- lidar + fusion: the run's own view while active, live preview otherwise
    if s is not None and tick is not None:
        lap = s.lap
        lidar_res = tick.lidar_res
        fusion = lap.fusion
        obstacles = tick.obstacles
        near_idx = tick.near_idx
        picks = tick.picks
        tol = lap.tol
        vis = s.vision
    else:
        cfg = saved_cfg()
        _, lidar_res = shared.snapshot()
        fusion = _saved_fusion(cfg)
        prod = preview_vision.get()
        fresh = prod is not None and (now - prod["t"]) < obstacle_lap.CAMERA_STALE_S
        obstacles = _fused_copies(prod["cam_res"], lidar_res, fusion) if fresh else []
        near_idx = -1
        tol = max(0, int(cfg.get("lidar", {}).get("bearing_tol_deg", 2)))
        picks = (None, None, None)
        if lidar_res is not None:
            picks = obstacle_lap.read_three_picks(lidar_res.ranges,
                                                  lidar_res.qualities, tol)
        vis = preview_vision

    lidar = {"ok": lidar_res is not None, "ranges": [], "qualities": [],
             "stats": None, "front_mm": None,
             "thread_alive": bool(lidar_thread is not None and lidar_thread.is_alive())}
    windows = []
    if lidar_res is not None:
        ranges, quals = lidar_res.ranges, lidar_res.qualities
        valid = [d for d in ranges if not math.isinf(d)]
        lidar.update(
            ranges=[None if math.isinf(d) else round(d) for d in ranges],
            qualities=list(quals),
            stats={"valid_count": len(valid), "total": len(ranges),
                   "min_mm": round(min(valid)) if valid else None,
                   "max_mm": round(max(valid)) if valid else None,
                   "age_s": round(now - lidar_res.timestamp, 2)})
        f = sector_min(ranges, 0, 15)
        lidar["front_mm"] = None if math.isinf(f) else round(f)
        match, floor_mm, gap_mm = fusion
        for o in obstacles:
            center = int(round(o.bearing_deg)) % 360
            windows.append(_window_diag(ranges, quals, center, match, floor_mm,
                                        gap_mm, o))
    lidar["picks"] = {"deg0": _pick_json(picks[0]), "deg90": _pick_json(picks[1]),
                      "deg270": _pick_json(picks[2])}
    lidar["tol"] = tol

    out = {
        "now": now,
        "active": s is not None,
        "busy": run_mgr.busy,
        "reboot": dict(run_mgr.reboot),
        "stream_busy": bool(stream_thread is not None and stream_thread.is_alive()),
        "default_port": obstacle_lap.UART_PORT,
        "camera_ok": get_frame_raw()[0] is not None,
        "camera_error": capture.error,
        "vision_fps": round(vis.fps, 1) if vis else 0.0,
        "vision_rejected": _rejected_json(vis),
        "floor_filter": camera.floor_params(
            (s.cfg if s is not None else saved_cfg()).get("floor", {})),
        "vision_error": vis.error if vis else None,
        "lidar": lidar,
        "fusion": {"bearing_match_deg": fusion[0], "range_floor_mm": fusion[1],
                   "gap_split_mm": fusion[2]},
        "obstacles": [_obs_json(o) for o in obstacles],
        "near_idx": near_idx,
        "windows": windows,
        "radar_max_mm": RADAR_MAX_MM,
        "session": None,
        "tick": None,
    }

    if view is not None:
        lap = view.lap
        a = lap.acfg
        out["session"] = {
            "active": s is not None,
            "dry": view.dry, "no_avoid": view.no_avoid, "port": view.port,
            "started_at": view.started_at, "ended_at": view.ended_at,
            "ended_by": view.ended_by,
            "elapsed_s": round((view.ended_at or now) - view.started_at, 1)
                         if view.started_at else None,
            "loop_alive": view.alive(), "error": view.error,
            "pending_cmd": pl.CMD_NAMES[lap.pending_command()],
            "cfg": {"send_hz": obstacle_lap.SEND_HZ, "tol": lap.tol,
                    "d_clear": round(a.clearance_mm, 1), "engage": a.engage_mm,
                    "freeze": a.freeze_mm, "max_bearing": a.max_bearing_deg,
                    "max_heading": a.max_heading_deg, "min_range": a.min_range_mm,
                    "tail_clear": a.tail_clear_mm, "margin": a.margin_mm,
                    "car_half_width": a.car_half_width_mm,
                    "pillar_half_width": a.pillar_half_width_mm,
                    "wall_guard": a.wall_guard, "wall_margin": a.wall_margin_mm,
                    "confirm_ticks": lap.sup.confirm_ticks,
                    "refractory_mm": lap.sup.refractory_mm,
                    "lidar_stale_s": obstacle_lap.LIDAR_STALE_S,
                    "camera_stale_s": obstacle_lap.CAMERA_STALE_S,
                    "cmd_timeout_s": obstacle_lap.CMD_TIMEOUT_S},
            "link": ({"rx_frames": view.link.rx_frames,
                      "rx_status": view.link.rx_status,
                      "rx_bad": view.link.rx_bad, "tx_frames": view.link.tx_frames,
                      "error": view.link.error} if not view.dry else None),
        }
    if tick is not None:
        # Since the supervisor dropped COMMIT (control/supervisor.py), the
        # only actions it reports are NONE and TRACK, and there is no leg or
        # frozen-since timer left in snapshot() to surface here any more.
        sup = dict(tick.sup)
        sup["refractory_left_mm"] = _j(max(0.0, sup["refractory_until"] - tick.telem.odo_mm), 0)
        sup["refractory_until"] = _j(sup["refractory_until"], 0)
        sup["heading_abs"] = _j(sup["heading_abs"], 2)
        out["tick"] = {
            "t": tick.t, "age_s": round(now - tick.t, 3),
            "lidar_live": tick.lidar_live, "cam_live": tick.cam_live,
            "front": _j(tick.front, 0), "left": _j(tick.left, 0),
            "right": _j(tick.right, 0),
            "picks": [_pick_json(p) for p in tick.picks],
            "obs_count": len(tick.obstacles),
            "near": (_obs_json(tick.obstacles[tick.near_idx])
                     if tick.near_idx >= 0 else None),
            "action": tick.action, "action_name": obstacle_lap._aname(tick.action),
            "color": tick.color, "heading": _j(tick.heading, 2),
            "leg": _j(tick.leg, 0), "note": tick.note,
            "sent": tick.sent, "hello_sent": tick.hello_sent,
            "cmd": pl.CMD_NAMES.get(tick.cmd, str(tick.cmd)),
            "rev": tick.rev, "rev_per_s": tick.rev_per_s,
            "queue_depth": tick.queue_depth, "link_state": tick.link_state,
            "tick_ms": round(tick.tick_ms, 2),
            "telem": _dc_json(tick.telem),
            "status": _dc_json(tick.status),
            "sup": sup,
        }

    lines, last_idx = runlog.since(since)
    out["log"] = {"last": last_idx,
                  "lines": [[i, round(t, 3), text] for i, t, text in lines]}
    return out


# ------------------------------------------------------- composite video ---

def _mask_tile(mask, colour, w, h, title=None, bgr=None):
    tile = np.zeros((h, w, 3), np.uint8)
    if mask is not None:
        small = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        tile[small > 0] = bgr or BOX_BGR.get(colour, (255, 255, 255))
        cover = 100.0 * float(np.count_nonzero(mask)) / mask.size
        text = f"{title or colour + ' pillars'}  {cover:.1f}%"
    else:
        text = f"{title or colour}: filter off"
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (235, 235, 235), 1, cv2.LINE_AA)
    return tile


def render_run_composite(prod, sess):
    """Detections (640x480) | RED pillars, GREEN pillars, white floor (stacked)."""
    W, H = camera.FRAME_W, camera.FRAME_H
    det = cv2.cvtColor(prod["frame"], cv2.COLOR_RGB2BGR)
    tick = sess.lap.last if sess is not None else None
    if tick is not None and tick.cam_res is prod["cam_res"]:
        labelled, near = tick.obstacles, tick.near_idx      # exactly what the run fused
    else:
        fusion = sess.lap.fusion if sess is not None else _saved_fusion(saved_cfg())
        _, lidar_res = shared.snapshot()
        labelled, near = _fused_copies(prod["cam_res"], lidar_res, fusion), -1
    cx = W // 2
    cv2.line(det, (cx, 0), (cx, H), (200, 200, 200), 1)
    for i, b in enumerate(prod["blobs"]):
        col = BOX_BGR.get(b["colour"], (255, 255, 255))
        o = labelled[i] if i < len(labelled) else None
        target = (i == near)
        cv2.rectangle(det, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                      col, 4 if target else 2)
        label = b["colour"][0]
        if o is not None:
            label += f" {o.bearing_deg:+.1f}deg"
            if math.isfinite(o.distance_mm):
                label += f" {o.distance_mm:.0f}mm"
        if target:          # the supervisor's pick: tracked only in TRACK
            label = ("TRACKING " if tick.action == AVOID_TRACK else "NEAREST ") + label
        y = max(16, b["y"] - 7)
        cv2.putText(det, label, (b["x"], y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(det, label, (b["x"], y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    col, 1, cv2.LINE_AA)
    draw_rejected(det, prod.get("rejected", []))
    tiles = [_mask_tile(prod["masks"].get(c), c, TILE_W, TILE_H) for c in MASK_COLOURS]
    tiles.append(_mask_tile(prod["masks"].get("FLOOR"), "FLOOR", TILE_W, TILE_H,
                            title="white floor", bgr=(235, 235, 235)))
    return np.hstack([det, np.vstack(tiles)])


def _placeholder(text):
    img = np.full((camera.FRAME_H, camera.FRAME_W + TILE_W, 3), 24, np.uint8)
    cv2.putText(img, text, (24, camera.FRAME_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def run_mjpeg():
    period = 1.0 / RUN_STREAM_FPS
    last_seq, last_emit = None, 0.0
    while True:
        t0 = time.time()
        sess = run_mgr.session
        if sess is not None:
            prod = sess.vision.get()
        else:
            preview_vision.wanted_until = t0 + PREVIEW_IDLE_S
            prod = preview_vision.get()
        if prod is None or t0 - prod["t"] > 2.0:
            if t0 - last_emit < 0.5:
                time.sleep(0.05)
                continue
            img = _placeholder(f"no camera frame  ({capture.error})" if capture.error
                               else "waiting for camera frames...")
            last_seq = None
        elif prod["seq"] == last_seq:
            time.sleep(0.01)
            continue
        else:
            img = render_run_composite(prod, sess)
            last_seq = prod["seq"]
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            last_emit = time.time()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


# ----------------------------------------------------------------- routes ---

def _req_json():
    try:
        return request.get_json(force=True, silent=True) or {}
    except Exception:
        return {}


def _reply(ok, err, code=409):
    return (jsonify({"ok": True}) if ok
            else (jsonify({"ok": False, "error": err}), code))


@app.route("/run/stream")
def run_stream():
    return Response(run_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/run/state")
def api_run_state():
    try:
        since = int(request.args.get("since", 0))
    except (TypeError, ValueError):
        since = 0
    return jsonify(run_state(since))


@app.route("/api/run/start", methods=["POST"])
def api_run_start():
    body = _req_json()
    port = str(body.get("port") or obstacle_lap.UART_PORT).strip()
    ok, err = run_mgr.start(bool(body.get("dry")), bool(body.get("no_avoid")), port)
    return _reply(ok, err)


@app.route("/api/run/stop_car", methods=["POST"])
def api_run_stop_car():
    return _reply(*run_mgr.stop_car())


@app.route("/api/run/rerun", methods=["POST"])
def api_run_rerun():
    return _reply(*run_mgr.rerun())


@app.route("/api/run/end", methods=["POST"])
def api_run_end():
    return _reply(*run_mgr.end())


@app.route("/api/run/reboot", methods=["POST"])
def api_run_reboot():
    body = _req_json()
    port = str(body.get("port") or obstacle_lap.UART_PORT).strip()
    return _reply(*run_mgr.start_reboot(port))


# --------------------------------------------------------------------------

def main():
    global lidar_thread
    import signal
    # systemd stops robodash.service with SIGTERM. Python's default handler
    # would exit without running the finally below, leaving a car that is
    # mid-run driving with nobody sending STOP. Exit normally instead.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.stdout = _StdoutTee(sys.stdout)     # feeds the run console (see RunLog)
    capture.start()
    preview_vision.start()
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
        if run_mgr.session is not None:
            try:
                run_mgr.end()               # STOP the car, release the port
            except Exception as e:
                print(f"[dashboard] ending the run session failed: {e}")
        preview_vision.stop()
        capture.stop()
        if stream_thread is not None:
            stream_thread.stop()
            stream_thread.join(timeout=2.0)
        if lidar_thread is not None:
            lidar_thread.stop()
            lidar_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
