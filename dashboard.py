#!/usr/bin/env python3
"""
dashboard.py - the one page that runs and tunes the car.

Run this INSTEAD of anything else: one process may hold the camera and the
LiDAR at a time. It replaces the three pages this repo used to have
(dashboard on :8080, obstacleRound's own page on :5000, calibrate_vision's on
:5000) - they could not run together, which meant you could not calibrate
while looking at what the detector made of it.

    python3 dashboard.py        then open http://<pi>:8080

TABS
  Run        camera with the detector's boxes, Start / Stop, the FSM's state
             and everything it is deciding on, the STM32's telemetry, console
  Tune       every tunable, live. Pi values apply on the next tick; STM32
             values are queued and pushed by the loop, so tuning can never
             stall the 50 Hz feed. Save writes config.json
  Calibrate  click the mat and the pillars to fit Lab colour ranges, with the
             REAL detector running on the live frame beside it so you can see
             what each change accepts and rejects

NOTHING HERE REIMPLEMENTS THE ROBOT. The loop is control.obstacle_round's
ObstacleRound - the same class the CLI runs - on the camera and LiDAR this
process already owns. What you tune is what it runs.

WHO HOLDS THE PORT
  A run session holds /dev/ttyACM0. Reboot STM32 needs it free, so it ends
  the session first. The car is always sent a Stop and given a chance to
  acknowledge it before the port is released.
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

import params as prm
import sensors.camera as camera
from calibration.lab_fit import CLASSES, FITTED_NAMES, LabCalibration
from control.intent import SteerMode
from control.link import Link, CMD_REBOOT
from control.obstacle_round import ObstacleRound
from sensors.lidar import LidarThread, cones, lidar_candidates, sector_min
from worldstate import SharedState

PORT = 8080
JPEG_QUALITY = 70
RADAR_MAX_MM = 3000
LOG_LINES = 4000
PREVIEW_IDLE_S = 3.0
STREAM_FPS = 15
TILE_W, TILE_H = 214, 160

BOX_BGR = {"RED": (55, 39, 238), "GREEN": (44, 214, 68)}
REJECT_BGR = (150, 150, 150)

app = Flask(__name__)

# ---- the one live parameter set, shared by the page and the loop ----
PI = prm.PiParams()
STM = prm.StmParams()

shared = SharedState()
lidar_thread = None
vision = None
cal = LabCalibration()


# ==========================================================================
# capture - one owner of the camera, frames handed to whoever wants them
# ==========================================================================

class CaptureThread(threading.Thread):
    def __init__(self):
        super().__init__(name="Capture", daemon=True)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.cam = None
        self.error = None
        self.frame = None          # RAW, no R/B swap; readers apply their own
        self.seq = 0

    def run(self):
        try:
            self.cam = camera.open_camera()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[capture] camera failed: {self.error}")
            return
        while not self._stop.is_set():
            try:
                f = camera.grab_rgb(self.cam, False)
                with self._lock:
                    self.frame = f
                    self.seq += 1
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            time.sleep(0.033)
        try:
            self.cam.stop()
        except Exception:
            pass

    def get(self):
        with self._lock:
            return self.frame, self.seq

    def stop(self):
        self._stop.set()


capture = CaptureThread()


# ==========================================================================
# console
# ==========================================================================

_real_stdout = sys.stdout


class RunLog:
    """The page's console. emit() is the loop's print(): the line goes to the
    ring the page reads AND to the real stdout (the journal)."""

    def __init__(self, maxlen=LOG_LINES):
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


# ==========================================================================
# session
# ==========================================================================

def _serial_hint(port, e):
    import glob
    found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    return (f"[run] cannot open {port}: {e}\n"
            f"      serial devices present: {', '.join(found) or 'none'}")


class RunSession:
    """One Start .. End session: a Link, an ObstacleRound and its thread.
    Mirrors control.obstacle_round.main(), minus the camera and LiDAR, which
    belong to the dashboard."""

    def __init__(self, dry, port, baud):
        self.dry, self.port, self.baud = dry, port, baud
        self.link = None
        if not dry:
            self.link = Link(port, baud,
                             on_log=lambda s: runlog.emit("[stm32] " + s),
                             on_param=self._param_line)
        self.loop = ObstacleRound(PI, STM, shared, lidar_thread, link=self.link,
                                  dry=dry, quiet=True, emit=runlog.emit)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ObstacleRound",
                                       daemon=True)
        self.started_at = None
        self.ended_at = None
        self.ended_by = None
        self.error = None
        self._pushed = False

    def _param_line(self, line):
        note = STM.on_line(line)
        if note:
            runlog.emit(note)
        if STM.synced and not self._pushed:
            self._pushed = True
            STM.queue_all()
            runlog.emit("[pi] pushing saved tuning to the STM32")
        if not STM.synced:
            self._pushed = False

    def open(self):
        if not self.dry:
            try:
                self.link.open()
            except Exception as e:
                raise RuntimeError(_serial_hint(self.port, e))
            self.link.start()
            STM.request_dump()
        self.started_at = time.time()
        runlog.capture = True
        self.loop.banner()
        self.thread.start()

    def _run(self):
        try:
            self.loop.run(self.stop_event)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            runlog.emit("[run] loop crashed: " + self.error)
            for line in traceback.format_exc().rstrip().split("\n"):
                runlog.emit("    " + line)

    def alive(self):
        return self.thread.is_alive()

    def stop_car_and_wait(self, timeout=1.0):
        """Ask the FSM to stop and give it time to command it. True if the
        STM32 reported the motor off."""
        self.loop.stop()
        t0 = time.time()
        while time.time() - t0 < timeout:
            tick = self.loop.last
            if tick is not None and tick.intent is not None \
                    and tick.intent.mode == SteerMode.STOP:
                tel = tick.telem
                if self.dry or (tel is not None and not tel.enabled):
                    return True
            time.sleep(0.02)
        return False

    def close(self, reason="end"):
        self.ended_by = reason
        runlog.emit("")
        runlog.emit("[run] stopping...")
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if not self.dry:
            self.link.stop()
            self.link.close()
        runlog.emit("[run] done.")
        self.ended_at = time.time()
        runlog.capture = False


def _node_id(path):
    """Identity of a device node, or None. A re-enumerated /dev/ttyACM0 is a
    new node even when it comes back under the same name."""
    try:
        st = os.stat(path)
        return (st.st_ino, st.st_rdev, st.st_ctime_ns)
    except OSError:
        return None


class RunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.session = None
        self.last = None
        self.busy = None              # "ending" | "rebooting" | None
        self.reboot = {"phase": "idle", "message": "", "t": 0.0}

    def start(self, dry, port, baud):
        with self._lock:
            if self.session is not None:
                return False, "a session is already running"
            if self.busy:
                return False, f"busy ({self.busy})"
            if lidar_thread is None:
                return False, "the lidar thread failed to start; see the log"
            try:
                s = RunSession(dry, port, baud)
                s.open()
            except Exception as e:
                runlog.emit(str(e))
                return False, str(e)
            self.session = s
            self.last = s
            return True, None

    def car_start(self):
        s = self.session
        if s is None:
            return False, "no session is running"
        s.loop.start()
        return True, None

    def car_stop(self):
        s = self.session
        if s is None:
            return False, "no session is running"
        s.loop.stop()
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
            if not s.stop_car_and_wait():
                runlog.emit("[run] WARNING: the STM32 did not confirm the motor "
                            "was off before the port closed")
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
                self._phase("failed",
                            f"{port} does not exist - is the STM32 plugged in?")
                return
            if s is not None and not s.dry:
                self._phase("stopping", "stopping the car, then ending the session")
                if not s.stop_car_and_wait():
                    runlog.emit("[reboot] motor-off not confirmed - rebooting "
                                "anyway (REBOOT cuts the motor first)")
                self._phase("sending", f"sending REBOOT on {port}")
                s.loop.reboot()
                time.sleep(0.3)             # let the held frames go out
                s.close("reboot")           # releases the port now
            else:
                if s is not None:
                    s.close("reboot")
                self._phase("sending", f"sending REBOOT on {port}")
                from control.link import send_command_burst
                from control.intent import ActionIntent
                send_command_burst(port, 115200, ActionIntent.stop("reboot"),
                                   frames=8)
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
                self._phase("failed", f"{port} never dropped: the firmware did not "
                                      f"take REBOOT (wrong port? older firmware?)")
                return
            self._phase("waiting", f"{port} dropped - waiting for it to come back")
            while time.time() - t0 < 12.0:
                nid = _node_id(port)
                if nid is not None and nid != before:
                    self._phase("done", f"STM32 back on {port} after "
                                        f"{time.time() - t0:.1f} s. It zeroes its "
                                        f"yaw now: keep the car still, then Start.")
                    return
                time.sleep(0.05)
            self._phase("failed",
                        f"{port} did not come back within 12 s - check USB and power")
        except Exception as e:
            self._phase("failed", f"reboot failed: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                if self.session is s:
                    self.session = None
                self.busy = None


run_mgr = RunManager()


# ==========================================================================
# drawing
# ==========================================================================

def _swapped(raw):
    return (np.ascontiguousarray(raw[:, :, ::-1]) if PI["SWAP_RB"] else raw)


def _detect_now(want_masks=False):
    """Run the real detector on the newest frame. Used by the Calibrate tab
    and the preview, so what you see is what the robot sees."""
    raw, seq = capture.get()
    if raw is None:
        return None, None, None
    frame = _swapped(raw)
    small = cv2.resize(frame, camera.PROC_SIZE, interpolation=cv2.INTER_AREA)
    det = camera.PillarDetector(PI).detect(small, want_masks=want_masks)
    return frame, small, det


def outlined(img, text, org, col, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1,
                cv2.LINE_AA)


def dashed_rect(img, x0, y0, x1, y1, col, thick=1, dash=6):
    for xa in range(x0, x1, dash * 2):
        cv2.line(img, (xa, y0), (min(xa + dash, x1), y0), col, thick)
        cv2.line(img, (xa, y1), (min(xa + dash, x1), y1), col, thick)
    for ya in range(y0, y1, dash * 2):
        cv2.line(img, (x0, ya), (x0, min(ya + dash, y1)), col, thick)
        cv2.line(img, (x1, ya), (x1, min(ya + dash, y1)), col, thick)


def _mask_tile(mask, title, w, h, bgr=None, colour=None):
    tile = np.zeros((h, w, 3), np.uint8)
    if mask is not None:
        small = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        tile[small > 0] = bgr or BOX_BGR.get(colour, (255, 255, 255))
        cover = 100.0 * float(np.count_nonzero(mask)) / mask.size
        text = f"{title}  {cover:.1f}%"
    else:
        text = f"{title}: off"
    cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    cv2.putText(tile, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (235, 235, 235), 1, cv2.LINE_AA)
    return tile


def render_run(frame, det, tick):
    """Detections (640x480) | RED, GREEN, floor masks stacked beside it."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    W, H = camera.FRAME_W, camera.FRAME_H
    cv2.line(bgr, (W // 2, 0), (W // 2, H), (200, 200, 200), 1)

    for box, colour, code, frac in det.rejected:
        x0, y0, x1, y1 = box
        dashed_rect(bgr, x0, y0, x1, y1, REJECT_BGR, 1)
        label = f"{colour[0]} {code}"
        if frac is not None:
            label += f" {frac:.0%}"
        outlined(bgr, label, (x0, max(14, y0 - 6)), REJECT_BGR, 0.45)

    chosen = tick.pillar_xy if tick is not None else None
    second = tick.sec_xy if tick is not None else None
    for i, p in enumerate(det.pillars):
        col = BOX_BGR.get(p.colour, (255, 255, 255))
        x0, y0, x1, y1 = p.box
        target = (i == 0 and chosen is not None)
        # The second-largest blob is what decides which side the next corner
        # comes out on, so it is worth seeing on the overlay: if the corner
        # goes the wrong way, this box is where to look first.
        runner = (i == 1 and second is not None)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), col, 4 if target else 2)
        label = f"{p.colour[0]} {p.bearing_deg:+.1f}deg"
        if target:
            label = f"TRACKED {label} {chosen[0]:.0f},{chosen[1]:.0f}mm"
        elif runner:
            label = f"2ND {label} {second[0]:.0f},{second[1]:.0f}mm"
        outlined(bgr, label, (x0, max(16, y0 - 7)), col, 0.52)

    tiles = [_mask_tile(det.masks.get("RED"), "RED pillars", TILE_W, TILE_H,
                        colour="RED"),
             _mask_tile(det.masks.get("GREEN"), "GREEN pillars", TILE_W, TILE_H,
                        colour="GREEN"),
             _mask_tile(det.masks.get("FLOOR"), "white mat", TILE_W, TILE_H,
                        bgr=(235, 235, 235))]
    return np.hstack([bgr, np.vstack(tiles)])


def render_cal(mode):
    """The Calibrate tab's views, at detection resolution, upscaled."""
    frame, small, det = _detect_now(want_masks=True)
    if small is None:
        return None
    hsv, lab = camera.colour_spaces(small)

    if mode in ("red", "green"):
        m = det.masks.get(mode.upper())
        out = cv2.cvtColor(m if m is not None else np.zeros(small.shape[:2],
                                                            np.uint8),
                           cv2.COLOR_GRAY2BGR)
    elif mode == "mat":
        out = cv2.cvtColor(det.ctx.floor, cv2.COLOR_GRAY2BGR)
    elif mode == "linked":
        # blue = mat linked to the floor in front; purple = white but cut off
        base = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        out = (base * 0.4).astype(np.uint8)
        floor = det.ctx.floor > 0
        linked = det.ctx.linked_floor() > 0
        out[linked] = (0.45 * base[linked]
                       + 0.55 * np.array((235, 170, 60))).astype(np.uint8)
        cut = floor & ~linked
        out[cut] = (0.45 * base[cut]
                    + 0.55 * np.array((170, 60, 170))).astype(np.uint8)
    elif mode == "a":
        out = cv2.applyColorMap(lab[:, :, 1], cv2.COLORMAP_COOL)
    elif mode == "chroma":
        ch = np.clip(camera.chroma(lab) * 2, 0, 255).astype(np.uint8)
        out = cv2.applyColorMap(ch, cv2.COLORMAP_VIRIDIS)
    else:
        out = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    sx = camera.PROC_SIZE[0] / float(camera.FRAME_W)
    sy = camera.PROC_SIZE[1] / float(camera.FRAME_H)
    for box, colour, code, _ in det.rejected:
        x0, y0, x1, y1 = box
        cv2.rectangle(out, (int(x0 * sx), int(y0 * sy)),
                      (int(x1 * sx), int(y1 * sy)), (140, 140, 140), 1)
        cv2.putText(out, code, (int(x0 * sx), max(9, int(y0 * sy) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    for i, p in enumerate(det.pillars):
        x0, y0, x1, y1 = p.box
        cv2.rectangle(out, (int(x0 * sx), int(y0 * sy)),
                      (int(x1 * sx), int(y1 * sy)), BOX_BGR[p.colour], 2)
        tag = {0: "1st", 1: "2nd"}.get(i)
        if tag:
            cv2.putText(out, tag, (int(x0 * sx), max(9, int(y0 * sy) - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, BOX_BGR[p.colour], 1)

    for cls, x, y in cal.markers():
        col = {"mat": (255, 255, 255), "red": (60, 60, 255),
               "green": (60, 220, 60)}[cls]
        cv2.drawMarker(out, (x, y), col, cv2.MARKER_CROSS, 7, 1)
    return cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST)


def mjpeg(render_fn, fps=STREAM_FPS):
    period = 1.0 / fps
    while True:
        t0 = time.time()
        img = render_fn()
        if img is None:
            time.sleep(0.1)
            continue
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY),
                                int(PI["JPEG_QUALITY"])])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def _run_frame():
    s = run_mgr.session
    prod = vision.get() if vision else None
    if prod is None or prod.get("small") is None:
        frame, small, det = _detect_now(want_masks=True)
        if det is None:
            img = np.full((camera.FRAME_H, camera.FRAME_W + TILE_W, 3), 24,
                          np.uint8)
            cv2.putText(img, f"no camera frame ({capture.error})"
                        if capture.error else "waiting for camera frames...",
                        (24, camera.FRAME_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1, cv2.LINE_AA)
            return img
    else:
        det = prod["det"]
        frame = vision.frame()
        if frame is None:
            frame = _swapped(capture.get()[0])
    return render_run(frame, det, s.loop.last if s else None)


# ==========================================================================
# routes
# ==========================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/run/stream")
def run_stream():
    if vision is not None:
        vision.viewers += 1
    try:
        return Response(mjpeg(_run_frame),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    finally:
        pass


@app.route("/cal/stream")
def cal_stream():
    mode = request.args.get("mode", "camera")
    return Response(mjpeg(lambda: render_cal(mode)),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------- params

@app.route("/api/params")
def api_params():
    return jsonify({
        "pi": PI.describe(),
        "derived": {k: round(v, 1) for k, v in PI.derived.items()},
        "stm32": STM.describe(),
        "stm32_synced": STM.synced,
        "stm32_version": STM.version,
        "stm32_boot": STM.boot,
        "stm32_groups": prm.STM_GROUPS,
    })


@app.route("/api/param", methods=["POST"])
def api_param():
    """One parameter. Pi values apply on the next tick; STM32 values are
    queued and pushed by the loop, so this never blocks on the port."""
    body = request.get_json(force=True, silent=True) or {}
    side, name, value = body.get("side"), body.get("name"), body.get("value")
    try:
        if side == "pi":
            v = PI.set(name, value)
        elif side == "stm32":
            v = STM.set(name, value)
        else:
            return jsonify({"ok": False, "error": "unknown side"}), 400
    except KeyError:
        return jsonify({"ok": False, "error": "no such parameter"}), 404
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e) or "bad value"}), 400
    return jsonify({"ok": True, "value": v,
                    "derived": {k: round(x, 1) for k, x in PI.derived.items()}})


@app.route("/api/save", methods=["POST"])
def api_save():
    try:
        msg = prm.save(PI, STM)
    except OSError as e:
        return jsonify({"ok": False, "msg": f"could not save: {e}"}), 500
    runlog.emit(msg)
    return jsonify({"ok": True, "msg": msg})


@app.route("/api/revert", methods=["POST"])
def api_revert():
    side = (request.get_json(force=True, silent=True) or {}).get("side")
    if side == "pi":
        PI.reset()
        msg = "[pi] Pi parameters back to defaults (not saved yet)"
    elif side == "stm32":
        STM.reset()
        msg = "[pi] STM32 parameters back to firmware defaults (not saved yet)"
    else:
        return jsonify({"ok": False, "msg": "unknown side"}), 400
    runlog.emit(msg)
    return jsonify({"ok": True, "msg": msg})


# ------------------------------------------------------------- calibrate

@app.route("/api/cal/sample", methods=["POST"])
def api_cal_sample():
    b = request.get_json(force=True, silent=True) or {}
    cls = b.get("cls")
    if cls not in CLASSES:
        return jsonify({"ok": False, "error": "unknown class"}), 400
    frame, small, det = _detect_now()
    if small is None:
        return jsonify({"ok": False, "error": "no camera frame yet"}), 503
    hsv, lab = camera.colour_spaces(small)
    s = cal.sample(cls, b.get("x", 0), b.get("y", 0), hsv, lab)
    out = {"ok": True, "s": s}
    # A click inside an accepted blob, with a measured distance typed in, is
    # what gives AREA_K.
    if cls in ("red", "green"):
        sx = camera.PROC_SIZE[0] / float(camera.FRAME_W)
        sy = camera.PROC_SIZE[1] / float(camera.FRAME_H)
        for p in det.pillars:
            x0, y0, x1, y1 = p.box
            if x0 * sx <= s["x"] < x1 * sx and y0 * sy <= s["y"] < y1 * sy:
                k = cal.area_k_from_area(p.area)
                if k:
                    PI.set("AREA_K", k)
                    out["area_k"] = {"area": p.area,
                                     "dist": int(cal.area_k["dist_mm"]),
                                     "k": int(k)}
                break
    return jsonify(out)


@app.route("/api/cal/undo", methods=["POST"])
def api_cal_undo():
    cal.undo((request.get_json(force=True, silent=True) or {}).get("cls"))
    return jsonify({"ok": True})


@app.route("/api/cal/clear", methods=["POST"])
def api_cal_clear():
    cal.clear((request.get_json(force=True, silent=True) or {}).get("cls"))
    return jsonify({"ok": True})


@app.route("/api/cal/fitparams", methods=["POST"])
def api_cal_fitparams():
    b = request.get_json(force=True, silent=True) or {}
    return jsonify({"ok": True, "fit": cal.set_fit_params(**b)})


@app.route("/api/cal/dist", methods=["POST"])
def api_cal_dist():
    b = request.get_json(force=True, silent=True) or {}
    return jsonify({"ok": True, "dist": cal.set_distance(b.get("d", 600))})


@app.route("/api/cal/fit", methods=["POST"])
def api_cal_fit():
    ok, msg, vals = cal.fit()
    if ok:
        PI.set_many(vals)
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/cal/state")
def api_cal_state():
    _, small, det = _detect_now()
    blobs = []
    if det is not None:
        for i, p in enumerate(det.pillars[:8]):
            # 1st is what the planner steers around; 2nd is what decides the
            # next corner's exit side. Everything after that is not used.
            rank = {0: " (1st)", 1: " (2nd)"}.get(i, "")
            blobs.append({"colour": p.colour, "area": p.area,
                          "verdict": "ACCEPTED" + rank})
        for box, colour, code, _ in det.rejected[:8]:
            w = max(1, box[2] - box[0])
            h = max(1, box[3] - box[1])
            blobs.append({"colour": colour, "area": w * h // 4,
                          "verdict": camera.REJECT_TEXT.get(code, code)})
    return jsonify({
        "classes": cal.stats(),
        "fit": cal.fit_params,
        "dist_mm": cal.area_k["dist_mm"],
        "space": det.space if det else None,
        "min_area": PI["MIN_AREA_PROC"],
        "blobs": blobs,
        "vals": [[n, ("on" if PI[n] else "off") if isinstance(PI[n], bool)
                  else round(float(PI[n]), 1)]
                 for n in FITTED_NAMES],
    })


# -------------------------------------------------------------- the run

def _req():
    try:
        return request.get_json(force=True, silent=True) or {}
    except Exception:
        return {}


def _reply(ok, err, code=409):
    return (jsonify({"ok": True}) if ok
            else (jsonify({"ok": False, "error": err}), code))


@app.route("/api/run/start", methods=["POST"])
def api_run_start():
    b = _req()
    cfg = prm.read_config().get("serial", {})
    port = str(b.get("port") or cfg.get("port", "/dev/ttyACM0")).strip()
    baud = int(cfg.get("baud", 115200))
    return _reply(*run_mgr.start(bool(b.get("dry")), port, baud))


@app.route("/api/run/end", methods=["POST"])
def api_run_end():
    return _reply(*run_mgr.end())


@app.route("/api/car/start", methods=["POST"])
def api_car_start():
    return _reply(*run_mgr.car_start())


@app.route("/api/car/stop", methods=["POST"])
def api_car_stop():
    return _reply(*run_mgr.car_stop())


@app.route("/api/run/reboot", methods=["POST"])
def api_run_reboot():
    cfg = prm.read_config().get("serial", {})
    port = str(_req().get("port") or cfg.get("port", "/dev/ttyACM0")).strip()
    return _reply(*run_mgr.start_reboot(port))


# ------------------------------------------------------------ the state

def _j(v, nd=None):
    if v is None:
        return None
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, nd) if nd is not None else v
    return v


@app.route("/api/state")
def api_state():
    try:
        since = int(request.args.get("since", 0))
    except (TypeError, ValueError):
        since = 0
    now = time.time()
    s = run_mgr.session
    view = s if s is not None else run_mgr.last
    tick = view.loop.last if view is not None else None

    if vision is not None:
        vision.viewers = max(vision.viewers, 1)

    # ---- lidar, live even with no session ----
    lid = {"ok": False, "ranges": [], "front_mm": None, "stats": None,
           "thread_alive": bool(lidar_thread and lidar_thread.is_alive())}
    wall = {"left": None, "right": None, "yaw": None}
    cands = 0
    if lidar_thread is not None and lidar_thread.is_alive():
        ranges = list(lidar_thread._ranges)
        valid = [d for d in ranges if not math.isinf(d)]
        f = sector_min(ranges, 0, 15)
        lid.update(ok=True,
                   ranges=[None if math.isinf(d) else round(d) for d in ranges],
                   front_mm=None if math.isinf(f) else round(f),
                   stats={"valid_count": len(valid), "total": len(ranges),
                          "min_mm": round(min(valid)) if valid else None,
                          "max_mm": round(max(valid)) if valid else None,
                          "rev": lidar_thread.rev,
                          "queue": lidar_thread.queue_depth()})
        if tick is not None and tick.wall is not None:
            w = tick.wall
            wall = {"left": _j(w.left_mm, 0), "right": _j(w.right_mm, 0),
                    "yaw": _j(w.yaw_deg, 1)}
            cands = tick.n_candidates
        else:
            w = cones(ranges, PI)
            wall = {"left": _j(w.left_mm, 0), "right": _j(w.right_mm, 0),
                    "yaw": _j(w.yaw_deg, 1)}
            cands = len(lidar_candidates(ranges, w, PI))

    cam_res, _ = shared.snapshot()
    out = {
        "now": now,
        "active": s is not None,
        "busy": run_mgr.busy,
        "reboot": dict(run_mgr.reboot),
        "camera_ok": capture.get()[0] is not None,
        "camera_error": capture.error,
        "vision_fps": round(vision.fps, 1) if vision else 0.0,
        "vision_error": vision.error if vision else None,
        "space": cam_res.space if cam_res else None,
        "n_pillars": len(cam_res.pillars) if cam_res else 0,
        "lidar": lid,
        "wall": wall,
        "candidates": cands,
        "radar_max_mm": RADAR_MAX_MM,
        "derived": {k: round(v, 1) for k, v in PI.derived.items()},
        "session": None,
        "tick": None,
    }

    if view is not None:
        out["session"] = {
            "active": s is not None,
            "dry": view.dry, "port": view.port,
            "started_at": view.started_at, "ended_at": view.ended_at,
            "ended_by": view.ended_by, "error": view.error,
            "elapsed_s": round((view.ended_at or now) - view.started_at, 1)
                         if view.started_at else None,
            "loop_alive": view.alive(),
            "link": view.link.stats() if view.link else None,
        }

    if tick is not None:
        t = tick
        out["tick"] = {
            "t": t.t, "age_s": round(now - t.t, 3),
            "lidar_live": t.lidar_live, "cam_live": t.cam_live,
            "front": _j(t.front, 0), "left": _j(t.left, 0),
            "right": _j(t.right, 0),
            "pillar_colour": t.pillar_colour,
            "pillar_xy": ([round(t.pillar_xy[0]), round(t.pillar_xy[1])]
                          if t.pillar_xy else None),
            "pillar_area": t.pillar_area,
            "sec_colour": t.sec_colour,
            "sec_xy": ([round(t.sec_xy[0]), round(t.sec_xy[1])]
                       if t.sec_xy else None),
            "unknown_xy": ([round(t.unknown_xy[0]), round(t.unknown_xy[1])]
                           if t.unknown_xy else None),
            "n_candidates": t.n_candidates,
            "sent": t.sent, "rev": t.rev, "rev_per_s": t.rev_per_s,
            "queue_depth": t.queue_depth, "tick_ms": round(t.tick_ms, 2),
            "telem_fresh": t.telem_fresh,
            "intent": {"mode": t.intent.mode.name,
                       "heading": _j(t.intent.target_heading_deg, 1),
                       "steer": _j(t.intent.steer_deg, 1),
                       "speed": t.intent.speed_pwm,
                       "reverse": t.intent.reverse,
                       "arc_lock": _j(t.intent.arc_lock, 2),
                       "reason": t.intent.reason,
                       "clamped": t.clamped} if t.intent else None,
            "telem": ({"heading": _j(t.telem.heading_deg, 1),
                       "yaw_rate": _j(t.telem.yaw_rate_dps, 1),
                       "odo": t.telem.odo_ticks,
                       "servo": _j(t.telem.servo_deg, 1),
                       "floor": t.telem.floor_name,
                       "enabled": t.telem.enabled,
                       "imu_ok": t.telem.imu_ok,
                       "arc_done": t.telem.arc_done,
                       "recovering": t.telem.recovering,
                       "recover_tries": t.telem.recover_tries,
                       "link_stale": t.telem.link_stale,
                       "boot_id": t.telem.boot_id,
                       "seq": t.telem.seq} if t.telem else None),
            "fsm": t.fsm,
        }

    lines, last_idx = runlog.since(since)
    out["log"] = {"last": last_idx,
                  "lines": [[i, round(ts, 3), text] for i, ts, text in lines]}
    return jsonify(out)


# ==========================================================================

def main():
    global lidar_thread, vision
    import signal
    # systemd stops robodash.service with SIGTERM. Python's default handler
    # would exit without running the finally below, leaving a car that is
    # mid-run driving with nobody sending STOP. Exit normally instead.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    sys.stdout = _StdoutTee(sys.stdout)

    print(prm.load(PI, STM))
    capture.start()
    vision = camera.VisionThread(PI, shared, frames_from=capture.get,
                                 name="DashVision")
    vision.start()
    try:
        lidar_thread = LidarThread(shared)
        lidar_thread.start()
    except Exception as e:
        print(f"[dashboard] lidar unavailable, camera tuning still works: {e}")

    print(f"Dashboard on http://0.0.0.0:{PORT}")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False,
                use_reloader=False)
    finally:
        if run_mgr.session is not None:
            try:
                run_mgr.end()          # stop the car, release the port
            except Exception as e:
                print(f"[dashboard] ending the session failed: {e}")
        if vision is not None:
            vision.stop()
        capture.stop()
        if lidar_thread is not None:
            lidar_thread.stop()
            lidar_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
