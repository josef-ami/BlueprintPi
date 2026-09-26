#!/usr/bin/env python3
"""
obstacleRound.py - Obstacle-challenge feeder (LiDAR + camera) for
ObstacleRound.cpp, with a web page that runs AND tunes the car.

The STM32 owns all driving; this process is a sensor pipe, a debug stream, the
start button, and now the tuning console for both halves.

Run it INSTEAD of openRound.py / dashboard.py / main.py - one process may hold
the lidar and the camera at a time. Put it in the repo root next to
openRound.py (it imports from there and from sensors/).

    python3 obstacleRound.py
    then open  http://<pi>:5000

PAGE
  Run        camera stream, START / STOP, state, STM32 log
  Tune       every tunable on the Pi and every tunable in the firmware,
             grouped, live. A change applies on the next frame (Pi) or within
             a few hundred ms (STM32). Save writes tuning.json; Revert puts
             the defaults back.
  Telemetry  what the planner is actually seeing: lane offset, cone fits,
             wall angle, the located pillar, the unknown LiDAR object.

TUNING - WHO OWNS WHAT
  The firmware has no storage. It boots with its compiled-in defaults and
  announces a boot id; this process notices a new id, re-reads the firmware's
  parameter table and pushes the whole saved set back. So the STM32 can reset
  mid-session and be back on your numbers within a few hundred ms, instead of
  quietly running defaults for the rest of the race.

  Pushes are rate-limited (STM_PUSH_PER_LOOP) and drained by the same loop that
  writes sensor frames, so tuning can never stall the 50 Hz feed. Nothing in
  this file blocks on the serial port.

Wire frame - one ASCII line per send, SEND_HZ times a second, 20 fields:

    left,front,right,rev,color,err,area,vseq,coneL,coneR,wallAng,pX,pY,uX,uY,sColor,sX,sY,mX,mY\n

  left/front/right  mm at 90 / 0 / 270 deg, 65535 = no return   (as openRound)
  rev               lidar revolution counter                     (as openRound)
  color             first sign: 1 = red, 0 = green, 2 = none
  err               first sign centre x - 320, 640-px frame, + = right (debug;
                    the rear ToF is on the STM32 now, mux CH4 - not in the frame)
  area              first sign blob area, 320x240 detection pixels (debug)
  vseq              camera frame counter (debug)
  coneL / coneR     perpendicular mm to the left / right wall, line-fitted
                    over a 45 deg LiDAR cone each side; 65535 = no fit
  wallAng           car yaw to the walls x10 (deci-deg), + = pointing left;
                    32767 = no fit
  pX / pY           first sign centre, mm from the LiDAR (x fwd, y left):
                    camera bearing + LiDAR range on that ray; 32767 = none
  uX / uY           nearest LiDAR object in the corridor the camera has NOT
                    coloured (LazyGo edge detector); 32767 = none
  sColor / sX / sY  always 2,32767,32767 - there is ONE sign per frame, the
                    largest blob. The three fields stay so the frame layout
                    (and older firmware) is unchanged.
  mX / mY           firmware v11 park-in: the nearest MAGENTA parking-lot block
                    (the model's magenta / parking class), mm from the LiDAR
                    (x fwd, y left): camera bearing + the nearest LiDAR return
                    on that ray, its face toward the car; 32767 = none. Never
                    a sign - firmware v10 and older stop reading at field 18.

Commands on the same serial line:
    S            START            X            STOP
    (firmware v10: START first drives the car out of the parking lot -
     PARK OUT, pivot / exit / reverse arc - then the 3 laps; v11 then parks
     in the lot - PARK IN.)
    N <name> <v> set a firmware parameter       ?P  dump the table
    ?V           firmware version / boot id
    (firmware v11 has its parameter table OFF: every value is set in the
     .ino and these lines are ignored - the Tune tab changes nothing there.)
Replies from the firmware start with '!' (parameters) or '#' (log).

Silence rules: lidar stale -> no frames are sent (commands still are).
Camera stale (> VISION_STALE_S) -> color is sent as 2 (none).

PILLAR DETECTION: YOLO FIRST, COLOUR MASKING AS THE FALLBACK
  The camera thread finds pillars with the trained YOLO26n model in
  models/pillars26/ (sensors/yolo_detector.py): 2 classes, GREEN PILLAR and
  RED PILLAR, trained on frames from this car's own camera at 416 px. It runs
  on ncnn, OpenVINO or onnxruntime - whichever is installed (YOLO_BACKEND 0 =
  try them in that order), no torch needed. Tune tab, "YOLO detector":
  USE_YOLO, backend, threads, confidence, NMS overlap, minimum box height.
  If no backend can load the model, or USE_YOLO is off, the Lab/HSV masking
  below does the job exactly as before. Everything after detection - bearing,
  LiDAR range, the 18-field frame - is the same for both, so the STM32 cannot
  tell which one found the pillar. The model was trained on RGB frames taken
  with SWAP_RB on: keep it on, or red and blue swap and the model sees nonsense.
  The Run tab shows which detector is live and its time per frame.

RECORDING (training images for a YOLO detector) - see tools/yolo/PROTOCOL.md
  With RECORD_RUNS on (Tune tab, "Recording"; OFF by default so a competition
  run never fills the SD card), every run is saved to take/ next to this file,
  one folder per run:

      take/run_20260923_141502/run_20260923_141502_00000.jpg ...
      take/run_20260923_141502/frames.csv     time, state, corner, lidar front, colour

  Recording starts when the STM32 logs "# GO" and stops at "# FINISHED" or a
  STOP. The Record button on the Run tab saves frames the same way without a
  run (push the car around the mat by hand - the fastest way to get every
  angle). The frames are the RAW camera image, exactly what the detector sees.
  A frame that is nearly identical to the last saved one (car standing still)
  is skipped (RECORD_MIN_CHANGE). Saving runs on its own thread and drops
  frames rather than ever slowing the camera. The PC pulls them with
  `python tools/yolo/yolo.py pull`.

WHICH MODEL (models/ACTIVE)
  models/ACTIVE holds the name of the folder under models/ to load (default
  pillars_20260925_1000_416: pillars + parking lot; pillars26 is the rollback). `tools/yolo/yolo.py deploy` copies a new model to the Pi and
  rewrites ACTIVE; the vision thread notices within 2 s and reloads - no
  restart. Classes are matched by NAME: any class with RED / GREEN in its name
  is a sign; anything else (PARKING LOT) is drawn on the page, never steered by.

A NOTE ON THE CAMERA MODEL
  config.json disagrees with itself: camera_intrinsics.K has fx = 383 px/rad,
  which on a 640 px frame is +/-48 deg, while hfov_deg says 160 (+/-80), a
  factor of 1.67. The calibrated distortion polynomial also stops being
  monotonic at 62 deg, so it cannot describe anything further out. USE_INTRINSICS
  picks which model is used for bearings; measure a pillar at a known angle
  before trusting either, because the answer decides whether the car can see a
  pillar just past a corner at all.
"""

import collections
import csv
import math
import os
import queue
import shutil
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

import serial

import sensors.camera as camera
from worldstate import SharedState
from sensors.lidar import LidarThread
from openRound import UART_PORT, UART_BAUD, load_tol, read_three, lidar_live, _u16

import params as prm
from sensors.yolo_detector import YoloDetector, DEFAULT_MODEL_DIR

MODELS_DIR = os.path.dirname(DEFAULT_MODEL_DIR)
ACTIVE_FILE = os.path.join(MODELS_DIR, "ACTIVE")


def active_model_dir():
    """models/<name> named in models/ACTIVE, else the default (sensors/yolo_detector.py)."""
    try:
        with open(ACTIVE_FILE) as f:
            name = f.read().strip()
        d = os.path.join(MODELS_DIR, name)
        if name and os.path.isdir(d):
            return d
    except OSError:
        pass
    return DEFAULT_MODEL_DIR

# ---------------- fixed, not tunable ----------------
PROC_SIZE   = (320, 240)     # detection resolution
FLASK_PORT  = 5000
LOG_LINES   = 200            # STM32 log lines kept for the page
# ----------------------------------------------------

COL_RED, COL_GREEN, COL_NONE = 1, 0, 2
CODE = {"RED": COL_RED, "GREEN": COL_GREEN}
NAMES = {COL_RED: "RED", COL_GREEN: "GREEN", COL_NONE: "none"}
BOX_BGR = {COL_RED: (0, 0, 255), COL_GREEN: (0, 255, 0)}

CMD_START = b"S\n"
CMD_STOP = b"X\n"
CMD_NAME = {CMD_START: "START", CMD_STOP: "STOP"}

CONE_NONE = 65535
ANG_NONE = 32767
PXY_NONE = 32767

# --------------------------------------------------------------------------
# tunables
# --------------------------------------------------------------------------
#
# The values live in PI (params.py). They are MIRRORED into module globals of
# the same name by sync_globals(), because the per-frame code paths below read
# them dozens of times per frame and a bare global is the cheapest read there
# is. Everything that changes a parameter calls sync_globals() straight after,
# so the mirror is never stale by more than one call.

PI = prm.PiParams(prm.PI_SPECS)
STM = prm.StmParams()

_MIRRORED = [s.name for s in prm.PI_SPECS
             if s.group in ("camera", "cone", "locate", "cand", "link", "lab", "yolo", "record")]

# defaults, so the names exist before the first sync
MIN_AREA_PROC = 250
VISION_STALE_S = 0.2
JPEG_QUALITY = 50
CAMERA_FWD_MM = 0.0
CAMERA_OFFSET_DEG = 5.0
HFOV_DEG = 160.0
USE_INTRINSICS = True
SWAP_RB = True
CONE_DEG = 45
CONE_MAX_RANGE_MM = 1500
CONE_MIN_RANGE_MM = 60
CONE_INLIER_MM = 25
CONE_MIN_INLIERS = 8
CONE_MIN_SPAN_MM = 150
CONE_AGREE_DEG = 6.0
RAY_WINDOW_DEG = 8.0
PILLAR_MAX_MM = 2000.0
AREA_K = 14000.0
FACE_TO_CENTRE_MM = 25.0
CORRIDOR_MM = 1000.0
CAND_MAX_MM = 1800.0
CAND_WALL_MM = 70.0
CAND_EDGE_MM = 150.0
CAND_MIN_WIDTH_MM = 25.0
CAND_MAX_WIDTH_MM = 120.0
CAND_FOV_DEG = 90
SEND_HZ = 50
CMD_REPEAT = 3
BEARING_TOL_DEG = 2
STM_PUSH_PER_LOOP = 4

USE_YOLO = True
YOLO_BACKEND = 0
YOLO_THREADS = 3
YOLO_CONF = 0.5
YOLO_IOU = 0.5
YOLO_MIN_BOX_H = 0
LOT_SUPPRESS_IOU = 0.3
YOLO_BACKENDS = ("auto", "ncnn", "openvino", "onnx")      # YOLO_BACKEND index

RECORD_RUNS = False
RECORD_HZ = 3.0
RECORD_MIN_CHANGE = 3.0
RECORD_JPEG_Q = 92
RECORD_MIN_FREE_MB = 500

USE_LAB = False
FLOOR_L_MIN = 120
FLOOR_AB_TOL = 14
LAB_CHROMA_MIN = 20

HSV_CFG = PI.hsv_config()
LAB_CFG = PI.lab_config()
PILLAR_FILTER = PI.pillar_filter()


def sync_globals():
    """Copy PI into the module globals the per-frame code reads."""
    g = globals()
    snap = PI.snapshot()
    for name in _MIRRORED:
        g[name] = snap[name]
    g["HSV_CFG"] = PI.hsv_config()
    g["LAB_CFG"] = PI.lab_config()
    g["PILLAR_FILTER"] = PI.pillar_filter()


sync_globals()

lock = threading.Lock()
vision = {"color": COL_NONE, "err": 0, "area": 0, "seq": 0, "t": 0.0,
          "box": None, "bearing": None,          # box in 640x480 px, for the overlay
          "m_bearing": None, "m_area": 0,        # the largest MAGENTA (parking lot) box
          "mode": "starting", "infer_ms": 0.0, "fps": 0.0, "yolo_error": None}
latest_frame = None                              # RGB, only kept while someone watches
viewers = 0

link = {"line": "", "t": 0.0, "live": False, "serial_ok": False,
        "f": None, "l": None, "r": None, "cl": None, "cr": None,
        "yaw": None, "pxy": None, "uxy": None, "hz": 0.0}
stm_log = collections.deque(maxlen=LOG_LINES)
stm_state = {"state": "waiting for STM32", "corner": "", "exit": "", "lane": ""}
commands = queue.Queue()                          # written to serial by the main loop only


def note(msg):
    print(msg)
    stm_log.append(msg)


# --------------------------------------------------------------------------
# recorder - raw camera frames to take/<session>/ for training a detector
# --------------------------------------------------------------------------
#
# The vision thread calls offer() once per camera frame. offer() only checks
# the rate and hands the frame REFERENCE to a small queue (grab_rgb() returns a
# fresh array every frame and nothing writes to it afterwards, so no copy is
# needed). This thread does the JPEG encode and the SD-card write. If the card
# is slow the queue fills and frames are DROPPED and counted - the camera, and
# with it the 50 Hz feed to the STM32, never waits on the disk.
#
# A session starts on "# GO" from the STM32 (or the Record button) and ends on
# "# FINISHED" / "# STOP from Pi" (or the button again). Each session is its
# own folder, and every file name carries the folder name, so images from
# many runs can be merged into one training set without name clashes.

TAKE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "take")


class FrameRecorder(threading.Thread):

    def __init__(self):
        super().__init__(name="Recorder", daemon=True)
        self._q = queue.Queue(maxsize=8)
        self._halt = threading.Event()   # NOT _stop: that name is taken by Thread
        self._lock = threading.Lock()
        self.run_active = False          # between "# GO" and "# FINISHED"/"# STOP"
        self.manual = False              # the page's Record button
        self.session = 0                 # bumped on every start; the writer
        self._kind = "run"               #   opens a new folder when it changes
        self.folder = ""                 # current folder name, for the page
        self.saved = 0                   # frames written in the current session
        self.dropped = 0                 # frames lost to a full queue (slow SD)
        self.skipped = 0                 # near-duplicates not saved (car standing still)
        self.error = None
        self._last_offer = 0.0

    # ---- control, from the serial loop and the web page ----

    def _begin(self, kind):
        with self._lock:
            self.session += 1
            self._kind = kind
            self.error = None
            self.folder, self.saved = "", 0   # the writer fills these on the first frame
            self._last_offer = 0.0

    def start_run(self):
        """'# GO' seen. Honours RECORD_RUNS; the button keeps its own session."""
        if not RECORD_RUNS or self.manual:
            return
        self._begin("run")
        self.run_active = True

    def _stopped_note(self):
        note(f"[pi] recording stopped: {self.saved} frames in take/{self.folder}"
             if self.folder else "[pi] recording stopped: no frames saved")

    def end_run(self):
        if self.run_active:
            self.run_active = False
            self._stopped_note()

    def set_manual(self, on):
        on = bool(on)
        if on and not self.manual:
            self.run_active = False          # the button's session replaces a run's
            self._begin("manual")
            self.manual = True
            note("[pi] recording (Record button)")
        elif not on and self.manual:
            self.manual = False
            self._stopped_note()

    def recording(self):
        return self.run_active or self.manual

    def status(self):
        return {"on": self.recording(), "manual": self.manual, "folder": self.folder,
                "saved": self.saved, "dropped": self.dropped, "skipped": self.skipped,
                "error": self.error}

    # ---- from the vision thread, once per camera frame ----

    def offer(self, frame):
        if not self.recording():
            return
        now = time.monotonic()
        if now - self._last_offer < 1.0 / max(0.1, float(RECORD_HZ)):
            return
        self._last_offer = now
        meta = (time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}",
                stm_state["state"], stm_state["corner"],
                "" if link["f"] is None else int(link["f"]),
                NAMES.get(vision.get("color", COL_NONE), "none"))
        try:
            self._q.put_nowait((self.session, self._kind, frame, meta))
        except queue.Full:
            self.dropped += 1

    # ---- the writer ----

    def stop(self):
        self._halt.set()

    def run(self):
        cur, path, fh, writer = None, None, None, None
        last_thumb = None
        try:
            while not self._halt.is_set():
                try:
                    sess, kind, frame, meta = self._q.get(timeout=0.5)
                except queue.Empty:
                    if fh is not None and not self.recording():
                        fh.close()
                        fh = writer = None
                    continue
                if sess != self.session:
                    continue                             # left over from an ended session
                try:
                    if sess != cur or fh is None:
                        if fh is not None:
                            fh.close()
                        name = f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}"
                        if os.path.exists(os.path.join(TAKE_DIR, name)):
                            name += f"_{sess}"           # two sessions in one second
                        path = os.path.join(TAKE_DIR, name)
                        os.makedirs(path, exist_ok=True)
                        fh = open(os.path.join(path, "frames.csv"), "a", newline="")
                        writer = csv.writer(fh)
                        if fh.tell() == 0:
                            writer.writerow(["file", "time", "state", "corner",
                                             "front_mm", "colour_seen"])
                        cur = sess
                        last_thumb = None
                        self.folder, self.saved, self.skipped = name, 0, 0
                        note(f"[pi] recording to take/{name}")

                    free_mb = shutil.disk_usage(path).free / 1e6
                    if free_mb < RECORD_MIN_FREE_MB:
                        if self.error is None:
                            self.error = f"SD card nearly full ({free_mb:.0f} MB free)"
                            note(f"[pi] recording paused: {self.error}")
                        continue

                    # a frame almost the same as the last one saved adds nothing
                    # to a training set (car parked, waiting at the start)
                    thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (32, 24),
                                       interpolation=cv2.INTER_AREA).astype(np.int16)
                    if (last_thumb is not None and RECORD_MIN_CHANGE > 0 and
                            float(np.abs(thumb - last_thumb).mean()) < RECORD_MIN_CHANGE):
                        self.skipped += 1
                        continue
                    last_thumb = thumb

                    fname = f"{self.folder}_{self.saved:05d}.jpg"
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(os.path.join(path, fname), bgr,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), int(RECORD_JPEG_Q)]):
                        raise OSError(f"could not write {fname}")
                    writer.writerow([fname, *meta])
                    self.saved += 1
                    if self.saved % 25 == 0:
                        fh.flush()                       # a power cut loses < 25 rows
                except OSError as e:
                    if self.error is None:
                        note(f"[pi] recording error: {e}")
                    self.error = str(e)
        finally:
            if fh is not None:
                fh.close()


RECORDER = FrameRecorder()


# --------------------------------------------------------------------------
# bearing: pixel -> direction
# --------------------------------------------------------------------------
#
# Two models, and which one is right is a MEASUREMENT, not a preference:
#
#   USE_INTRINSICS on   the calibrated fisheye K/D. fx = 383 px/rad puts the
#                       frame edge at 48 deg and the polynomial folds back at
#                       62, so nothing beyond that can be described at all.
#   USE_INTRINSICS off  ideal equidistant lens over HFOV_DEG: r = fx * theta
#                       with fx = (W/2) / radians(HFOV/2). Covers the full
#                       frame, assumes no distortion beyond equidistance.
#
# Sign convention throughout: + = LEFT of forward, matching the LiDAR frame
# (index = degree, increasing anticlockwise, y positive to the left).

def _principal_point():
    intr = camera.load_intrinsics()
    if intr is not None:
        K = intr["K"]
        return float(K[0, 2]), float(K[1, 2])
    return camera.FRAME_W / 2.0, camera.FRAME_H / 2.0


def bearing_from_px(cx, cy):
    """Blob centre in 640x480 pixels -> bearing in degrees, + = left."""
    if USE_INTRINSICS:
        return camera.px_to_bearing(cx, camera.FRAME_W, HFOV_DEG, cy=cy,
                                    offset_deg=CAMERA_OFFSET_DEG)
    ppx, ppy = _principal_point()
    fx = (camera.FRAME_W / 2.0) / math.radians(HFOV_DEG / 2.0)
    dx, dy = cx - ppx, cy - ppy
    r = math.hypot(dx, dy)
    if r < 1e-9:
        return CAMERA_OFFSET_DEG
    th = r / fx
    x = math.sin(th) * dx / r
    z = math.cos(th)
    return -math.degrees(math.atan2(x, z)) + CAMERA_OFFSET_DEG


# --------------------------------------------------------------------------
# camera thread
# --------------------------------------------------------------------------

# ---- pillar isolation by floor / pillar contrast -------------------------
#
# A real pillar is an upright, solid, saturated block STANDING ON the white
# mat. Everything else the colour mask picks up fails at least one of these:
#
#   A  aspect     h >= aspect_min * w    orange/red floor lines are flat
#   S  solidity   area >= solidity_min * bbox   thin streaks, ragged noise
#   F  on floor   the strip just under the blob is mostly white mat
#                 (low saturation, high value)  red/green things beyond the
#                 walls sit on black wall, not on mat
#   C  contrast   blob saturation - floor saturation >= contrast_s_min
#                 washed-out reflections / shadows on the mat
#
# A blob whose bottom touches the image bottom is a pillar too close to see
# its base; F and C are skipped for it (A and S still apply).
#
# Every threshold is tunable from the Tune tab ("filter" group).

# TWO COLOUR SPACES
#
# HSV was the original. It separates the pillars on hue, but it gates on
# SATURATION, and a matte pillar under dim indoor light falls under the S
# threshold and simply vanishes - which is the usual reason green stops being
# detected while red still works (red has two hue bands and survives longer).
#
# Lab does the same job without that failure mode. In OpenCV's 8-bit Lab, a
# and b are centred on 128: a > 128 is red, a < 128 is green, and the white mat
# sits near (128, 128) whatever the lighting does to L. So the pillars separate
# on ONE channel, and brightness never removes the colour.
#
# Both are kept and USE_LAB picks between them, so a bad calibration is one
# checkbox away from the behaviour you had before. calibrate_vision.py fits the
# Lab ranges by clicking and turns USE_LAB on when you save.


def colour_spaces(rgb):
    """(hsv, lab) for one detection-sized frame. Both are cheap; the overlay
    and the calibrator want whichever one you are not classifying in."""
    return (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))


def chroma(lab):
    """Distance from neutral grey in the a/b plane - the Lab analogue of HSV
    saturation, and what the contrast test compares."""
    ab = lab[:, :, 1:].astype(np.int16) - 128
    return np.hypot(ab[:, :, 0], ab[:, :, 1]).astype(np.float32)


def floor_mask(hsv, lab, pf):
    """White mat. HSV: unsaturated and bright. Lab: bright and near-neutral."""
    if USE_LAB:
        L, a, b = lab[:, :, 0], lab[:, :, 1].astype(np.int16), lab[:, :, 2].astype(np.int16)
        tol = FLOOR_AB_TOL
        m = ((L >= FLOOR_L_MIN) & (np.abs(a - 128) <= tol)
             & (np.abs(b - 128) <= tol)).astype(np.uint8) * 255
    else:
        m = cv2.inRange(hsv, (0, 0, pf["floor_v_min"]), (180, pf["floor_s_max"], 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def colour_mask(hsv, lab, name):
    """Binary mask for one pillar colour, in whichever space is selected."""
    if USE_LAB:
        lo, hi = LAB_CFG[name]
        m = cv2.inRange(lab, np.array(lo, np.uint8), np.array(hi, np.uint8))
        k = np.ones((5, 5), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return camera.build_mask(hsv, HSV_CFG[name])


def check_pillar(cnt, area, hsv, lab, floor, pf, bottom=None, ch=None):
    """None if the blob is a pillar, else a one-letter reject code (H/A/S/F/C).

    bottom = the lowest usable image row (the ROI bottom); ch = the frame's
    chroma image, computed once per frame by find_pillars()."""
    x, y, w, h = cv2.boundingRect(cnt)
    if h < pf["min_box_h_px"] * PROC_SIZE[1] / float(camera.FRAME_H):
        return "H"                                    # LazyGo: too short to be a sign
    if h < pf["aspect_min"] * w:
        return "A"
    if area < pf["solidity_min"] * w * h:
        return "S"

    H, W = floor.shape
    if bottom is None:
        bottom = H
    y0 = y + h
    if y0 >= bottom - pf["bottom_margin_px"]:
        return None                                   # base out of view: too close to check

    # centre 60 % of the width, so a wall edge next to the pillar can't count
    xa = x + int(0.2 * w)
    xb = max(xa + 1, x + int(0.8 * w))
    y1 = min(H, y0 + pf["strip_px"])
    strip = floor[y0:y1, xa:xb]
    if strip.size == 0 or strip.mean() / 255.0 < pf["floor_below_min"]:
        return "F"

    blob = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob, [cnt - (x, y)], -1, 255, -1)
    if USE_LAB:
        if ch is None:
            ch = chroma(lab)
        c_in = cv2.mean(ch[y:y + h, x:x + w], mask=blob)[0]
        c_floor = cv2.mean(ch[y0:y1, xa:xb], mask=strip)[0]
        if c_in - c_floor < LAB_CHROMA_MIN:
            return "C"
    else:
        s_in = cv2.mean(hsv[y:y + h, x:x + w, 1], mask=blob)[0]
        s_floor = cv2.mean(hsv[y0:y1, xa:xb, 1], mask=strip)[0]
        if s_in - s_floor < pf["contrast_s_min"]:
            return "C"
    return None


# ---- LazyGo additions (lazybot/detection_cam.py + helper/util.py) --------
#
#   ROI         only rows roi_top_px .. roi_bottom_px (640x480 units) are
#               searched: above is the hall beyond the walls, below is the
#               car's own nose. LazyGo uses 60 .. 440.
#   mask blur   Gaussian blur on the colour mask, re-thresholded at 127:
#               rounds ragged edges and fills pinholes before contours.
#   min height  a box shorter than min_box_h_px (640x480 units) is not a sign
#               (reject code H). LazyGo uses 30 px on the robot, 12 in sim.
#   rank        "closest = tallest box": a sign's height in the image falls
#               off with distance and is not cut by a side occluder the way
#               its area is, so it orders signs better than area does.

def roi_rows(pf, h):
    """(top, bottom) rows of the search band at detection resolution h."""
    k = h / float(camera.FRAME_H)
    top = max(0, min(h - 1, int(pf["roi_top_px"] * k)))
    bot = max(top + 1, min(h, int(pf["roi_bottom_px"] * k)))
    return top, bot


def lazygo_mask(m, pf, top, bot):
    """ROI + optional blur/threshold, LazyGo style."""
    if top > 0:
        m[:top] = 0
    if bot < m.shape[0]:
        m[bot:] = 0
    k = int(pf["mask_blur_px"])
    if k > 0:
        k = k + 1 if k % 2 == 0 else k
        m = cv2.GaussianBlur(m, (k, k), 0)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def find_pillars(hsv, lab, pf):
    """(accepted, candidates, floor).

    accepted = every blob that passed, as (cnt, area, code), LARGEST AREA
    FIRST. Only the first one goes to the firmware.

    candidates = [(cnt, area, code, reject_or_None)] for the debug overlay."""
    floor = floor_mask(hsv, lab, pf)
    top, bot = roi_rows(pf, floor.shape[0])
    ch = chroma(lab) if USE_LAB else None             # once per frame, not per blob
    cands, accepted = [], []
    for name in ("RED", "GREEN"):
        m = lazygo_mask(colour_mask(hsv, lab, name), pf, top, bot)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < MIN_AREA_PROC:
                continue
            why = check_pillar(c, a, hsv, lab, floor, pf, bottom=bot, ch=ch)
            cands.append((c, a, CODE[name], why))
            if why is None:
                accepted.append((c, a, CODE[name]))
    accepted.sort(key=lambda t: -t[1])
    return accepted, cands, floor


def is_lot_class(name):
    """A model class that is the parking lot's magenta blocks."""
    n = name.upper()
    return any(k in n for k in ("MAGENTA", "PARK", "LOT", "PINK", "PURPLE"))


def on_lot_box(b, lot):
    """A sign box that overlaps a parking-lot box by LOT_SUPPRESS_IOU, or
    whose centre lies inside it."""
    ix = max(0.0, min(b[2], lot[2]) - max(b[0], lot[0]))
    iy = max(0.0, min(b[3], lot[3]) - max(b[1], lot[1]))
    inter = ix * iy
    union = (b[2] - b[0]) * (b[3] - b[1]) + (lot[2] - lot[0]) * (lot[3] - lot[1]) - inter
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    inside = lot[0] <= cx <= lot[2] and lot[1] <= cy <= lot[3]
    return inside or (union > 0 and inter / union >= LOT_SUPPRESS_IOU)


def yolo_colour(name):
    """Model class name -> wire colour code; None for a class that is not a
    sign (e.g. a later 'magenta' parking class): shown, never steered by."""
    n = name.upper()
    if "RED" in n:
        return COL_RED
    if "GREEN" in n:
        return COL_GREEN
    return None


class VisionThread(threading.Thread):
    """Reads the tunables fresh every frame, so the Tune tab is live: there is
    no restart, no re-open of the camera, and the thread never blocks on the
    web side. Changing USE_YOLO / YOLO_BACKEND / YOLO_THREADS reloads the
    model on the next frame."""

    def __init__(self):
        super().__init__(name="Vision", daemon=True)
        self._halt = threading.Event()   # NOT _stop: that name is taken by Thread
        camera.load_intrinsics(camera.load_config())
        self.error = None
        self.yolo = None
        self.yolo_error = None
        self._yolo_key = None
        self._model_dir = active_model_dir()
        self._model_check = 0.0
        self._lot = None                  # largest parking-lot box this frame: (area, box)

    def stop(self):
        self._halt.set()

    def _detector(self):
        """The YOLO detector for the current settings, or None = colour masking.
        A failed load is not retried until one of the settings changes."""
        if not USE_YOLO:
            self._yolo_key, self.yolo = None, None
            return None
        now = time.monotonic()
        if now - self._model_check > 2.0:                 # models/ACTIVE changed?
            self._model_check = now
            self._model_dir = active_model_dir()
        key = (YOLO_BACKENDS[max(0, min(3, int(YOLO_BACKEND)))], int(YOLO_THREADS),
               self._model_dir)
        if key != self._yolo_key:
            self._yolo_key, self.yolo = key, None
            try:
                self.yolo = YoloDetector(key[2], backend=key[0], threads=key[1])
                self.yolo_error = None
                note(f"[vision] YOLO {os.path.basename(key[2])} on {self.yolo.backend} "
                     f"({key[1]} threads, {self.yolo.imgsz} px): "
                     f"{', '.join(self.yolo.names.values())}")
            except Exception as e:
                self.yolo_error = str(e)
                note(f"[vision] YOLO unavailable, using colour masking: {e}")
        return self.yolo

    def run(self):
        global latest_frame
        try:
            cam = camera.open_camera(camera.FRAME_W, camera.FRAME_H)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"[vision] camera failed: {self.error} - sending color=2")
            return

        sx = camera.FRAME_W / float(PROC_SIZE[0])
        sy = camera.FRAME_H / float(PROC_SIZE[1])
        centre = camera.FRAME_W // 2
        seq = 0
        fps, t_prev = 0.0, time.monotonic()
        try:
            while not self._halt.is_set():
                pf = PILLAR_FILTER                        # one read, one frame
                frame = camera.grab_rgb(cam, SWAP_RB)
                RECORDER.offer(frame)                     # raw frame, no overlay

                # hits: (colour, area in 320x240 px, box in 640x480 px, confidence)
                hits, cands, floor, mode = self._find(frame, pf, sx, sy)

                def describe(hit):
                    colour, area, (x0, y0, x1, y1), _ = hit
                    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                    return (colour, int(cx) - centre, int(area), (x0, y0, x1, y1),
                            bearing_from_px(cx, cy))

                m_bearing, m_area = None, 0
                if self._lot is not None:                 # the parking lot (never a sign)
                    m_area, (x0, y0, x1, y1) = self._lot
                    m_bearing = bearing_from_px((x0 + x1) / 2.0, (y0 + y1) / 2.0)

                seq = (seq + 1) & 0xFFFFFFFF
                if not hits:
                    color, err, area, box, bearing = COL_NONE, 0, 0, None, None
                else:                                     # the largest blob, only
                    color, err, area, box, bearing = describe(hits[0])

                now = time.monotonic()
                dt, t_prev = now - t_prev, now
                if dt > 0:
                    fps += 0.1 * (1.0 / dt - fps)
                with lock:
                    vision.update(color=color, err=err, area=int(area),
                                  seq=seq, t=now, box=box, bearing=bearing,
                                  m_bearing=m_bearing, m_area=int(m_area),
                                  mode=mode, fps=fps,
                                  infer_ms=self.yolo.infer_ms if mode.startswith("YOLO") else 0.0,
                                  yolo_error=self.yolo_error)
                    if viewers > 0:
                        latest_frame = frame
                        vision["cands"] = cands
                        vision["dets"] = [(h[2], h[0], h[3]) for h in hits]
                        vision["floor"] = floor
                    else:
                        latest_frame = None
        finally:
            try:
                cam.stop()
            except Exception:
                pass
            print("[vision] stopped")

    def _find(self, frame, pf, sx, sy):
        """(hits, cands, floor, mode). hits largest first; cands = every box for
        the overlay as (box, colour, reject_code_or_None)."""
        det = self._detector()
        self._lot = None
        if det is not None:
            try:
                found = det.detect(frame, conf=YOLO_CONF, iou=YOLO_IOU)
            except Exception as e:                  # never lose the camera over it
                if self.yolo_error is None:
                    note(f"[vision] YOLO failed, colour masking this frame: {e}")
                self.yolo_error = str(e)
                found = None
            if found is not None:
                hits, cands, lot_boxes = [], [], []
                for d in found:
                    box, code = d.box(), yolo_colour(d.name)
                    if code is None:                         # not a sign class:
                        tag = "LOT" if "PARK" in d.name.upper() else "M"   # shown, never steered by
                        cands.append((box, code, f"{tag} {d.conf:.2f}"))
                        if is_lot_class(d.name):             # the parking lot -> mX/mY
                            lot_boxes.append(box)
                            a = d.w * d.h / (sx * sy)
                            if self._lot is None or a > self._lot[0]:
                                self._lot = (a, box)
                    elif d.h < YOLO_MIN_BOX_H:
                        cands.append((box, code, "H"))
                    else:
                        # box area in 320x240 px, the unit AREA_K and the
                        # locate_pillar() sanity window were written for
                        hits.append((code, d.w * d.h / (sx * sy), box, d.conf))
                # a RED/GREEN box on top of a lot box is the lot block read twice
                # (the model sometimes calls a magenta block RED as well): drop it
                if LOT_SUPPRESS_IOU > 0 and lot_boxes:
                    keep = []
                    for h in hits:
                        if any(on_lot_box(h[2], lb) for lb in lot_boxes):
                            cands.append((h[2], h[0], "LOT?"))
                        else:
                            keep.append(h)
                    hits = keep
                hits.sort(key=lambda t: -t[1])              # largest blob first
                return hits, cands, None, f"YOLO {det.backend} {os.path.basename(det.model_dir)}"

        small = cv2.resize(frame, PROC_SIZE, interpolation=cv2.INTER_AREA)
        hsv, lab = colour_spaces(small)
        accepted, found_c, floor = find_pillars(hsv, lab, pf)

        def box640(cnt):
            x, y, w, h = cv2.boundingRect(cnt)
            return (int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy))

        hits = [(code, area, box640(c), 1.0) for (c, area, code) in accepted]
        cands = [(box640(c), code, why) for (c, _, code, why) in found_c]
        return hits, cands, floor, "Lab" if USE_LAB else "HSV"


def vision_now(now):
    with lock:
        v = dict(vision)
    if now - v["t"] > VISION_STALE_S:
        v["color"], v["err"], v["area"], v["bearing"] = COL_NONE, 0, 0, None
        v["m_bearing"], v["m_area"] = None, 0
    return v


# --------------------------------------------------------------------------
# LiDAR 45 deg cones -> perpendicular wall distance + car yaw to the walls
# --------------------------------------------------------------------------
#
# Instead of one beam at 90 / 270 deg, take EVERY return in a 45 deg cone
# centred on each side (67.5-112.5 deg left, 247.5-292.5 deg right), convert
# to x (forward) / y (left) and fit a straight line to the wall:
#
#   distance  = perpendicular distance to that line   (doesn't grow when yawed)
#   yaw       = -atan(slope), + = car pointing LEFT of the wall direction
#
# The fit is RANSAC over every point pair (vectorised, ~1k pairs): the line
# with the most points within CONE_INLIER_MM wins, ties go to the FARTHER
# line, so a pillar between the car and the wall is rejected as outliers
# instead of pulling the fit. Then a least-squares refit on the inliers.
# A fit needs CONE_MIN_INLIERS points spread over CONE_MIN_SPAN_MM along the
# wall - a 50 mm pillar face can't pass that on its own.

def fit_wall(ranges, centre_deg):
    """(perp_mm, yaw_deg, n_inliers, m, c) for the wall in the cone, or None."""
    half = int(CONE_DEG) // 2
    idx = np.arange(centre_deg - half, centre_deg + half + 1) % 360
    d = np.asarray([ranges[i] for i in idx], dtype=np.float64)
    ok = np.isfinite(d) & (d > CONE_MIN_RANGE_MM) & (d < CONE_MAX_RANGE_MM)
    if ok.sum() < CONE_MIN_INLIERS:
        return None
    b = np.radians(idx[ok])
    x, y = d[ok] * np.cos(b), d[ok] * np.sin(b)

    i, j = np.triu_indices(len(x), 1)
    keep = np.abs(x[j] - x[i]) > 40.0            # well-separated pairs only
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return None
    m = (y[j] - y[i]) / (x[j] - x[i])
    c = y[i] - m * x[i]
    res = np.abs(y[None, :] - m[:, None] * x[None, :] - c[:, None]) \
        / np.sqrt(1.0 + m[:, None] ** 2)
    inl = res < CONE_INLIER_MM
    cnt = inl.sum(axis=1)
    far = np.abs(c) / np.sqrt(1.0 + m * m)
    best = np.lexsort((far, cnt))[-1]            # most inliers, then farthest
    mask = inl[best]
    if mask.sum() < CONE_MIN_INLIERS:
        return None
    xs, ys = x[mask], y[mask]
    wall_d = float(np.median(np.abs(ys)))
    need = min(CONE_MIN_SPAN_MM,
               0.6 * 2 * wall_d * math.tan(math.radians(CONE_DEG / 2)))
    if xs.max() - xs.min() < need:
        return None
    m, c = np.polyfit(xs, ys, 1)
    return (abs(c) / math.sqrt(1.0 + m * m),
            -math.degrees(math.atan(m)),
            int(mask.sum()),
            float(m), float(c))                  # the line y = m x + c (LiDAR frame)


def cones_full(ranges):
    """cones() plus the two fitted wall lines (m, c) or None, for pillar search."""
    L = fit_wall(ranges, 90)
    R = fit_wall(ranges, 270)
    yaw = None
    if L and R:
        if abs(L[1] - R[1]) <= CONE_AGREE_DEG:
            yaw = (L[1] * L[2] + R[1] * R[2]) / (L[2] + R[2])
        # walls disagree (corner, inner-wall end): no yaw rather than a bad one
    elif L:
        yaw = L[1]
    elif R:
        yaw = R[1]
    return (L[0] if L else None, R[0] if R else None, yaw,
            (L[3], L[4]) if L else None, (R[3], R[4]) if R else None)


# --------------------------------------------------------------------------
# pillar position: camera bearing + LiDAR range along that ray
# --------------------------------------------------------------------------
#
# The camera says WHICH colour and in WHICH direction. The LiDAR says how far:
# we look for the nearest LiDAR return lying on the camera's ray. The camera
# sits CAMERA_FWD_MM ahead of the LiDAR, so the ray starts there, not at the
# LiDAR - at 40 cm that parallax is several degrees. Fallback: distance from
# blob area. Result is (x forward, y left) in mm from the LiDAR, pillar CENTRE.

_DEG = np.radians(np.arange(360))
_COS, _SIN = np.cos(_DEG), np.sin(_DEG)


def locate_pillar(ranges, bearing_deg, area, cam_fwd=None):
    """(x, y) mm of the pillar centre in the LiDAR frame, or None.

    Direction comes from the CAMERA (fresh, 30 Hz). Only the DISTANCE comes
    from the LiDAR, because a scan can be up to 100 ms old and while the car
    swerves at ~100 deg/s its bearings are ~10 deg stale - matching an exact
    ray then hits the wall behind the pillar. So: take every return within
    RAY_WINDOW_DEG of the camera ray (seen from the camera), keep those whose
    range agrees with the size estimate (AREA_K / sqrt(area)) within
    [0.5x, 2x], and use the nearest. No agreeing return -> size estimate.
    """
    if bearing_deg is None:
        return None
    if cam_fwd is None:
        cam_fwd = CAMERA_FWD_MM
    b = math.radians(bearing_deg)
    r = np.asarray(ranges, dtype=np.float64)
    ok = np.isfinite(r) & (r > 60) & (r < PILLAR_MAX_MM + 300)
    px, py = r * _COS - cam_fwd, r * _SIN            # relative to the camera
    dist_c = np.hypot(px, py)
    ang = np.degrees(np.arctan2(py, px)) - bearing_deg
    ang = (ang + 180.0) % 360.0 - 180.0
    d_area = AREA_K / math.sqrt(area) if area and area > 0 else None
    cand = ok & (np.abs(ang) <= RAY_WINDOW_DEG) & (dist_c > 30)
    if d_area is not None:
        cand &= (dist_c > 0.5 * d_area) & (dist_c < 2.0 * d_area)
    if cand.any():
        dist = float(dist_c[cand].min()) + FACE_TO_CENTRE_MM    # face -> centre
    elif d_area is not None:
        dist = d_area
    else:
        return None
    if dist > PILLAR_MAX_MM:
        return None
    return (cam_fwd + dist * math.cos(b), dist * math.sin(b))


def locate_block(ranges, bearing_deg, cam_fwd=None):
    """(x, y) mm of a parking-lot block's face in the LiDAR frame, or None.

    Like locate_pillar(), but a block is 200 x 20 mm and seen from any angle,
    so there is no size window: the nearest LiDAR return within RAY_WINDOW_DEG
    of the camera ray, 150..2500 mm from the camera. The firmware (v11 park-in)
    only uses it to know roughly where the lot is and which side it is on -
    the LiDAR side beam finds the blocks themselves.
    """
    if bearing_deg is None:
        return None
    if cam_fwd is None:
        cam_fwd = CAMERA_FWD_MM
    r = np.asarray(ranges, dtype=np.float64)
    ok = np.isfinite(r) & (r > 60)
    r = np.where(ok, r, 0.0)                     # no inf / nan in the maths below
    px, py = r * _COS - cam_fwd, r * _SIN
    dist_c = np.hypot(px, py)
    ang = np.degrees(np.arctan2(py, px)) - bearing_deg
    ang = (ang + 180.0) % 360.0 - 180.0
    cand = ok & (np.abs(ang) <= RAY_WINDOW_DEG) & (dist_c > 150) & (dist_c < 2500)
    if not cand.any():
        return None
    dist = float(dist_c[cand].min())
    b = math.radians(bearing_deg)
    return (cam_fwd + dist * math.cos(b), dist * math.sin(b))


# --------------------------------------------------------------------------
# LiDAR pillar candidates - objects inside the corridor, colour unknown
# --------------------------------------------------------------------------
#
# The camera only covers about +/-48 deg, so a sign near the far wall can stay
# out of view until it is too late. The LiDAR sees it from much further back.
#
# Detection is LazyGo's edge-stack detector (lazybot/control.py,
# detectContrast()), adapted to these 1-degree bins. Sweeping the scan from
# right to left, a sign is a VALLEY: the range drops sharply where it starts
# and rises sharply where it ends.
#   drop  > CAND_EDGE_MM   push the bin index on a stack   (object starts)
#   rise  > CAND_EDGE_MM   pop the last drop: the bins in between are one
#                          object; keep it if its width is CAND_MIN_WIDTH_MM ..
#                          CAND_MAX_WIDTH_MM (a sign is 50 mm, 71 diagonal)
# Matching drops to rises with a stack is bracket matching (a pushdown
# automaton), so a sign standing in front of a recess still closes correctly.
# A wall never has a deep step on BOTH sides, so it can never pass - unlike
# gap clustering, which could chop a far, oblique wall into sign-sized pieces.
#
# Survivors must then be ahead, closer than CAND_MAX_MM, and more than
# CAND_WALL_MM inside both fitted wall lines (a missing
# side is placed CORRIDOR_MM from the other): that removes signs of other
# straights seen across a corner. The nearest one that the camera has not
# already coloured is sent as uX/uY; the STM32 lines up with it so the camera
# can name it, and goes to the roomier side if it never does.

_FAR_MM = 4000.0          # an empty bin reads as "far background" for the edge test


def _filled_ranges(r, idx):
    """Ranges along idx with short dropouts (<= 2 bins) filled from the nearest
    valid neighbour, and longer ones treated as far background (LazyGo
    fix_missing(), simplified)."""
    seq = r[idx].copy()
    bad = ~np.isfinite(seq) | (seq <= 80)
    n = seq.size
    out = seq.copy()
    for i in np.nonzero(bad)[0]:
        best = None
        for off in (1, -1, 2, -2):
            j = i + off
            if 0 <= j < n and not bad[j]:
                best = seq[j]
                break
        out[i] = best if best is not None else _FAR_MM
    return out


def lidar_candidates(ranges, left_line, right_line):
    """[(x, y)] sign-centre candidates in the LiDAR frame, nearest first."""
    if left_line is None and right_line is None:
        return []                                    # corner: no corridor to search in
    r = np.asarray(ranges, dtype=np.float64)
    fov = int(CAND_FOV_DEG)
    idx = np.arange(-fov, fov + 1) % 360             # right (270..359) through left (..90)
    d = _filled_ranges(r, idx)
    x, y = d * _COS[idx], d * _SIN[idx]

    out, stack = [], []
    for i in range(1, d.size):
        step = d[i] - d[i - 1]
        if step < -CAND_EDGE_MM:                     # falling edge: object starts
            stack.append(i)
        elif step > CAND_EDGE_MM and stack:          # rising edge: object ends
            j = stack.pop()
            seg = slice(j, i)                        # bins j .. i-1
            if d[j:i].max() >= _FAR_MM:
                continue
            mid_r = float(np.median(d[seg]))
            width = math.hypot(x[i - 1] - x[j], y[i - 1] - y[j]) + mid_r * math.radians(1.0)
            if not (CAND_MIN_WIDTH_MM <= width <= CAND_MAX_WIDTH_MM):
                continue
            mx, my = float(x[seg].mean()), float(y[seg].mean())
            dist = math.hypot(mx, my)
            # no "short of the wall ahead" test: the depth step on both sides
            # already proves there is background behind it (and the old front-
            # beam test rejected a sign straight ahead, whose own face IS the
            # front beam)
            if mx <= 0 or dist > CAND_MAX_MM:
                continue
            if left_line is not None:
                if my > left_line[0] * mx + left_line[1] - CAND_WALL_MM: continue
            elif my > right_line[0] * mx + right_line[1] + CORRIDOR_MM - CAND_WALL_MM: continue
            if right_line is not None:
                if my < right_line[0] * mx + right_line[1] + CAND_WALL_MM: continue
            elif my < left_line[0] * mx + left_line[1] - CORRIDOR_MM + CAND_WALL_MM: continue
            out.append((mx + FACE_TO_CENTRE_MM * mx / dist,
                        my + FACE_TO_CENTRE_MM * my / dist))   # face -> centre
    out.sort(key=lambda p: math.hypot(*p))
    return out


def unclassified(cands, classified, match_mm=200.0):
    """Nearest candidate that is NOT a sign the camera already named.
    classified = the located signs (entries may be None)."""
    known = [p for p in classified if p is not None]
    for c in cands:
        if all(math.hypot(c[0] - k[0], c[1] - k[1]) > match_mm for k in known):
            return c
    return None


def _pxy(p):
    return (PXY_NONE, PXY_NONE) if p is None else (int(round(p[0])), int(round(p[1])))


def _cone_u16(mm):
    return CONE_NONE if mm is None else min(CONE_NONE - 1, int(round(mm)))


def _ang(yaw):
    return ANG_NONE if yaw is None else int(round(yaw * 10))


# --------------------------------------------------------------------------
# STM32 log -> state shown on the page
# --------------------------------------------------------------------------

def track_stm32(line):
    """Turn the firmware's '#' state-transition prints into a state label."""
    s = line.lstrip("# ").strip()
    if s.startswith("WAIT_START"):
        stm_state["state"], stm_state["corner"] = "ARMED - press START", ""
    elif s.startswith("START refused"):
        stm_state["state"] = s                       # "START refused: lidar stale / IMU has no heading"
    elif s.startswith("PARK OUT start"):
        stm_state.update(state="PARK OUT", corner="", exit="")
    elif s.startswith(("PIVOT", "EXIT", "REALIGN")) and stm_state["state"].startswith("PARK OUT"):
        stm_state["state"] = "PARK OUT - " + s.split(":")[0].split()[0]
    elif s.startswith("PARK OUT ABORT"):
        stm_state["state"] = s.replace("PARK OUT ABORT", "PARK OUT ABORTED")
    elif s.startswith("PARK OUT DONE"):
        stm_state["state"] = "RUNNING"
    elif s.startswith("GO"):
        stm_state.update(state="RUNNING", corner="0/12", exit="")
        RECORDER.start_run()
    elif s.startswith("TURN "):
        stm_state["corner"] = s.split()[1]
    elif s.startswith("next straight"):
        stm_state["exit"] = s.replace("next straight ", "")
    elif s.startswith("lane heading"):
        stm_state["lane"] = s.split()[2]             # "lane heading <deg> turned <deg>"
    elif s.startswith("RECOVER"):
        stm_state["state"] = "RECOVER"
    elif s.startswith("BACKOFF"):
        stm_state["state"] = "BACKOFF (reverse and re-plan)"
    elif s.startswith("DRIVE") or s.startswith("recover") or s.startswith("backoff"):
        stm_state["state"] = "RUNNING"
    elif s.startswith("FINAL_STRAIGHT"):
        stm_state["state"] = "FINAL STRAIGHT"
    elif s.startswith("PARK IN:"):                     # firmware v11 park-in
        stm_state["state"] = "PARK IN - finding the lot"
    elif s.startswith("LOT FOUND"):
        stm_state["state"] = "PARK IN - lot found"
    elif s.startswith(("SWING IN", "BACK:", "MEASURE", "PIVOT IN", "REDO", "SIDESTEP", "SWING OUT")) \
            and stm_state["state"].startswith("PARK IN"):
        stm_state["state"] = "PARK IN - " + s.split(":")[0].split(" (")[0]
    elif s.startswith("PARK IN ABORT"):
        stm_state["state"] = s.replace("PARK IN ABORT", "PARK IN ABORTED")
    elif s.startswith("PARKED"):
        stm_state["state"] = "PARKED"
    elif s.startswith("STOP from Pi") or s.startswith("STOP from button"):
        stm_state["state"] = "STOPPED"
        RECORDER.end_run()
    elif s.startswith("FINISHED"):
        if stm_state["state"] not in ("STOPPED", "PARKED") and "ABORTED" not in stm_state["state"]:
            stm_state["state"] = "FINISHED"
        RECORDER.end_run()
    elif s.startswith("first Pi frame"):
        stm_state["state"] = "connected"


# --------------------------------------------------------------------------
# web page
# --------------------------------------------------------------------------

app = Flask(__name__)

PAGE = r"""<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Obstacle round</title><style>
:root{--bg:#111;--panel:#1b1b1b;--line:#333;--fg:#eee;--dim:#999;--ok:#6d6;--bad:#f66;--acc:#2a7}
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:12px}
h2{margin:0 0 10px}
.tabs{display:flex;gap:4px;border-bottom:2px solid var(--line);margin-bottom:12px;flex-wrap:wrap}
.tabs button{background:none;border:0;border-bottom:3px solid transparent;color:var(--dim);
  font-size:16px;padding:8px 16px;cursor:pointer}
.tabs button.on{color:var(--fg);border-bottom-color:var(--acc)}
.pane{display:none}.pane.on{display:block}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
img{width:100%;max-width:640px;border:1px solid var(--line);background:#000}
button.act{font-size:20px;padding:12px 26px;margin:4px 6px 4px 0;border:0;border-radius:8px;
  cursor:pointer;color:#fff}
#start{background:var(--acc)}#stop{background:#c33}
.small{font-size:13px;padding:7px 12px;background:#444;color:#fff;border:0;border-radius:6px;cursor:pointer}
.big{font-size:19px;font-variant-numeric:tabular-nums;line-height:1.6}
.state{font-size:26px;font-weight:700}
pre{background:#161616;padding:10px;border-radius:6px;height:300px;overflow:auto;font-size:12px;
  white-space:pre-wrap;margin:0}
.bad{color:var(--bad)}.ok{color:var(--ok)}.dim{color:var(--dim)}
fieldset{border:1px solid var(--line);border-radius:8px;margin:0 0 12px;padding:8px 12px 12px;
  background:var(--panel);min-width:300px;flex:1 1 340px}
legend{color:var(--acc);font-weight:600;padding:0 6px;text-transform:uppercase;font-size:12px;
  letter-spacing:.08em}
.p{display:grid;grid-template-columns:1fr 92px;gap:6px 8px;align-items:center;margin:3px 0}
.p label{font-size:13px;font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis}
.p input[type=number]{width:100%;background:#0d0d0d;color:var(--fg);border:1px solid var(--line);
  border-radius:4px;padding:4px 6px;font-size:13px;font-variant-numeric:tabular-nums}
.p input[type=checkbox]{width:20px;height:20px}
.p input.dirty{border-color:#da3;color:#fd8}
.p .u{grid-column:1/3;font-size:11px;color:var(--dim);margin:-2px 0 2px}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5;border-bottom:1px solid var(--line);
  margin-bottom:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
table{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
td{padding:2px 14px 2px 0}td.k{color:var(--dim)}
</style></head><body>

<h2>Obstacle round <span id=hdr class=dim></span></h2>
<div class=tabs>
  <button id=tRun class=on onclick="tab('Run')">Run</button>
  <button id=tTune onclick="tab('Tune')">Tune</button>
  <button id=tTel onclick="tab('Tel')">Telemetry</button>
</div>

<div id=pRun class="pane on"><div class=row>
  <div><img id=cam src="/stream"><br>
    <label><input type=checkbox onchange="cam.src=this.checked?'/stream?view=floor':'/stream'">
    show floor mask</label>
    <div class=dim style="font-size:12px">grey boxes = rejected:
      H too short &middot; A flat &middot; S ragged &middot; F not on mat &middot; C low contrast
      &middot; M not a sign class (YOLO)</div></div>
  <div style="min-width:300px;flex:1">
    <div class=state id=state>&hellip;</div>
    <div class=big id=corner></div>
    <button class=act id=start onclick="cmd('start')">START</button>
    <button class=act id=stop onclick="cmd('stop')">STOP</button>
    <div><button class=small id=rec onclick="rec()">Record</button>
      <span id=recMsg class=dim></span></div>
    <div class=big id=live></div>
    <h3>STM32 log</h3><pre id=log></pre>
  </div>
</div></div>

<div id=pTune class=pane>
  <div class=bar>
    <button class=small onclick="save()">Save to tuning.json</button>
    <button class=small onclick="revert('pi')">Revert Pi</button>
    <button class=small onclick="revert('stm32')">Revert STM32</button>
    <button class=small onclick="loadParams(true)">Reload</button>
    <span id=tuneMsg class=dim></span>
  </div>
  <div id=tunePi></div>
  <h3 style="margin:18px 0 8px">STM32 <span id=stmState class=dim></span></h3>
  <div id=tuneStm></div>
</div>

<div id=pTel class=pane><div class=row>
  <fieldset><legend>Lane</legend><table id=telLane></table></fieldset>
  <fieldset><legend>Vision</legend><table id=telVis></table></fieldset>
  <fieldset><legend>Link</legend><table id=telLink></table></fieldset>
</div><fieldset><legend>Last frame sent</legend><pre id=telLine style="height:auto"></pre></fieldset></div>

<script>
const $=id=>document.getElementById(id);
let pane='Run', dirty={}, built=false;
function tab(n){pane=n;for(const k of ['Run','Tune','Tel']){
  $('p'+k).classList.toggle('on',k===n);
  $({Run:'tRun',Tune:'tTune',Tel:'tTel'}[k]).classList.toggle('on',k===n);}
  if(n==='Tune'&&!built) loadParams(true);}
const f=x=>x===null||x===undefined?'--':Math.round(x);
async function cmd(c){await fetch('/api/'+c,{method:'POST'});}
let recManual=false;
async function rec(){await fetch('/api/record',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({on:!recManual})});}

// ---- Tune ----
const GROUPS={camera:'Camera',hsv:'HSV colour ranges',lab:'Lab colour ranges',
  filter:'Pillar filter (incl. LazyGo ROI / height)',
  cone:'Wall cone fit',locate:'Pillar location',cand:'LiDAR candidates (LazyGo edges)',link:'Link',
  yolo:'YOLO detector',
  record:'Recording (training images to take/)'};
const SGROUPS=['Drive & heading','Colour trigger & turn','Lane planner','Passing a pillar',
  'Wall levelling','Reverse & re-plan','Corner exit','Safety','Link','Park out'];

function field(p,side){
  const id=side+':'+p.name, d=document.createElement('div'); d.className='p';
  const lab=document.createElement('label'); lab.textContent=p.name; lab.title=p.help||p.name;
  const inp=document.createElement('input');
  if(p.kind==='b'){inp.type='checkbox';inp.checked=!!p.value;}
  else{inp.type='number';inp.value=p.value;
       inp.step=(p.kind==='i')?1:(Math.abs(p.hi-p.lo)<=2?0.01:(Math.abs(p.hi-p.lo)<=50?0.1:1));
       inp.min=p.lo;inp.max=p.hi;}
  inp.id=id; inp.title=`${p.lo} .. ${p.hi}`;
  inp.onchange=()=>send(side,p.name,p.kind==='b'?(inp.checked?1:0):inp.value,inp);
  d.append(lab,inp);
  if(p.help){const u=document.createElement('div');u.className='u';u.textContent=p.help;d.append(u);}
  return d;
}
async function send(side,name,value,inp){
  inp.classList.add('dirty');
  const r=await (await fetch('/api/param',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({side,name,value})})).json();
  if(r.ok){inp.classList.remove('dirty');
    if(inp.type!=='checkbox')inp.value=r.value; $('tuneMsg').textContent=name+' = '+r.value;}
  else{$('tuneMsg').innerHTML='<span class=bad>'+name+': '+r.error+'</span>';}
}
function section(title,items,side){
  const fs=document.createElement('fieldset');
  const lg=document.createElement('legend'); lg.textContent=title; fs.append(lg);
  items.forEach(p=>fs.append(field(p,side)));
  return fs;
}
async function loadParams(rebuild){
  const r=await (await fetch('/api/params')).json();
  $('stmState').textContent=r.stm32_synced
    ? `v${r.stm32_version}, ${r.stm32.length} parameters, boot ${r.stm32_boot}`
    : 'not connected - values appear once the STM32 answers';
  if(!rebuild&&built) return;
  const pi=$('tunePi'); pi.innerHTML=''; const row=document.createElement('div'); row.className='row';
  for(const g in GROUPS){const items=r.pi.filter(p=>p.group===g);
    if(items.length) row.append(section(GROUPS[g],items,'pi'));}
  pi.append(row);
  const st=$('tuneStm'); st.innerHTML=''; const row2=document.createElement('div'); row2.className='row';
  SGROUPS.forEach((g,i)=>{const items=r.stm32.filter(p=>p.group===i);
    if(items.length) row2.append(section(g,items,'stm32'));});
  st.append(row2); built=r.stm32.length>0;
}
async function save(){const r=await (await fetch('/api/save',{method:'POST'})).json();
  $('tuneMsg').textContent=r.msg;}
async function revert(side){
  if(!confirm('Put '+side+' parameters back to their defaults?'))return;
  const r=await (await fetch('/api/revert',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({side})})).json();
  $('tuneMsg').textContent=r.msg; built=false; setTimeout(()=>loadParams(true),600);}

// ---- poll ----
function rows(t,pairs){t.innerHTML=pairs.map(([k,v])=>
  `<tr><td class=k>${k}</td><td>${v}</td></tr>`).join('');}
async function tick(){
 try{
  const r=await (await fetch('/status')).json();
  $('hdr').textContent=r.serial_ok?'':'  (STM32 port closed)';
  $('state').textContent=r.stm32_state;
  $('corner').textContent=(r.corner?('corner '+r.corner):'')+(r.exit?('  -  exit '+r.exit):'');
  const lk=r.serial_ok?(r.lidar_live?'<span class=ok>lidar live</span>':
    '<span class=bad>lidar STALE</span>'):'<span class=bad>STM32 port closed</span>';
  $('live').innerHTML=
   `${lk} <span class=dim>${r.hz.toFixed(0)} Hz</span><br>`+
   `detector <b>${r.detector||'--'}</b> <span class=dim>${r.infer_ms?r.infer_ms+' ms, ':''}`+
     `${r.vis_fps} fps</span>`+(r.yolo_error?` <span class=bad title="${r.yolo_error}">YOLO error</span>`:'')+`<br>`+
   `L ${f(r.left)} &nbsp;F ${f(r.front)} &nbsp;R ${f(r.right)} mm<br>`+
   `cone L ${f(r.cone_left)} &nbsp;R ${f(r.cone_right)} mm &nbsp;yaw ${r.wall_yaw===null?'--':r.wall_yaw+'°'}<br>`+
   `pillar <b>${r.name}</b> err ${r.error} area ${r.area}`+
   (r.pillar_xy?` &nbsp;at ${r.pillar_xy[0]} fwd, ${r.pillar_xy[1]} left mm`:'')+
   (r.unknown_xy?`<br>lidar object (colour unknown) ${r.unknown_xy[0]} fwd, ${r.unknown_xy[1]} left mm`:'');
  const rc=r.rec||{};
  recManual=!!rc.manual;
  $('rec').textContent=recManual?'Stop recording':'Record';
  $('rec').style.background=recManual?'#c33':'#444';
  $('recMsg').innerHTML=(rc.on?`<span class=bad>&#9679; REC</span> take/${rc.folder||'...'} `+
     `${rc.saved} saved`:(rc.folder?`last: take/${rc.folder} (${rc.saved} frames)`:'not recording'))+
    (rc.skipped?` <span class=dim>(${rc.skipped} still frames skipped)</span>`:'')+
    (rc.dropped?` <span class=dim>(${rc.dropped} dropped)</span>`:'')+
    (rc.error?` <span class=bad>${rc.error}</span>`:'');
  const lg=$('log'), atBottom=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-5;
  lg.textContent=r.log.join('\n'); if(atBottom) lg.scrollTop=lg.scrollHeight;
  if(pane==='Tel'){
    rows($('telLane'),[['cone left',f(r.cone_left)+' mm'],['cone right',f(r.cone_right)+' mm'],
      ['wall yaw',(r.wall_yaw===null?'--':r.wall_yaw+'°')],
      ['lane heading',r.lane||'--'],['corner',r.corner||'--'],['planned exit',r.exit||'--']]);
    rows($('telVis'),[['detector',r.detector||'--'],['inference',r.infer_ms+' ms'],
      ['camera loop',r.vis_fps+' fps'],['colour',r.name],['err',r.error+' px'],['area',r.area],
      ['frame seq',r.vseq],
      ['pillar x,y',r.pillar_xy?r.pillar_xy.join(', ')+' mm':'--'],
      ['unknown x,y',r.unknown_xy?r.unknown_xy.join(', ')+' mm':'--']]);
    rows($('telLink'),[['lidar',r.lidar_live?'live':'STALE'],['serial',r.serial_ok?'open':'closed'],
      ['frame rate',r.hz.toFixed(1)+' Hz'],['front/left/right',
       f(r.front)+' / '+f(r.left)+' / '+f(r.right)+' mm']]);
    $('telLine').textContent=r.last_line||'(nothing sent - lidar stale)';
  }
  if(pane==='Tune'&&!built) loadParams(true);
 }catch(e){$('state').textContent='Pi not reachable';}
 setTimeout(tick,200);
}
tick();
</script></body></html>"""


@app.route("/favicon.ico")
def favicon():
    return ("", 204)       # keeps the browser console clean


@app.route("/")
def index():
    return PAGE


@app.route("/status")
def status():
    v = vision_now(time.monotonic())
    return jsonify({
        "color": v["color"], "name": NAMES[v["color"]],      # 1=red, 0=green, 2=none
        "error": v["err"], "area": v["area"], "vseq": v["seq"],
        "detector": v.get("mode", ""), "infer_ms": round(v.get("infer_ms", 0.0), 1),
        "vis_fps": round(v.get("fps", 0.0), 1), "yolo_error": v.get("yolo_error"),
        "left": link["l"], "front": link["f"], "right": link["r"],
        "cone_left": link["cl"], "cone_right": link["cr"],
        "wall_yaw": None if link["yaw"] is None else round(link["yaw"], 1),
        "pillar_xy": None if link["pxy"] is None else
                     [round(link["pxy"][0]), round(link["pxy"][1])],
        "unknown_xy": None if link.get("uxy") is None else
                      [round(link["uxy"][0]), round(link["uxy"][1])],
        "lidar_live": link["live"], "serial_ok": link["serial_ok"], "hz": link["hz"],
        "last_line": link["line"].strip(),
        "stm32_state": stm_state["state"], "corner": stm_state["corner"],
        "exit": stm_state["exit"], "lane": stm_state["lane"],
        "rec": RECORDER.status(),
        "log": list(stm_log),
    })


@app.route("/api/record", methods=["POST"])
def api_record():
    """The Record button: save frames without a run (car pushed by hand)."""
    body = request.get_json(force=True, silent=True) or {}
    RECORDER.set_manual(body.get("on", not RECORDER.manual))
    return jsonify({"ok": True, "rec": RECORDER.status()})


@app.route("/api/start", methods=["POST"])
def api_start():
    commands.put(CMD_START)
    note("[pi] START pressed")
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    commands.put(CMD_STOP)
    note("[pi] STOP pressed")
    return jsonify({"ok": True})


@app.route("/api/params")
def api_params():
    return jsonify({
        "pi": PI.describe(),
        "stm32": STM.describe(),
        "stm32_synced": STM.synced,
        "stm32_version": STM.version,
        "stm32_boot": STM.boot,
    })


@app.route("/api/param", methods=["POST"])
def api_param():
    """One parameter. Pi values apply on the next frame; STM32 values are
    queued and pushed by the serial loop, so this never blocks on the port."""
    body = request.get_json(force=True, silent=True) or {}
    side, name, value = body.get("side"), body.get("name"), body.get("value")
    try:
        if side == "pi":
            v = PI.set(name, value)
            sync_globals()
        elif side == "stm32":
            v = STM.set(name, value)
        else:
            return jsonify({"ok": False, "error": "unknown side"}), 400
    except KeyError:
        return jsonify({"ok": False, "error": "no such parameter"}), 404
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e) or "bad value"}), 400
    return jsonify({"ok": True, "value": v})


@app.route("/api/save", methods=["POST"])
def api_save():
    try:
        msg = prm.save(PI, STM)
    except OSError as e:
        return jsonify({"ok": False, "msg": f"could not save: {e}"}), 500
    note(msg)
    return jsonify({"ok": True, "msg": msg})


@app.route("/api/revert", methods=["POST"])
def api_revert():
    side = (request.get_json(force=True, silent=True) or {}).get("side")
    if side == "pi":
        PI.reset()
        sync_globals()
        msg = "[pi] Pi parameters back to defaults (not saved yet)"
    elif side == "stm32":
        STM.reset()
        msg = "[pi] STM32 parameters back to firmware defaults (not saved yet)"
    else:
        return jsonify({"ok": False, "msg": "unknown side"}), 400
    note(msg)
    return jsonify({"ok": True, "msg": msg})


def mjpeg(view="camera"):
    global viewers
    with lock:
        viewers += 1
    try:
        while True:
            with lock:
                frame = None if latest_frame is None else latest_frame.copy()
                v = dict(vision)
            if frame is None:
                time.sleep(0.05)
                continue
            if view == "floor" and v.get("floor") is not None:
                bgr = cv2.cvtColor(cv2.resize(v["floor"], (camera.FRAME_W, camera.FRAME_H),
                                              interpolation=cv2.INTER_NEAREST),
                                   cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.line(bgr, (camera.FRAME_W // 2, 0),
                     (camera.FRAME_W // 2, camera.FRAME_H), (255, 255, 255), 1)
            for (x0, y0, x1, y1), _, why in v.get("cands", []):
                if why is not None:
                    lot = why.startswith("LOT")                # the parking lot: magenta
                    col, th = ((255, 0, 255), 2) if lot else ((140, 140, 140), 1)
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), col, th)
                    cv2.putText(bgr, why, (x0, max(12, y0 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            for (x0, y0, x1, y1), code, conf in v.get("dets", []):
                cv2.rectangle(bgr, (x0, y0), (x1, y1), BOX_BGR[code], 1)
                if conf < 1.0:                               # YOLO: show the confidence
                    cv2.putText(bgr, f"{conf:.2f}", (x0, max(12, y0 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_BGR[code], 1)
            if v["box"] is not None and v["color"] != COL_NONE:
                col = BOX_BGR[v["color"]]
                x0, y0, x1, y1 = v["box"]
                cv2.rectangle(bgr, (x0, y0), (x1, y1), col, 2)
                cv2.putText(bgr, f"c:{v['color']} err:{v['err']} area:{v['area']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(bgr, f"{v.get('mode', '')}  {v.get('infer_ms', 0):.0f} ms  "
                             f"{v.get('fps', 0):.0f} fps",
                        (10, camera.FRAME_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
            ok, jpg = cv2.imencode(".jpg", bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg.tobytes() + b"\r\n")
            time.sleep(0.03)
    finally:
        with lock:
            viewers -= 1


@app.route("/stream")
def stream():
    view = request.args.get("view", "camera")        # camera | floor
    return Response(mjpeg(view), mimetype="multipart/x-mixed-replace; boundary=frame")


def run_flask():
    app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True,
            debug=False, use_reloader=False)


# --------------------------------------------------------------------------
# main - the ONLY place that writes to the serial port, so a command or a
# tuning line can never be spliced into the middle of a frame line
# --------------------------------------------------------------------------

def handle_rx(text):
    """One line from the STM32. '!' = parameter protocol, anything else = log."""
    if text.startswith("!"):
        msg = STM.on_line(text)
        if msg:
            note(msg)
        if STM.synced and not handle_rx.pushed:
            handle_rx.pushed = True
            STM.queue_all()
            note("[pi] pushing saved tuning to the STM32")
        if not STM.synced:
            handle_rx.pushed = False
        return
    print("[stm32] " + text)
    stm_log.append(text)
    track_stm32(text)


handle_rx.pushed = False


def main():
    PI.set("BEARING_TOL_DEG", load_tol())      # config.json first ...
    msg = prm.load(PI, STM)                    # ... tuning.json overrides it if it has one
    sync_globals()
    note(msg)

    try:
        ser = serial.Serial(UART_PORT, UART_BAUD, timeout=0, write_timeout=0)
    except serial.SerialException as e:
        import glob
        found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        raise SystemExit(f"[obstacle] cannot open {UART_PORT}: {e}\n"
                         f"           USB serial devices present: "
                         f"{', '.join(found) or 'none found'}")
    link["serial_ok"] = True

    shared = SharedState()
    lidar = LidarThread(shared)
    lidar.start()
    RECORDER.start()
    vis = VisionThread()
    vis.start()
    threading.Thread(target=run_flask, name="Flask", daemon=True).start()

    print(f"[obstacle] lidar+camera -> {UART_PORT}. "
          f"Page on http://<pi>:{FLASK_PORT}. Ctrl-C to stop.")

    last_log = time.monotonic()
    was_live = False
    rx_buf = b""
    f = l = r = None
    cl = cr = yaw = None
    cands = []
    cone_rev = -1
    frames = 0
    v = vision_now(time.monotonic())
    STM.request_dump()                      # ask who it is as soon as it answers
    retry = None                            # a line the port would not take

    def write(data):
        """Non-blocking. Returns False if the port is full; caller retries."""
        try:
            ser.write(data)
            return True
        except serial.SerialTimeoutException:
            return False

    try:
        while True:
            now = time.monotonic()
            live = lidar_live(lidar, now)
            link["live"] = live

            # 1. button presses, whatever the lidar is doing
            while not commands.empty():
                c = commands.get_nowait()
                for _ in range(CMD_REPEAT):
                    write(c)
                print(f"[obstacle] sent {CMD_NAME[c]}")

            # 2. tuning, rate-limited so the frame feed keeps its slot
            if retry is not None:
                if write(retry):
                    retry = None
            else:
                for line in STM.drain(STM_PUSH_PER_LOOP):
                    if not write((line + "\n").encode("ascii")):
                        retry = (line + "\n").encode("ascii")
                        break

            # 3. the sensor frame
            if live:
                f, l, r = read_three(lidar._ranges, lidar._quals, BEARING_TOL_DEG)
                if lidar.rev != cone_rev:            # bins only change once per rev
                    cone_rev = lidar.rev
                    cl, cr, yaw, lline, rline = cones_full(lidar._ranges)
                    cands = lidar_candidates(lidar._ranges, lline, rline)
                v = vision_now(now)
                pxy = (locate_pillar(lidar._ranges, v["bearing"], v["area"])
                       if v["color"] != COL_NONE else None)
                pX, pY = _pxy(pxy)
                uxy = unclassified(cands, (pxy,))
                uX, uY = _pxy(uxy)
                mxy = locate_block(lidar._ranges, v["m_bearing"])  # the parking lot
                mX, mY = _pxy(mxy)
                line = (f"{_u16(l)},{_u16(f)},{_u16(r)},{lidar.rev},"
                        f"{v['color']},{v['err']},{v['area']},{v['seq']},"
                        f"{_cone_u16(cl)},{_cone_u16(cr)},{_ang(yaw)},{pX},{pY},{uX},{uY},"
                        f"{COL_NONE},{PXY_NONE},{PXY_NONE},{mX},{mY}\n")   # no second sign; lot
                if write(line.encode("ascii")):
                    frames += 1
                link.update(line=line, t=now, f=f, l=l, r=r, cl=cl, cr=cr,
                            yaw=yaw, pxy=pxy, uxy=uxy, mxy=mxy)
            if live != was_live:
                note("[obstacle] lidar LIVE - streaming" if live else
                     "[obstacle] lidar STALE - silent")
                was_live = live

            # 4. everything the STM32 has said
            n = ser.in_waiting
            if n:
                rx_buf += ser.read(n)
                *lines, rx_buf = rx_buf.split(b"\n")
                for ln in lines:
                    text = ln.decode("ascii", "replace").rstrip()
                    if text:
                        handle_rx(text)
                if len(rx_buf) > 512:
                    rx_buf = b""

            if now - last_log >= 1.0:
                link["hz"] = frames / (now - last_log)
                frames = 0
                fmt = lambda x: "----" if x is None else f"{int(x):4d}"
                name = {COL_RED: "RED", COL_GREEN: "GRN"}.get(v["color"], "---")
                print(f"[obstacle] F {fmt(f)} L {fmt(l)} R {fmt(r)} mm | "
                      f"{name} err {v['err']:+4d} area {v['area']:5d} | "
                      f"{stm_state['state']} {stm_state['corner']} | "
                      f"{link['hz']:.0f} Hz queue {lidar.queue_depth()}"
                      f"{'' if live else '  (STALE)'}"
                      f"{'  (NO CAMERA)' if vis.error else ''}")
                last_log = now

            time.sleep(1.0 / max(1, SEND_HZ))
    except KeyboardInterrupt:
        pass
    except serial.SerialException as e:
        print(f"[obstacle] serial error: {e} (STM32 unplugged/reset?)")
    finally:
        link["serial_ok"] = False
        try:
            for _ in range(CMD_REPEAT):       # stop the car before letting go
                ser.write(CMD_STOP)
            ser.flush()
        except Exception:
            pass
        vis.stop()
        lidar.stop()
        vis.join(timeout=2.0)
        lidar.join(timeout=2.0)
        RECORDER.stop()                       # after the camera: closes frames.csv
        RECORDER.join(timeout=2.0)
        ser.close()
        print("[obstacle] stopped")


if __name__ == "__main__":
    main()
