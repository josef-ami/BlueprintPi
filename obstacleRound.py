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

Wire frame - one ASCII line per send, SEND_HZ times a second:

    left,front,right,rev,color,err,area,vseq,coneL,coneR,wallAng,pX,pY,uX,uY,
    sColor,sX,sY,navOk,navCross,navHdgDd,navCurvUm,navDone\\n

  left/front/right  mm at 90 / 0 / 270 deg, 65535 = no return   (as openRound)
  rev               lidar revolution counter                     (as openRound)
  color             1 = red, 0 = green, 2 = none                 (as tracker.py)
  err               pillar centre x - 320, in 640-px frame units, + = right
  area              largest ACCEPTED pillar area, 320x240 detection pixels
  vseq              camera frame counter; STM32 runs its pillar PD only
                    when it changes
  coneL / coneR     perpendicular mm to the left / right wall, line-fitted
                    over a 45 deg LiDAR cone each side; 65535 = no fit
  wallAng           car yaw to the walls x10 (deci-deg), + = pointing left;
                    32767 = no fit
  pX / pY           chosen pillar centre, mm from the LiDAR (x fwd, y left):
                    camera bearing (fisheye-undistorted) + LiDAR range on
                    that ray; 32767 = none
  uX / uY           nearest LiDAR object inside the corridor that is NOT the
                    camera's pillar (colour unknown yet); 32767 = none
  sColor            SECOND pillar's colour - the next-largest accepted blob,
                    i.e. one that is further away but still seen. 2 = none
  sX / sY           where it is, same frame as pX/pY. Approaching a corner this
                    is almost always the first pillar of the NEXT straight, and
                    the firmware shapes the corner from its colour: green means
                    go further forward and take it wide, red means turn early
                    and take it short.

Commands on the same serial line:
    S            START            X            STOP
    N <name> <v> set a firmware parameter       ?P  dump the table
    ?V           firmware version / boot id     C   recompute derived values
Replies from the firmware start with '!' (parameters) or '#' (log).

Silence rules: lidar stale -> no frames are sent (commands still are).
Camera stale (> VISION_STALE_S) -> color is sent as 2 (none).

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
import math
import os
import queue
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
from nav import NavService

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

# ---------------- nav geometry, measure these on the car ----------------
NAV_TICKS_PER_MM    = 1.4853   # = firmware TICKS_PER_CM / 10
NAV_LIDAR_AHEAD_MM  = 130.0    # LiDAR centre ahead of the REAR AXLE (CAD: 130)
NAV_CAMERA_AHEAD_MM = 150.0    # camera ahead of the rear axle
NAV = None

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
             if s.group in ("camera", "cone", "locate", "cand", "link", "lab")]

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
CAND_GAP_MM = 60.0
CAND_MAX_WIDTH_MM = 120.0
CAND_MIN_POINTS = 2
SEND_HZ = 50
CMD_REPEAT = 3
BEARING_TOL_DEG = 2
STM_PUSH_PER_LOOP = 4

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
          "s_color": COL_NONE, "s_area": 0, "s_box": None, "s_bearing": None}
latest_frame = None                              # RGB, only kept while someone watches
viewers = 0

link = {"line": "", "t": 0.0, "live": False, "serial_ok": False,
        "f": None, "l": None, "r": None, "cl": None, "cr": None,
        "yaw": None, "pxy": None, "uxy": None, "sxy": None,
        "s_name": "none", "hz": 0.0}
stm_log = collections.deque(maxlen=LOG_LINES)
stm_state = {"state": "waiting for STM32", "corner": "", "exit": "", "lane": ""}
commands = queue.Queue()                          # written to serial by the main loop only


def note(msg):
    print(msg)
    stm_log.append(msg)


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

def largest(mask):
    """Largest contour above MIN_AREA_PROC, no shape checks.
    Kept for anything that imported it; the robot uses find_pillars()."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, area = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a > area and a >= MIN_AREA_PROC:
            best, area = c, a
    return best, area


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


def check_pillar(cnt, area, hsv, lab, floor, pf):
    """None if the blob is a pillar, else a one-letter reject code (A/S/F/C)."""
    x, y, w, h = cv2.boundingRect(cnt)
    if h < pf["aspect_min"] * w:
        return "A"
    if area < pf["solidity_min"] * w * h:
        return "S"

    H, W = floor.shape
    y0 = y + h
    if y0 >= H - pf["bottom_margin_px"]:
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


def find_pillars(hsv, lab, pf):
    """(accepted, candidates, floor).

    accepted = every blob that passed, as (cnt, area, code), LARGEST FIRST.
    The firmware wants two of them: the nearest pillar to steer around, and the
    next one back. On the run up to a corner that second pillar is almost
    always the first pillar of the next straight, and its colour is what decides
    whether the corner is taken short or wide - see planCornerExit() in the
    firmware. Sending only the largest blob, as this did before, made that
    pillar invisible exactly when it mattered.

    candidates = [(cnt, area, code, reject_or_None)] for the debug overlay."""
    floor = floor_mask(hsv, lab, pf)
    cands, accepted = [], []
    for name in ("RED", "GREEN"):
        cnts, _ = cv2.findContours(colour_mask(hsv, lab, name),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < MIN_AREA_PROC:
                continue
            why = check_pillar(c, a, hsv, lab, floor, pf)
            cands.append((c, a, CODE[name], why))
            if why is None:
                accepted.append((c, a, CODE[name]))
    accepted.sort(key=lambda t: -t[1])
    return accepted, cands, floor


class VisionThread(threading.Thread):
    """Reads the tunables fresh every frame, so the Tune tab is live: there is
    no restart, no re-open of the camera, and the thread never blocks on the
    web side."""

    def __init__(self):
        super().__init__(name="Vision", daemon=True)
        self._halt = threading.Event()   # NOT _stop: that name is taken by Thread
        camera.load_intrinsics(camera.load_config())
        self.error = None

    def stop(self):
        self._halt.set()

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
        try:
            while not self._halt.is_set():
                pf = PILLAR_FILTER                        # one read, one frame
                frame = camera.grab_rgb(cam, SWAP_RB)
                small = cv2.resize(frame, PROC_SIZE, interpolation=cv2.INTER_AREA)
                hsv, lab = colour_spaces(small)

                accepted, cands, floor = find_pillars(hsv, lab, pf)

                def describe(hit):
                    cnt, area, colour = hit
                    x, y, w, h = cv2.boundingRect(cnt)
                    return (colour,
                            int((x + w / 2.0) * sx) - centre,
                            int(area),
                            (int(x * sx), int(y * sy),
                             int((x + w) * sx), int((y + h) * sy)),
                            bearing_from_px((x + w / 2.0) * sx, (y + h / 2.0) * sy))

                seq = (seq + 1) & 0xFFFFFFFF
                if not accepted:
                    color, err, area, box, bearing = COL_NONE, 0, 0, None, None
                else:
                    color, err, area, box, bearing = describe(accepted[0])
                if len(accepted) > 1:
                    s_color, _, s_area, s_box, s_bearing = describe(accepted[1])
                else:
                    s_color, s_area, s_box, s_bearing = COL_NONE, 0, None, None

                with lock:
                    vision.update(color=color, err=err, area=int(area),
                                  seq=seq, t=time.monotonic(), box=box, bearing=bearing,
                                  s_color=s_color, s_area=s_area, s_box=s_box,
                                  s_bearing=s_bearing)
                    if viewers > 0:
                        latest_frame = frame
                        vision["cands"] = [
                            ((int(bx * sx), int(by * sy), int((bx + bw) * sx), int((by + bh) * sy)),
                             code, why)
                            for (c, _, code, why) in cands
                            for (bx, by, bw, bh) in [cv2.boundingRect(c)]]
                        vision["floor"] = floor
                    else:
                        latest_frame = None
        finally:
            try:
                cam.stop()
            except Exception:
                pass
            print("[vision] stopped")


def vision_now(now):
    with lock:
        v = dict(vision)
    if now - v["t"] > VISION_STALE_S:
        v["color"], v["err"], v["area"], v["bearing"] = COL_NONE, 0, 0, None
        v["s_color"], v["s_area"], v["s_bearing"] = COL_NONE, 0, None
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


def cones(ranges):
    """(coneL_mm, coneR_mm, yaw_deg) - any of them None when not fitted."""
    return cones_full(ranges)[:3]


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


# --------------------------------------------------------------------------
# LiDAR pillar candidates - objects inside the corridor, colour unknown
# --------------------------------------------------------------------------
#
# The camera only covers a limited wedge, so a pillar near the far wall can
# stay out of view until it is too late. The LiDAR sees it from the start line.
# A candidate is a small cluster of returns that is:
#   * ahead of the car (x > 0) and closer than CAND_MAX_MM
#   * inside the corridor: more than CAND_WALL_MM from both fitted wall lines
#     (a missing side is placed CORRIDOR_MM from the other)
#   * short of the wall ahead (front distance - CAND_WALL_MM)
#   * no wider than CAND_MAX_WIDTH_MM (a pillar is 50 mm, 71 mm diagonal)
# The nearest one is sent; the STM32 lines up with it so the camera can name
# the colour, and dodges to the roomier side if it never does.

def lidar_candidates(ranges, left_line, right_line):
    """[(x, y)] pillar-centre candidates in the LiDAR frame, nearest first.

    Cluster FIRST, filter second: a wall is one long run of returns and is
    thrown out whole by the width test. Filtering first would chop a wall
    into short fragments at the cut lines, and those look like pillars.
    """
    if left_line is None and right_line is None:
        return []                                    # corner: no corridor to search in
    r = np.asarray(ranges, dtype=np.float64)
    ok = np.isfinite(r) & (r > 80) & (r < CAND_MAX_MM + 400)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return []
    x, y = r[idx] * _COS[idx], r[idx] * _SIN[idx]
    fwd = r[(np.arange(-3, 4)) % 360]
    fwd = fwd[np.isfinite(fwd)]
    front = float(np.median(fwd)) if fwd.size else np.inf

    clusters, cur = [], [0]
    for k in range(1, idx.size):
        if idx[k] - idx[cur[-1]] <= 3 and \
                math.hypot(x[k] - x[cur[-1]], y[k] - y[cur[-1]]) < CAND_GAP_MM:
            cur.append(k)
        else:
            clusters.append(cur); cur = [k]
    clusters.append(cur)
    if len(clusters) > 1 and idx[0] + 360 - idx[-1] <= 3 and \
            math.hypot(x[0] - x[-1], y[0] - y[-1]) < CAND_GAP_MM:   # wrap at 0/359
        clusters[0] = clusters[-1] + clusters[0]; clusters.pop()

    out = []
    for c in clusters:
        if len(c) < CAND_MIN_POINTS:
            continue
        cx, cy = x[c], y[c]
        if math.hypot(cx.max() - cx.min(), cy.max() - cy.min()) > CAND_MAX_WIDTH_MM:
            continue                                  # wall run
        mx, my = float(cx.mean()), float(cy.mean())
        d = math.hypot(mx, my)
        if mx <= 0 or d > CAND_MAX_MM or mx > front - CAND_WALL_MM:
            continue
        if left_line is not None:
            if my > left_line[0] * mx + left_line[1] - CAND_WALL_MM: continue
        elif my > right_line[0] * mx + right_line[1] + CORRIDOR_MM - CAND_WALL_MM: continue
        if right_line is not None:
            if my < right_line[0] * mx + right_line[1] + CAND_WALL_MM: continue
        elif my < left_line[0] * mx + left_line[1] - CORRIDOR_MM + CAND_WALL_MM: continue
        out.append((mx + FACE_TO_CENTRE_MM * mx / d,
                    my + FACE_TO_CENTRE_MM * my / d))   # face -> centre
    out.sort(key=lambda p: math.hypot(*p))
    return out


def unclassified(cands, classified_xy, match_mm=200.0):
    """Nearest candidate that is NOT the pillar the camera already named."""
    for c in cands:
        if classified_xy is None or \
                math.hypot(c[0] - classified_xy[0], c[1] - classified_xy[1]) > match_mm:
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
        stm_state["state"] = "START refused (lidar stale)"
    elif s.startswith("GO"):
        stm_state.update(state="RUNNING", corner="0/12", exit="")
    elif s.startswith("TURN "):
        stm_state["corner"] = s.split()[1]
    elif s.startswith("next straight"):
        stm_state["exit"] = s.replace("next straight ", "")
    elif s.startswith("lane heading"):
        stm_state["lane"] = s.split()[-1]
    elif s.startswith("RECOVER"):
        stm_state["state"] = "RECOVER"
    elif s.startswith("DRIVE") or s.startswith("recover"):
        stm_state["state"] = "RUNNING"
    elif s.startswith("FINAL_STRAIGHT"):
        stm_state["state"] = "FINAL STRAIGHT"
    elif s.startswith("STOP from Pi"):
        stm_state["state"] = "STOPPED"
    elif s.startswith("FINISHED"):
        if stm_state["state"] != "STOPPED":
            stm_state["state"] = "FINISHED"
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
      A flat &middot; S ragged &middot; F not on mat &middot; C low contrast</div></div>
  <div style="min-width:300px;flex:1">
    <div class=state id=state>&hellip;</div>
    <div class=big id=corner></div>
    <button class=act id=start onclick="cmd('start')">START</button>
    <button class=act id=stop onclick="cmd('stop')">STOP</button>
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

// ---- Tune ----
const GROUPS={camera:'Camera',hsv:'HSV colour ranges',filter:'Pillar filter',
  cone:'Wall cone fit',locate:'Pillar location',cand:'LiDAR candidates',link:'Link'};
const SGROUPS=['Drive & heading','Corner trigger & arc','Lane planner','Passing a pillar',
  'Wall levelling','3-point corner','Corner exit','Safety','Link'];

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
   `L ${f(r.left)} &nbsp;F ${f(r.front)} &nbsp;R ${f(r.right)} mm<br>`+
   `cone L ${f(r.cone_left)} &nbsp;R ${f(r.cone_right)} mm &nbsp;yaw ${r.wall_yaw===null?'--':r.wall_yaw+'°'}<br>`+
   `pillar <b>${r.name}</b> err ${r.error} area ${r.area}`+
   (r.second&&r.second!=='none'?` &nbsp;<span class=dim>2nd</span> <b>${r.second}</b>`+
     (r.second_xy?` at ${r.second_xy[0]},${r.second_xy[1]}`:''):'')+
   (r.pillar_xy?` &nbsp;at ${r.pillar_xy[0]} fwd, ${r.pillar_xy[1]} left mm`:'')+
   (r.unknown_xy?`<br>lidar object (colour unknown) ${r.unknown_xy[0]} fwd, ${r.unknown_xy[1]} left mm`:'');
  const lg=$('log'), atBottom=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-5;
  lg.textContent=r.log.join('\n'); if(atBottom) lg.scrollTop=lg.scrollHeight;
  if(pane==='Tel'){
    rows($('telLane'),[['cone left',f(r.cone_left)+' mm'],['cone right',f(r.cone_right)+' mm'],
      ['wall yaw',(r.wall_yaw===null?'--':r.wall_yaw+'°')],
      ['lane heading',r.lane||'--'],['corner',r.corner||'--'],['planned exit',r.exit||'--']]);
    rows($('telVis'),[['colour',r.name],['err',r.error+' px'],['area',r.area],
      ['frame seq',r.vseq],
      ['pillar x,y',r.pillar_xy?r.pillar_xy.join(', ')+' mm':'--'],
      ['2nd pillar',r.second+(r.second_xy?' at '+r.second_xy.join(', ')+' mm':'')],
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
        "left": link["l"], "front": link["f"], "right": link["r"],
        "cone_left": link["cl"], "cone_right": link["cr"],
        "wall_yaw": None if link["yaw"] is None else round(link["yaw"], 1),
        "pillar_xy": None if link["pxy"] is None else
                     [round(link["pxy"][0]), round(link["pxy"][1])],
        "unknown_xy": None if link.get("uxy") is None else
                      [round(link["uxy"][0]), round(link["uxy"][1])],
        "second": link.get("s_name", "none"),
        "second_xy": None if link.get("sxy") is None else
                     [round(link["sxy"][0]), round(link["sxy"][1])],
        "lidar_live": link["live"], "serial_ok": link["serial_ok"], "hz": link["hz"],
        "last_line": link["line"].strip(),
        "stm32_state": stm_state["state"], "corner": stm_state["corner"],
        "exit": stm_state["exit"], "lane": stm_state["lane"],
        "log": list(stm_log),
    })


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
                    cv2.rectangle(bgr, (x0, y0), (x1, y1), (140, 140, 140), 1)
                    cv2.putText(bgr, why, (x0, max(12, y0 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)
            if v["box"] is not None and v["color"] != COL_NONE:
                col = BOX_BGR[v["color"]]
                x0, y0, x1, y1 = v["box"]
                cv2.rectangle(bgr, (x0, y0), (x1, y1), col, 2)
                cv2.putText(bgr, f"c:{v['color']} err:{v['err']} area:{v['area']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
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
    """One line from the STM32. '!' = parameters, 'T' = telemetry, else a log."""
    if text.startswith("T "):
        # T <heading_deci_deg> <encoder_ticks> <state>
        # This is what lets nav dead-reckon between LiDAR revolutions. Without
        # it the pose would have to coast ~100 ms on no information at all,
        # which at 400 mm/s is 40 mm of drift per scan.
        if NAV is not None:
            try:
                _, hd, enc, _st = text.split()
                NAV.on_telemetry(int(hd) / 10.0, int(enc))
            except ValueError:
                pass
        return
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
    msg = prm.load(PI, STM)
    sync_globals()
    note(msg)
    PI.set("BEARING_TOL_DEG", load_tol())      # honour config.json unless tuning.json overrode it
    sync_globals()

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
    vis = VisionThread()
    vis.start()

    # The nav solve takes ~200 ms and MUST NOT run in this loop - it would
    # stall the feed past the firmware's LIDAR_STALE_MS of 200 mid-corner.
    # NavService does the heavy work on its own thread and this loop only
    # reads five already-computed integers out of it.
    global NAV
    # MUST match the firmware's TICKS_PER_CM / 10. If you re-calibrate
    # odometry, change it in BOTH places or the dead reckoning between scans
    # will be wrong by that ratio.
    NAV = NavService(lidar, ticks_per_mm=NAV_TICKS_PER_MM,
                     lidar_ahead_mm=NAV_LIDAR_AHEAD_MM,
                     camera_ahead_mm=NAV_CAMERA_AHEAD_MM,
                     log=note)
    NAV.start()
    note("[pi] nav service started (firmware param USE_NAV_TRACK gates it)")
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
                sxy = (locate_pillar(lidar._ranges, v["s_bearing"], v["s_area"])
                       if v["s_color"] != COL_NONE else None)
                sX, sY = _pxy(sxy)
                uxy = unclassified(cands, pxy)
                uX, uY = _pxy(uxy)
                # hand the camera's colours to nav. It only needs a BEARING:
                # position comes from the LiDAR, which has 360 deg of coverage
                # and no 885 mm range limit.
                if NAV is not None:
                    dets = []
                    if v["color"] != COL_NONE and v["bearing"] is not None:
                        dets.append((v["color"], v["bearing"]))
                    if v["s_color"] != COL_NONE and v["s_bearing"] is not None:
                        dets.append((v["s_color"], v["s_bearing"]))
                    NAV.on_camera(dets)
                    nOk, nCross, nHdg, nCurv, nDone = NAV.wire_fields()
                else:
                    nOk = nCross = nHdg = nCurv = nDone = 0
                line = (f"{_u16(l)},{_u16(f)},{_u16(r)},{lidar.rev},"
                        f"{v['color']},{v['err']},{v['area']},{v['seq']},"
                        f"{_cone_u16(cl)},{_cone_u16(cr)},{_ang(yaw)},{pX},{pY},{uX},{uY},"
                        f"{v['s_color']},{sX},{sY},"
                        f"{nOk},{nCross},{nHdg},{nCurv},{nDone}\n")
                if write(line.encode("ascii")):
                    frames += 1
                link.update(line=line, t=now, f=f, l=l, r=r, cl=cl, cr=cr,
                            yaw=yaw, pxy=pxy, uxy=uxy, sxy=sxy,
                            s_name=NAMES[v["s_color"]])
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
                if NAV is not None:
                    print("[obstacle] " + NAV.status())
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
        ser.close()
        print("[obstacle] stopped")


if __name__ == "__main__":
    main()
