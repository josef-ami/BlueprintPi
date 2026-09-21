"""
Camera adapter - finds pillars and their bearings.

A pillar is an upright, solid, saturated block STANDING ON the white mat. The
colour mask alone cannot tell one from an orange floor line, a red shirt
behind the wall, or a green banner in the hall, so every blob must pass five
tests. This file is the merge of the two detectors this repo used to carry -
the obstacle round's A/S/F/C tests and the calibration dashboard's
linked-floor test - into one, because there is one robot and it should be
tuned once:

    A  aspect     h >= aspect_min * w        orange/red floor lines are flat
    S  solidity   area >= solidity_min * bbox   thin streaks, ragged noise
    F  on floor   the strip just under the blob is mostly white mat
    L  linked     that white connects to the mat in front of the car through
                  pixels that are not wall-dark. Corner lines and other
                  pillars do not break the link; the black walls do, which is
                  what rejects anything seen over or beyond them
    C  contrast   blob colourfulness minus the mat's under it

A blob whose base is at or below the bottom of the view is a pillar too close
to see its base: F, L and C are skipped for it, A and S still apply. Nothing
but a pillar can be that close.

TWO COLOUR SPACES
  HSV was the original. It separates the pillars on hue but gates on
  SATURATION, and a matte pillar under dim indoor light falls under the S
  threshold and simply vanishes - which is the usual reason green stops being
  detected while red still works (red has two hue bands and survives longer).

  Lab does the same job without that failure mode. In OpenCV's 8-bit Lab, a
  and b are centred on 128: a > 128 is red, a < 128 is green, and the white
  mat sits near (128, 128) whatever the lighting does to L. So the pillars
  separate on ONE channel, and brightness never removes the colour.

  Both are kept and USE_LAB picks between them, so a bad calibration is one
  checkbox away from the behaviour you had before. The dashboard's Calibrate
  tab fits the Lab ranges by clicking and turns USE_LAB on when you save.

BEARINGS - and which model is right is a MEASUREMENT, not a preference
  USE_INTRINSICS on   the calibrated fisheye K/D. fx = 383 px/rad puts the
                      frame edge at 48 deg and the polynomial folds back at
                      62, so nothing beyond that can be described at all.
  USE_INTRINSICS off  ideal equidistant lens over HFOV_DEG: r = fx * theta
                      with fx = (W/2) / radians(HFOV/2). Covers the full
                      frame, assumes no distortion beyond equidistance.

  config.json disagrees with itself here: camera_intrinsics.K has fx = 383
  px/rad, which on a 640 px frame is +/-48 deg, while HFOV_DEG says 160
  (+/-80) - a factor of 1.67. Measure a pillar at a known angle before
  trusting either, because the answer decides whether the car can see a
  pillar just past a corner at all.

Sign convention, everywhere: + = LEFT of forward, matching the LiDAR frame
(index = degree, increasing anticlockwise, y positive to the left).
"""

import json
import math
import os
import threading
import time

import cv2
import numpy as np

from worldstate import CameraResult, Pillar, SharedState

FRAME_W = 640
FRAME_H = 480
PROC_SIZE = (320, 240)       # detection resolution; boxes scale back up

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config.json")

# reject codes, in the order they are tested
REJECT_ASPECT = "A"
REJECT_SOLIDITY = "S"
REJECT_FLOOR = "F"
REJECT_LINKED = "L"
REJECT_CONTRAST = "C"

REJECT_TEXT = {
    REJECT_ASPECT: "too flat (aspect)",
    REJECT_SOLIDITY: "ragged (solidity)",
    REJECT_FLOOR: "not standing on the mat",
    REJECT_LINKED: "mat not linked to the floor in front (wall between?)",
    REJECT_CONTRAST: "too little colour vs the mat",
}

FLOOR_STRIP_GAP_PX = 2       # skip the blob's own soft bottom edge
FLOOR_SEED_ROWS = 8          # "floor in front of the car" = white in these rows
FLOOR_MIN_STRIP_ROWS = 3     # fewer visible strip rows = base below the view


# --------------------------------------------------------------------------
# config (structured blocks only - the tunables live in params.py)
# --------------------------------------------------------------------------

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)


def save_config(cfg, path=CONFIG_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# camera intrinsics (fisheye)
# --------------------------------------------------------------------------

_INTRINSICS = None   # {"K": ndarray(3,3), "D": ndarray(4,1)} or None


def load_intrinsics(cfg=None):
    """Parse camera_intrinsics into numpy K/D once. Returns
    {"K","D","image_size"} or None if not calibrated yet."""
    global _INTRINSICS
    if _INTRINSICS is not None:
        return _INTRINSICS
    if cfg is None:
        try:
            cfg = load_config()
        except Exception:
            return None
    intr = cfg.get("camera_intrinsics")
    if not intr or intr.get("model") != "fisheye":
        return None
    _INTRINSICS = {
        "K": np.array(intr["K"], dtype=np.float64),
        "D": np.array(intr["D"], dtype=np.float64).reshape(-1, 1),
        "image_size": tuple(intr.get("image_size", (FRAME_W, FRAME_H))),
    }
    return _INTRINSICS


def principal_point():
    intr = load_intrinsics()
    if intr is not None:
        K = intr["K"]
        return float(K[0, 2]), float(K[1, 2])
    return FRAME_W / 2.0, FRAME_H / 2.0


def px_to_bearing_fisheye(cx, cy, offset_deg=0.0):
    """Calibrated fisheye model. Undistorts this one point through K/D and
    reads the bearing off the undistorted pinhole plane."""
    intr = load_intrinsics()
    if intr is None:
        return None
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float64)
    und = cv2.fisheye.undistortPoints(pt, intr["K"], intr["D"])
    return -math.degrees(math.atan(float(und[0, 0, 0]))) + offset_deg


def px_to_bearing_equidistant(cx, cy, hfov_deg, offset_deg=0.0):
    """Ideal equidistant lens over hfov_deg. Covers the whole frame."""
    ppx, ppy = principal_point()
    fx = (FRAME_W / 2.0) / math.radians(hfov_deg / 2.0)
    dx, dy = cx - ppx, cy - ppy
    r = math.hypot(dx, dy)
    if r < 1e-9:
        return offset_deg
    th = r / fx
    x = math.sin(th) * dx / r
    z = math.cos(th)
    return -math.degrees(math.atan2(x, z)) + offset_deg


def px_to_bearing(cx, cy, p):
    """Blob centre in 640x480 pixels -> bearing in degrees, + = left.
    `p` is a PiParams (or any mapping with the camera keys)."""
    offset = p["CAMERA_OFFSET_DEG"]
    if p["USE_INTRINSICS"]:
        b = px_to_bearing_fisheye(cx, cy, offset)
        if b is not None:
            return b
    return px_to_bearing_equidistant(cx, cy, p["HFOV_DEG"], offset)


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------

def colour_spaces(rgb):
    """(hsv, lab) for one detection-sized frame. Both are cheap, and the
    overlay and the calibrator want whichever one you are not classifying in."""
    return (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))


def chroma(lab):
    """Distance from neutral grey in the a/b plane - the Lab analogue of HSV
    saturation, and what the contrast test compares."""
    ab = lab[:, :, 1:].astype(np.int16) - 128
    return np.hypot(ab[:, :, 0], ab[:, :, 1]).astype(np.float32)


def build_mask(hsv_img, ranges):
    """Union of HSV ranges, opened and closed. Red needs two (hue wraps)."""
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv_img, np.array(lo, dtype=np.uint8),
                        np.array(hi, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def colour_mask(hsv, lab, name, hsv_cfg, lab_cfg, use_lab):
    """Binary mask for one pillar colour, in whichever space is selected."""
    if use_lab:
        lo, hi = lab_cfg[name]
        m = cv2.inRange(lab, np.array(lo, np.uint8), np.array(hi, np.uint8))
        k = np.ones((5, 5), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return build_mask(hsv, hsv_cfg[name])


def floor_mask(hsv, lab, pf, use_lab, lab_floor):
    """White mat. HSV: unsaturated and bright. Lab: bright and near-neutral."""
    if use_lab:
        L = lab[:, :, 0]
        a = lab[:, :, 1].astype(np.int16)
        b = lab[:, :, 2].astype(np.int16)
        tol = lab_floor["FLOOR_AB_TOL"]
        m = ((L >= lab_floor["FLOOR_L_MIN"]) & (np.abs(a - 128) <= tol)
             & (np.abs(b - 128) <= tol)).astype(np.uint8) * 255
    else:
        m = cv2.inRange(hsv, (0, 0, pf["floor_v_min"]),
                        (180, pf["floor_s_max"], 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


# --------------------------------------------------------------------------
# the linked-floor test
# --------------------------------------------------------------------------

class FloorContext:
    """One frame's floor analysis, shared by every blob in that frame.

    The mat under a blob is only evidence that the blob stands on the floor
    THIS CAR IS ON if that white is connected to the white in front of the
    car through pixels that are not wall-dark. A red shirt beyond the wall
    has white mat under it too - the mat of the next straight - but the black
    wall between breaks the link.
    """

    def __init__(self, floor, value_channel, pf):
        self.pf = pf
        self.floor = floor
        self.h, self.w = floor.shape[:2]
        self.bottom = max(1, self.h - int(pf["ignore_bottom_px"]))
        self.labels = None      # connected regions of not-wall-dark pixels
        self.seeds = None       # the region(s) holding the floor in front
        if pf["linked_test"]:
            notdark = (value_channel > int(pf["dark_v_max"])).astype(np.uint8)
            _, self.labels = cv2.connectedComponents(notdark, connectivity=4)
            r0 = max(0, self.bottom - FLOOR_SEED_ROWS)
            white = self.floor[r0:self.bottom] > 0
            seeds = np.unique(self.labels[r0:self.bottom][white])
            seeds = seeds[seeds != 0]
            # No white at all in front (nose to a wall, say): the link cannot
            # be judged, so only the contact test applies this frame.
            self.seeds = seeds if seeds.size else None

    def strip(self, x, y, w, h):
        """(x0, y0, x1, y1) of the contact strip under a bounding box."""
        y0 = y + h + FLOOR_STRIP_GAP_PX
        y1 = min(y0 + int(self.pf["strip_px"]), self.bottom)
        m = int(w * 0.2)                       # central 60 % of the width
        x0, x1 = x + m, x + w - m
        if x1 <= x0:
            x0, x1 = x, x + w
        return x0, y0, x1, y1

    def base_below_view(self, y, h):
        """True when the blob's base is too low to judge: a pillar that close
        is a pillar, so the floor tests are skipped."""
        y0 = y + h + FLOOR_STRIP_GAP_PX
        if y + h >= self.bottom - int(self.pf["bottom_margin_px"]):
            return True
        return min(y0 + int(self.pf["strip_px"]), self.bottom) - y0 \
            < FLOOR_MIN_STRIP_ROWS

    def check(self, x, y, w, h):
        """(reject_code_or_None, white_frac). Runs F then L."""
        x0, y0, x1, y1 = self.strip(x, y, w, h)
        white = self.floor[y0:y1, x0:x1] > 0
        frac = float(white.mean()) if white.size else 0.0
        if frac < float(self.pf["floor_below_min"]):
            return REJECT_FLOOR, frac
        if self.labels is not None and self.seeds is not None:
            if not np.isin(self.labels[y0:y1, x0:x1][white], self.seeds).any():
                return REJECT_LINKED, frac
        return None, frac

    def linked_floor(self):
        """White floor that is linked to the floor in front, for the viewers."""
        if self.labels is None or self.seeds is None:
            return self.floor
        return np.where(np.isin(self.labels, self.seeds) & (self.floor > 0),
                        255, 0).astype(np.uint8)


# --------------------------------------------------------------------------
# the detector
# --------------------------------------------------------------------------

class Detection:
    """What one frame's detection pass produced.

    best      the largest ACCEPTED blob as a Pillar, or None
    second    the NEXT-largest accepted blob, or None
    pillars   every accepted blob, largest first
    rejected  [(box, colour, code)] for the overlay
    masks     {"RED", "GREEN", "FLOOR"} at detection resolution
    ctx       the FloorContext, for the floor view

    `second` exists because one pillar is not enough on the run up to a
    corner. The planner steers around the nearest pillar, but the corner has
    to know which side to come out on, and that is decided by the FIRST
    pillar of the NEXT straight - which, while the car is still on this one,
    is almost always the second-largest blob in frame. Reporting only the
    largest made that pillar invisible exactly when it mattered.

    Further away is not on its own a reason to believe it is in the next
    straight: a straight with two pillars on it also has a nearer and a
    further one. The planner applies the geometric test - see
    LanePlanner.track_secondary - and this class just carries the candidate.
    """

    __slots__ = ("best", "second", "pillars", "rejected", "masks", "ctx",
                 "space")

    def __init__(self, best, second, pillars, rejected, masks, ctx, space):
        self.best, self.second = best, second
        self.pillars, self.rejected = pillars, rejected
        self.masks, self.ctx, self.space = masks, ctx, space


class PillarDetector:
    """Holds one snapshot of the tunables and runs the pass.

    Rebuilt (cheaply) per frame from PiParams, so the Tune tab is live: there
    is no restart, no re-open of the camera, and the vision thread never
    blocks on the web side.
    """

    def __init__(self, p):
        self.p = p
        self.use_lab = bool(p["USE_LAB"])
        self.min_area = p["MIN_AREA_PROC"]
        self.pf = p.pillar_filter() if hasattr(p, "pillar_filter") else p
        self.hsv_cfg = p.hsv_config() if hasattr(p, "hsv_config") else {}
        self.lab_cfg = p.lab_config() if hasattr(p, "lab_config") else {}
        self.lab_floor = {"FLOOR_L_MIN": p["FLOOR_L_MIN"],
                          "FLOOR_AB_TOL": p["FLOOR_AB_TOL"]}
        self.chroma_min = p["LAB_CHROMA_MIN"]

    # ---- the per-blob tests ----

    def check(self, cnt, area, hsv, lab, ctx, ch=None):
        """None if the blob is a pillar, else its reject code. Order is
        cheapest-first, and the floor tests are skipped for a blob whose base
        is out of view."""
        x, y, w, h = cv2.boundingRect(cnt)
        pf = self.pf
        if h < pf["aspect_min"] * w:
            return REJECT_ASPECT, None
        if area < pf["solidity_min"] * w * h:
            return REJECT_SOLIDITY, None
        if ctx.base_below_view(y, h):
            return None, None                  # too close to see its base

        code, frac = ctx.check(x, y, w, h)
        if code is not None:
            return code, frac

        # C: colourfulness of the blob against the mat under it
        x0, y0, x1, y1 = ctx.strip(x, y, w, h)
        blob = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob, [cnt - (x, y)], -1, 255, -1)
        strip = (ctx.floor[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
        if strip.size == 0:
            return None, frac
        if self.use_lab:
            c_in = cv2.mean(ch[y:y + h, x:x + w], mask=blob)[0]
            c_floor = cv2.mean(ch[y0:y1, x0:x1], mask=strip)[0]
            if c_in - c_floor < self.chroma_min:
                return REJECT_CONTRAST, frac
        else:
            s_in = cv2.mean(hsv[y:y + h, x:x + w, 1], mask=blob)[0]
            s_floor = cv2.mean(hsv[y0:y1, x0:x1, 1], mask=strip)[0]
            if s_in - s_floor < pf["contrast_s_min"]:
                return REJECT_CONTRAST, frac
        return None, frac

    # ---- the pass ----

    def detect(self, small_rgb, want_masks=False):
        """One detection-resolution frame -> Detection. Coordinates are in
        detection pixels; scale by FRAME_W/PROC_SIZE[0] for display."""
        hsv, lab = colour_spaces(small_rgb)
        floor = floor_mask(hsv, lab, self.pf, self.use_lab, self.lab_floor)
        value = lab[:, :, 0] if self.use_lab else hsv[:, :, 2]
        ctx = FloorContext(floor, value, self.pf)
        ch = chroma(lab) if self.use_lab else None

        masks = {"FLOOR": floor} if want_masks else {}
        accepted, rejected = [], []
        sx = FRAME_W / float(small_rgb.shape[1])
        sy = FRAME_H / float(small_rgb.shape[0])

        for name in ("RED", "GREEN"):
            mask = colour_mask(hsv, lab, name, self.hsv_cfg, self.lab_cfg,
                               self.use_lab)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            kept = []
            for c in cnts:
                a = cv2.contourArea(c)
                if a < self.min_area:
                    continue
                code, frac = self.check(c, a, hsv, lab, ctx, ch)
                x, y, w, h = cv2.boundingRect(c)
                box = (int(x * sx), int(y * sy),
                       int((x + w) * sx), int((y + h) * sy))
                if code is not None:
                    rejected.append((box, name, code, frac))
                    continue
                kept.append(c)
                cx_full = (x + w / 2.0) * sx
                cy_full = (y + h / 2.0) * sy
                accepted.append(Pillar(
                    colour=name,
                    bearing_deg=px_to_bearing(cx_full, cy_full, self.p),
                    err_px=int(cx_full) - FRAME_W // 2,
                    area=int(a), box=box, white_frac=frac))
            if want_masks:
                pillar = np.zeros_like(mask)
                cv2.drawContours(pillar, kept, -1, 255, cv2.FILLED)
                masks[name] = cv2.bitwise_and(pillar, mask)

        accepted.sort(key=lambda b: -b.area)
        return Detection(accepted[0] if accepted else None,
                         accepted[1] if len(accepted) > 1 else None,
                         accepted, rejected, masks, ctx,
                         "Lab" if self.use_lab else "HSV")


# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------

def open_camera(width=FRAME_W, height=FRAME_H, hflip=True, vflip=True):
    from picamera2 import Picamera2
    from libcamera import Transform
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
        transform=Transform(hflip=int(hflip), vflip=int(vflip)),
    ))
    cam.start()
    time.sleep(0.5)
    return cam


def grab_rgb(cam, swap_rb=False):
    frame = cam.capture_array()
    if swap_rb:
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame)


# --------------------------------------------------------------------------
# thread
# --------------------------------------------------------------------------

class VisionThread(threading.Thread):
    """Owns the camera, runs the detector, publishes a CameraResult.

    Reads the tunables fresh every frame, so the Tune tab is live. Keeps the
    full-size frame only while someone is watching the stream, because
    copying 640x480x3 per frame for nobody is the most expensive thing this
    thread could do.

    frames_from  optional callable returning (raw_frame, seq) instead of
                 opening the camera - the dashboard already owns one camera
                 and hands its frames over rather than fighting for the
                 device.
    """

    def __init__(self, params, shared: SharedState, frames_from=None,
                 name="Vision"):
        super().__init__(name=name, daemon=True)
        self.p = params
        self.shared = shared
        self.frames_from = frames_from
        self._halt = threading.Event()   # NOT _stop: Thread owns that name
        self._lock = threading.Lock()
        self.viewers = 0
        self.latest_frame = None         # RGB, full size, only while watched
        self.latest = None               # the newest Detection + frame
        self.error = None
        self.fps = 0.0
        self.seq = 0

    def stop(self):
        self._halt.set()

    def get(self):
        with self._lock:
            return self.latest

    def frame(self):
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def _publish(self, det, t):
        self.shared.set_camera(CameraResult(
            timestamp=t, pillars=det.pillars, best=det.best,
            second=det.second, ok=True, seq=self.seq, space=det.space))

    def run(self):
        cam = None
        if self.frames_from is None:
            try:
                cam = open_camera(FRAME_W, FRAME_H)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                print(f"[vision] camera failed: {self.error} - no pillars")
                return
        load_intrinsics()
        n, t_fps, last_seq = 0, time.time(), -1
        try:
            while not self._halt.is_set():
                if self.frames_from is None:
                    frame = grab_rgb(cam, bool(self.p["SWAP_RB"]))
                else:
                    raw, seq = self.frames_from()
                    if raw is None or seq == last_seq:
                        time.sleep(0.005)
                        continue
                    last_seq = seq
                    frame = (np.ascontiguousarray(raw[:, :, ::-1])
                             if self.p["SWAP_RB"] else raw)

                small = cv2.resize(frame, PROC_SIZE, interpolation=cv2.INTER_AREA)
                watched = self.viewers > 0
                try:
                    det = PillarDetector(self.p).detect(small, want_masks=watched)
                except Exception as e:
                    self.error = f"{type(e).__name__}: {e}"
                    time.sleep(0.1)
                    continue

                self.seq = (self.seq + 1) & 0xFFFFFFFF
                t = time.time()
                self._publish(det, t)
                with self._lock:
                    self.latest = {"det": det, "seq": self.seq, "t": t,
                                   "small": small if watched else None}
                    self.latest_frame = frame if watched else None

                n += 1
                if time.time() - t_fps >= 1.0:
                    self.fps = n / (time.time() - t_fps)
                    n, t_fps = 0, time.time()
        finally:
            if cam is not None:
                try:
                    cam.stop()
                except Exception:
                    pass
            print("[vision] stopped")
