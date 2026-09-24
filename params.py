#!/usr/bin/env python3
"""
params.py - the single place that knows what is tunable, on the Pi and on the
STM32, and the only thing that writes tuning.json.

There are two halves and they are deliberately different:

  PiParams    values this process reads directly every frame (HSV ranges, the
              pillar filter, the cone fit, the LiDAR candidate search). Setting
              one takes effect on the next frame - nothing is restarted, and
              the camera and LiDAR threads are never touched.

  StmParams   a MIRROR of the firmware's table. The firmware owns no storage:
              it boots with its compiled-in defaults and announces a new boot
              id, and this side pushes the whole saved set over the serial link
              it is already using for sensor frames. A mid-race STM32 reset
              therefore re-tunes itself within a few hundred ms instead of
              silently running defaults.

Nothing here blocks. Pushes are queued and drained a few per loop by whoever
owns the serial port, so a tuning change can never stall the 50 Hz frame feed.

Readers are lock-free: values live in one dict that is REPLACED, never mutated,
so a reader either sees the whole old set or the whole new one.

PERSISTENCE - two files next to this one, both survive a power cut
  vision_cal.json  the camera calibration: the Lab colour ranges, the Lab mat
                   test, USE_LAB and AREA_K (CAL_KEYS). Written by
                   calibrate_vision.py when you press FIT or SAVE, and by the
                   Tune tab when you edit one of those values. "Revert Pi" never
                   touches it - calibrate once and it stays.
  tuning.json      every other Pi value, plus the STM32 table.
Every change is saved automatically about a second after the last edit
(AutoSave), so nothing is lost if the Pi is switched off without pressing Save.
Writes are atomic (temp file, fsync, rename, fsync the directory), so a power
cut mid-write leaves the previous file intact, never a truncated one.
"""

import json
import os
import queue
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
TUNING_PATH = os.path.join(_HERE, "tuning.json")
CAL_PATH = os.path.join(_HERE, "vision_cal.json")


# --------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------

class Spec:
    """One tunable: name, default, range, how to show it."""

    __slots__ = ("name", "default", "lo", "hi", "kind", "group", "help")

    def __init__(self, name, default, lo, hi, group, kind="f", help=""):
        self.name, self.default = name, default
        self.lo, self.hi = lo, hi
        self.kind = kind              # f float | i int | b bool
        self.group, self.help = group, help

    def coerce(self, v):
        """Value -> correct type, clamped. Raises ValueError on junk."""
        if self.kind == "b":
            if isinstance(v, str):
                v = v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)
        v = float(v)
        if v != v:                    # NaN
            raise ValueError("not a number")
        v = max(self.lo, min(self.hi, v))
        return int(round(v)) if self.kind == "i" else v

    def as_dict(self, value):
        return {"name": self.name, "value": value, "default": self.default,
                "lo": self.lo, "hi": self.hi, "kind": self.kind,
                "group": self.group, "help": self.help}


# --------------------------------------------------------------------------
# Pi-side tunables
# --------------------------------------------------------------------------
#
# Groups are only how the page lays them out.
#   camera   frame handling and the camera model
#   hsv      colour ranges (the thing that actually needs tuning on the day)
#   filter   the "is this blob really a pillar" tests
#   cone     the 45 deg wall fits that give lane offset and wall angle
#   locate   turning a camera bearing + LiDAR range into a pillar position
#   cand     LiDAR-only objects whose colour is not known yet (LazyGo edges)
#   link     serial framing
#   yolo     the trained pillar detector (sensors/yolo_detector.py)
#   record   saving camera frames to take/ for training a detector

PI_SPECS = [
    # ---- camera ----
    Spec("MIN_AREA_PROC", 250, 20, 5000, "camera", "i",
         "smallest blob (in 320x240 px) that counts as a pillar"),
    Spec("VISION_STALE_S", 0.2, 0.05, 2.0, "camera", "f",
         "no camera frame for this long -> colour is sent as none"),
    Spec("JPEG_QUALITY", 50, 10, 95, "camera", "i", "preview stream quality"),
    Spec("CAMERA_FWD_MM", 0.0, -200, 400, "camera", "f",
         "camera ahead of the LiDAR; yours sits above it, so 0"),
    Spec("CAMERA_OFFSET_DEG", 5.0, -30, 30, "camera", "f",
         "camera-to-LiDAR rotational alignment"),
    Spec("HFOV_DEG", 160.0, 40, 200, "camera", "f",
         "lens horizontal field of view, used by the equidistant model"),
    Spec("USE_INTRINSICS", True, 0, 1, "camera", "b",
         "bearing from the calibrated fisheye K/D (off = equidistant HFOV_DEG). "
         "Your K says +/-48 deg; hfov_deg says +/-80. Measure before trusting either"),
    Spec("SWAP_RB", True, 0, 1, "camera", "b", "swap red and blue channels"),
    Spec("USE_LAB", True, 0, 1, "lab", "b",
         "classify pillars in CIE Lab instead of HSV. Lab separates red from green on "
         "the a channel without needing saturation, so a matte pillar under dim light "
         "still passes. On by default; off falls back to the HSV ranges"),

    # ---- Lab ranges (OpenCV 8-bit: L 0-255, a/b 0-255 with 128 = neutral) ----
    # Defaults are built from the rulebook colours, not placeholders:
    #   red   RGB (238, 39, 55) -> Lab (132, 200, 171); in shadow ~ (65, 170, 151)
    #   green RGB (68, 214, 44) -> Lab (193, 61, 194); in shadow ~ (101, 89, 167)
    #   white mat -> (158..237, 128, 128)
    # a separates red / green / mat; b >= 135 drops the magenta parking walls
    # (b 67) and blue lines (b 28). Orange lines (a 183, b 199) do pass the red
    # range - the aspect test (flat) removes them. calibrate_vision.py replaces
    # all of these with values fitted on your mat and camera.
    Spec("RED_L_LO",   20, 0, 255, "lab", "i", "red pillar: lightness"),
    Spec("RED_L_HI",  255, 0, 255, "lab", "i", ""),
    Spec("RED_A_LO",  150, 0, 255, "lab", "i", "a: >128 is red, <128 is green"),
    Spec("RED_A_HI",  255, 0, 255, "lab", "i", ""),
    Spec("RED_B_LO",  135, 0, 255, "lab", "i", "b: >128 is yellow, <128 is blue"),
    Spec("RED_B_HI",  255, 0, 255, "lab", "i", ""),
    Spec("GREEN_L_LO",  20, 0, 255, "lab", "i", "green pillar: lightness"),
    Spec("GREEN_L_HI", 255, 0, 255, "lab", "i", ""),
    Spec("GREEN_A_LO",   0, 0, 255, "lab", "i", ""),
    Spec("GREEN_A_HI", 110, 0, 255, "lab", "i", ""),
    Spec("GREEN_B_LO", 135, 0, 255, "lab", "i", ""),
    Spec("GREEN_B_HI", 255, 0, 255, "lab", "i", ""),
    Spec("FLOOR_L_MIN", 120, 0, 255, "lab", "i",
         "white mat in Lab: at least this bright ..."),
    Spec("FLOOR_AB_TOL", 14, 1, 80, "lab", "i",
         "... and within this of neutral (128) on both a and b"),
    Spec("LAB_CHROMA_MIN", 20, 0, 128, "lab", "i",
         "pillar chroma minus mat chroma under it; the Lab version of contrast_s_min"),

    # ---- HSV ----
    Spec("RED1_H_LO",   0, 0, 180, "hsv", "i", "red range 1"),
    Spec("RED1_S_LO", 120, 0, 255, "hsv", "i", ""),
    Spec("RED1_V_LO",  70, 0, 255, "hsv", "i", ""),
    Spec("RED1_H_HI",  10, 0, 180, "hsv", "i", ""),
    Spec("RED1_S_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("RED1_V_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("RED2_H_LO", 170, 0, 180, "hsv", "i", "red range 2 (hue wraps)"),
    Spec("RED2_S_LO", 120, 0, 255, "hsv", "i", ""),
    Spec("RED2_V_LO",  70, 0, 255, "hsv", "i", ""),
    Spec("RED2_H_HI", 180, 0, 180, "hsv", "i", ""),
    Spec("RED2_S_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("RED2_V_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("GREEN_H_LO", 40, 0, 180, "hsv", "i", "green range"),
    Spec("GREEN_S_LO", 80, 0, 255, "hsv", "i", ""),
    Spec("GREEN_V_LO", 60, 0, 255, "hsv", "i", ""),
    Spec("GREEN_H_HI", 85, 0, 180, "hsv", "i", ""),
    Spec("GREEN_S_HI", 255, 0, 255, "hsv", "i", ""),
    Spec("GREEN_V_HI", 255, 0, 255, "hsv", "i", ""),

    # ---- pillar filter ----
    Spec("floor_s_max", 60, 0, 255, "filter", "i", "mat: saturation at or below this"),
    Spec("floor_v_min", 120, 0, 255, "filter", "i", "... and value at or above this"),
    Spec("strip_px", 6, 1, 40, "filter", "i", "rows checked just under a blob"),
    Spec("floor_below_min", 0.45, 0.0, 1.0, "filter", "f",
         "fraction of that strip that must be mat"),
    Spec("aspect_min", 0.8, 0.0, 5.0, "filter", "f", "height / width"),
    Spec("solidity_min", 0.5, 0.0, 1.0, "filter", "f", "contour area / bbox area"),
    Spec("contrast_s_min", 50, 0, 255, "filter", "i",
         "blob saturation minus mat saturation under it"),
    Spec("bottom_margin_px", 3, 0, 40, "filter", "i",
         "this close to the ROI bottom = base out of view, skip the floor tests"),
    # ---- LazyGo additions (detection_cam.py / helper/util.py) ----
    Spec("roi_top_px", 60, 0, 479, "filter", "i",
         "search band top row, 640x480 units (LazyGo 60): above it is the hall"),
    Spec("roi_bottom_px", 440, 1, 480, "filter", "i",
         "search band bottom row, 640x480 units (LazyGo 440): below it is the car"),
    Spec("mask_blur_px", 3, 0, 15, "filter", "i",
         "Gaussian blur on the colour mask then threshold 127 (0 = off)"),
    Spec("min_box_h_px", 20, 0, 200, "filter", "i",
         "a sign's box must be at least this tall, 640x480 units (LazyGo 30; reject H)"),

    # ---- cone wall fit ----
    Spec("CONE_DEG", 45, 10, 120, "cone", "i", "width of each side cone"),
    Spec("CONE_MAX_RANGE_MM", 1500, 300, 4000, "cone", "i", ""),
    Spec("CONE_MIN_RANGE_MM", 60, 10, 500, "cone", "i", ""),
    Spec("CONE_INLIER_MM", 25, 3, 200, "cone", "i", "RANSAC inlier band"),
    Spec("CONE_MIN_INLIERS", 8, 3, 60, "cone", "i", ""),
    Spec("CONE_MIN_SPAN_MM", 150, 20, 800, "cone", "i",
         "wall length the fit must cover; a 50 mm pillar face cannot"),
    Spec("CONE_AGREE_DEG", 6.0, 0.5, 45.0, "cone", "f",
         "left/right yaw must agree this well to be averaged"),

    # ---- pillar location ----
    Spec("RAY_WINDOW_DEG", 8.0, 1.0, 30.0, "locate", "f",
         "LiDAR returns this close to the camera ray are candidates"),
    Spec("PILLAR_MAX_MM", 2000.0, 300, 4000, "locate", "f", ""),
    Spec("AREA_K", 14000.0, 2000, 60000, "locate", "f",
         "distance ~ AREA_K / sqrt(area), the fallback when no LiDAR return agrees"),
    Spec("FACE_TO_CENTRE_MM", 25.0, 0, 100, "locate", "f",
         "half a pillar: LiDAR sees the face, the planner wants the centre"),

    # ---- LiDAR-only candidates ----
    Spec("CORRIDOR_MM", 1000.0, 300, 2000, "cand", "f", ""),
    Spec("CAND_MAX_MM", 1800.0, 300, 4000, "cand", "f", ""),
    Spec("CAND_WALL_MM", 70.0, 0, 400, "cand", "f", "keep this clear of a fitted wall"),
    Spec("CAND_EDGE_MM", 150.0, 50, 1500, "cand", "f",
         "edge: range step that starts / ends an object (LazyGo 250; 150 catches signs near a wall)"),
    Spec("CAND_MIN_WIDTH_MM", 25.0, 0, 200, "cand", "f",
         "narrower than this is noise"),
    Spec("CAND_MAX_WIDTH_MM", 120.0, 40, 600, "cand", "f",
         "wider than this is not a sign (50 mm, 71 diagonal)"),
    Spec("CAND_FOV_DEG", 90, 20, 180, "cand", "i",
         "search +/- this many degrees either side of forward"),

    # ---- link ----
    Spec("SEND_HZ", 50, 5, 200, "link", "i", "sensor frames per second"),
    Spec("CMD_REPEAT", 3, 1, 10, "link", "i", "copies of each START/STOP press"),
    Spec("BEARING_TOL_DEG", 2, 0, 15, "link", "i",
         "+/- degrees when picking the 0/90/270 beams"),
    Spec("STM_PUSH_PER_LOOP", 4, 1, 40, "link", "i",
         "tuning lines sent to the STM32 per loop; keep well under SEND_HZ budget"),

    # ---- YOLO pillar detector (models/pillars26, YOLO26n at 416) ----
    Spec("USE_YOLO", True, 0, 1, "yolo", "b",
         "find pillars with the trained YOLO model; off = the Lab/HSV colour masking. "
         "If the model cannot load, masking is used automatically"),
    Spec("YOLO_BACKEND", 0, 0, 3, "yolo", "i",
         "0 auto (ncnn, then openvino, then onnx), 1 ncnn, 2 openvino, 3 onnx"),
    Spec("YOLO_THREADS", 3, 1, 4, "yolo", "i",
         "CPU threads for inference; the LiDAR and serial loop need the rest"),
    Spec("YOLO_CONF", 0.5, 0.05, 0.95, "yolo", "f",
         "minimum confidence for a pillar; raise if phantom pillars appear, lower if far ones are missed"),
    Spec("YOLO_IOU", 0.5, 0.1, 0.9, "yolo", "f",
         "overlap above which two boxes of one colour count as the same pillar"),
    Spec("YOLO_MIN_BOX_H", 0, 0, 200, "yolo", "i",
         "ignore boxes shorter than this (640x480 px); 0 = keep every detection (reject code H)"),

    # ---- recording: training images for the YOLO detector ----
    Spec("RECORD_RUNS", False, 0, 1, "record", "b",
         "save camera frames to take/ from GO until FINISHED / STOP. Turn ON for "
         "practice rounds (training data), OFF for scored runs"),
    Spec("RECORD_HZ", 3.0, 0.2, 15.0, "record", "f",
         "frames saved per second (the camera runs ~30; 3 Hz is ~250 images per 3-lap run)"),
    Spec("RECORD_JPEG_Q", 92, 50, 100, "record", "i", "JPEG quality of the saved frames"),
    Spec("RECORD_MIN_CHANGE", 3.0, 0, 50, "record", "f",
         "skip a frame whose 32x24 grey thumbnail differs from the last saved one by less "
         "than this (mean grey levels): no piles of identical frames while the car stands. 0 = keep all"),
    Spec("RECORD_MIN_FREE_MB", 500, 50, 20000, "record", "i",
         "stop saving when the SD card has less than this many MB free"),
]


class PiParams:
    """Lock-free for readers: p['NAME']. Writers swap the whole dict."""

    def __init__(self, specs):
        self.specs = {s.name: s for s in specs}
        self.order = [s.name for s in specs]
        self._vals = {s.name: s.default for s in specs}

    def __getitem__(self, name):
        return self._vals[name]

    def get(self, name, default=None):
        return self._vals.get(name, default)

    def snapshot(self):
        return self._vals

    def set(self, name, value):
        """Returns the stored value. Raises KeyError / ValueError."""
        spec = self.specs[name]
        v = spec.coerce(value)
        new = dict(self._vals)
        new[name] = v
        self._vals = new            # atomic swap; readers never see a half-set
        return v

    def set_many(self, mapping):
        new = dict(self._vals)
        out = {}
        for name, value in mapping.items():
            if name not in self.specs:
                continue
            out[name] = new[name] = self.specs[name].coerce(value)
        self._vals = new
        return out

    def reset(self, keep=()):
        """Back to defaults, except the names in keep (the calibration)."""
        self._vals = {n: (self._vals[n] if n in keep else self.specs[n].default)
                      for n in self.order}

    def describe(self):
        return [self.specs[n].as_dict(self._vals[n]) for n in self.order]

    # ---- derived views the hot loop wants ready-made ----
    def hsv_config(self):
        p = self._vals
        return {
            "RED": [[[p["RED1_H_LO"], p["RED1_S_LO"], p["RED1_V_LO"]],
                     [p["RED1_H_HI"], p["RED1_S_HI"], p["RED1_V_HI"]]],
                    [[p["RED2_H_LO"], p["RED2_S_LO"], p["RED2_V_LO"]],
                     [p["RED2_H_HI"], p["RED2_S_HI"], p["RED2_V_HI"]]]],
            "GREEN": [[[p["GREEN_H_LO"], p["GREEN_S_LO"], p["GREEN_V_LO"]],
                       [p["GREEN_H_HI"], p["GREEN_S_HI"], p["GREEN_V_HI"]]]],
        }

    def lab_config(self):
        """{'RED': (lo, hi), 'GREEN': (lo, hi)} as (L, a, b) triples."""
        p = self._vals
        return {
            "RED": ((p["RED_L_LO"], p["RED_A_LO"], p["RED_B_LO"]),
                    (p["RED_L_HI"], p["RED_A_HI"], p["RED_B_HI"])),
            "GREEN": ((p["GREEN_L_LO"], p["GREEN_A_LO"], p["GREEN_B_LO"]),
                      (p["GREEN_L_HI"], p["GREEN_A_HI"], p["GREEN_B_HI"])),
        }

    def pillar_filter(self):
        p = self._vals
        return {k: p[k] for k in ("floor_s_max", "floor_v_min", "strip_px",
                                  "floor_below_min", "aspect_min", "solidity_min",
                                  "contrast_s_min", "bottom_margin_px",
                                  "roi_top_px", "roi_bottom_px", "mask_blur_px",
                                  "min_box_h_px")}


# --------------------------------------------------------------------------
# STM32 mirror
# --------------------------------------------------------------------------

class StmParams:
    """Mirror of the firmware's table, plus the push queue.

    The firmware is the authority on what EXISTS (names, ids, ranges, groups);
    this side is the authority on what the VALUES should be. `desired` is what
    tuning.json says; `live` is what the firmware last acknowledged. When they
    differ the name is queued for a push.
    """

    def __init__(self):
        self.table = {}          # name -> {id, kind, lo, hi, group}
        self.by_id = {}
        self.live = {}           # name -> value the firmware confirmed
        self.desired = {}        # name -> value we want
        self.version = None
        self.count = None
        self.boot = None
        self.synced = False      # a full dump has been received
        self.pending = queue.Queue()
        self._lock = threading.Lock()

    # ---- incoming firmware lines ----
    def on_line(self, line):
        """Handle one '!' line. Returns a note for the log, or None."""
        if line.startswith("!P "):
            try:
                _, pid, name, kind, val, lo, hi, group = line.split()
            except ValueError:
                return None
            with self._lock:
                self.table[name] = {"id": int(pid), "kind": int(kind),
                                    "lo": float(lo), "hi": float(hi),
                                    "group": int(group)}
                self.by_id[int(pid)] = name
                self.live[name] = float(val)
                if name not in self.desired:
                    self.desired[name] = float(val)   # adopt the firmware default
                if self.count and len(self.table) >= self.count:
                    self.synced = True
            return None

        if line.startswith("!p "):
            try:
                _, pid, val = line.split()
            except ValueError:
                return None
            name = self.by_id.get(int(pid))
            if name:
                with self._lock:
                    self.live[name] = float(val)
            return None

        if line.startswith("!V "):
            try:
                _, ver, count, boot = line.split()
            except ValueError:
                return None
            new_boot = (self.boot is not None and boot != self.boot)
            self.version, self.count, self.boot = ver, int(count), boot
            if new_boot or not self.table:
                # The firmware restarted (or we have never seen it). Its table
                # is back at compiled-in defaults, so ask for it and re-push.
                with self._lock:
                    self.synced = False
                    self.table.clear(); self.by_id.clear(); self.live.clear()
                self.request_dump()
                return f"[pi] STM32 boot {boot} - re-reading and re-pushing tuning"
            return None

        if line.startswith("!E"):
            return "[pi] STM32 rejected a tuning line: " + line
        return None

    # ---- outgoing ----
    def request_dump(self):
        self.pending.put("?P")

    def queue_all(self):
        """Push every desired value that the firmware does not already have."""
        with self._lock:
            names = [n for n in self.table if n in self.desired]
        for n in names:
            self._queue_if_stale(n)

    def _queue_if_stale(self, name):
        want = self.desired.get(name)
        have = self.live.get(name)
        if want is None:
            return
        if have is None or abs(float(have) - float(want)) > 1e-6:
            self.pending.put(f"N {name} {want:.6g}")

    def set(self, name, value):
        """Returns the coerced value. Raises KeyError / ValueError."""
        meta = self.table.get(name)
        if meta is None:
            raise KeyError(name)
        v = float(value)
        if v != v:
            raise ValueError("not a number")
        v = max(meta["lo"], min(meta["hi"], v))
        if meta["kind"] in (1, 2, 3, 4, 5):      # int-ish / bool
            v = float(round(v))
        with self._lock:
            self.desired[name] = v
        self.pending.put(f"N {name} {v:.6g}")
        return v

    def reset(self):
        """Back to the firmware's own defaults: forget desired, re-read."""
        with self._lock:
            self.desired = dict(self.live)
        self.request_dump()

    def drain(self, n):
        """Up to n queued lines, for the serial owner to write. Never blocks."""
        out = []
        for _ in range(n):
            try:
                out.append(self.pending.get_nowait())
            except queue.Empty:
                break
        return out

    def describe(self):
        with self._lock:
            rows = []
            for name, meta in self.table.items():
                rows.append({"name": name, "value": self.desired.get(name, meta and 0),
                             "live": self.live.get(name), "lo": meta["lo"],
                             "hi": meta["hi"],
                             "kind": "b" if meta["kind"] == 5 else
                                     ("f" if meta["kind"] == 0 else "i"),
                             "group": meta["group"], "id": meta["id"]})
        rows.sort(key=lambda r: r["id"])
        return rows


STM_GROUPS = ["drive", "colour trigger & turn", "planner", "passing", "levelling",
              "reverse & re-plan", "corner exit", "safety", "link"]


# --------------------------------------------------------------------------
# persistence - two files, atomic, auto-saved
# --------------------------------------------------------------------------

# the camera calibration: what calibrate_vision.py fits, kept in its own file
CAL_KEYS = frozenset([s.name for s in PI_SPECS if s.group == "lab"] + ["AREA_K"])


def _atomic_write(path, data):
    """Temp file, fsync, rename, fsync the directory: after a power cut the
    file is either the old one or the new one, never empty or half-written."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass                       # not supported on this filesystem; rename still done


def _read(path):
    try:
        with open(path) as f:
            return json.load(f), None
    except (OSError, ValueError) as e:
        return None, type(e).__name__


def calibrated():
    """True once a vision_cal.json exists."""
    return os.path.exists(CAL_PATH)


# Firmware defaults that CHANGED (name -> the old default). tuning.json keeps
# every STM32 value the page has ever seen, so without this a new firmware
# default would be overwritten by the old one on the first push. A saved value
# still equal to the old default is dropped (the firmware's new default wins);
# anything you tuned to another value is kept.
STM_OLD_DEFAULTS = {
    # firmware v9 (param table v8)
    "PASS_MARGIN_MM": 80.0,        # -> 150: more air to a sign
    "PASS_HOLD_MM": 250.0,         # -> 30
    "BACKOFF_MAX_MM": 250.0,       # -> 300
    "BACKOFF_MARGIN_MM": 60.0,     # -> 30, now a simulated clearance
    # firmware v9.2 (measured on the car)
    "CAR_HALF_LEN_MM": 85.0,       # -> 60
    "REACH_SPEED_MMPS": 400.0,     # -> 200
}


def _drop_old_defaults(saved_stm):
    out, dropped = {}, []
    for k, v in saved_stm.items():
        old = STM_OLD_DEFAULTS.get(k)
        if old is not None and abs(float(v) - old) < 1e-6:
            dropped.append(k)
            continue
        out[k] = float(v)
    return out, dropped


def load(pi: PiParams, stm: StmParams, path=TUNING_PATH, cal_path=CAL_PATH):
    """tuning.json first, then vision_cal.json on top (calibration always wins).
    An older tuning.json that holds a real Lab fit (USE_LAB on) is honoured until
    the first calibration file is written. Returns a note for the log."""
    notes = []
    saved, err = _read(path)
    if saved is None:
        notes.append(f"no saved tuning ({err})")
    else:
        vals = dict(saved.get("pi", {}))
        if not vals.get("USE_LAB", False):
            # an old tuning.json written before any Lab calibration: its Lab
            # numbers are the old wide-open placeholders, not a fit - ignore them
            # (and its USE_LAB = off). AREA_K is kept; it is measured separately.
            vals = {k: v for k, v in vals.items() if k not in CAL_KEYS or k == "AREA_K"}
        n_pi = len(pi.set_many(vals))
        stm_vals, dropped = _drop_old_defaults(saved.get("stm32", {}))
        stm.desired.update(stm_vals)
        notes.append(f"tuning: {n_pi} Pi, {len(stm_vals)} STM32")
        if dropped:
            notes.append("new firmware defaults for " + ", ".join(sorted(dropped)))
    cal, err = _read(cal_path)
    if cal is None:
        notes.append("NO vision_cal.json - Lab ranges are the rulebook defaults, "
                     "run calibrate_vision.py once")
    else:
        vals = cal.get("pi", cal)
        n = len(pi.set_many({k: v for k, v in vals.items() if k in CAL_KEYS}))
        notes.append(f"calibration: {n} values from {os.path.basename(cal_path)}")
    return "[pi] " + "; ".join(notes)


def save_calibration(pi: PiParams, cal_path=CAL_PATH):
    snap = pi.snapshot()
    _atomic_write(cal_path, {"pi": {k: snap[k] for k in sorted(CAL_KEYS)},
                             "saved": time.strftime("%Y-%m-%d %H:%M:%S")})
    return f"[pi] saved calibration to {os.path.basename(cal_path)}"


def save(pi: PiParams, stm: StmParams, path=TUNING_PATH, cal_path=CAL_PATH):
    """Both files: calibration to vision_cal.json, the rest to tuning.json."""
    snap = pi.snapshot()
    _atomic_write(path, {"pi": {k: v for k, v in snap.items() if k not in CAL_KEYS},
                         "stm32": dict(stm.desired)})
    save_calibration(pi, cal_path)
    return f"[pi] saved {os.path.basename(path)} + {os.path.basename(cal_path)}"


class AutoSave:
    """Saves both files DELAY seconds after the last change, off the caller's
    thread, so a burst of edits is one write and nothing waits on the SD card."""

    def __init__(self, pi, stm, delay=1.0, log=print):
        self.pi, self.stm, self.delay, self.log = pi, stm, delay, log
        self._timer = None
        self._lock = threading.Lock()

    def touch(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.delay, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self):
        with self._lock:
            self._timer = None
        try:
            save(self.pi, self.stm)
        except OSError as e:
            self.log(f"[pi] AUTOSAVE FAILED: {e}")

    def flush(self):
        """Write now if a save is pending (call on shutdown)."""
        with self._lock:
            t, self._timer = self._timer, None
        if t is not None:
            t.cancel()
            self._run()
