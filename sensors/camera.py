"""
Camera adapter — camera-first: detects colour AND bearing.

px_to_bearing now uses the pinhole-camera (atan) model instead of a linear
approximation, so bearings stay accurate toward the edges of the frame, not
just near the centre.

Distance is left as inf here; fusion fills it from the lidar range at the
obstacle's bearing.
"""

import json
import math
import os
import threading
import time

import cv2
import numpy as np
from libcamera import Transform

from worldstate import SharedState, CameraResult, Obstacle
from libcamera import Transform

FRAME_W = 640
FRAME_H = 480
PUBLISH_PERIOD = 0.033  # ~30 Hz

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config.json")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return json.load(f)


def save_config(cfg, path=CONFIG_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# camera intrinsics (fisheye) — loaded once; used to undistort bearings
# --------------------------------------------------------------------------

_INTRINSICS = None   # {"K": ndarray(3,3), "D": ndarray(4,1)} or None


def load_intrinsics(cfg=None):
    """
    Parse camera_intrinsics from config into numpy K/D once. Returns
    {"K","D","image_size"} or None if not calibrated yet. Cached in module
    state so px_to_bearing doesn't re-parse every call.
    """
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


def px_to_bearing(cx, frame_w, hfov_deg, cy=None, intrinsics=None,
                  offset_deg=0.0):
    """
    Pixel -> bearing in degrees, + = left of forward.

    If fisheye intrinsics (K, D) are available, undistort the single blob
    centre through the calibrated model and derive the true bearing — correct
    across the whole 160deg field. Otherwise fall back to the pinhole atan
    model (only accurate for a rectilinear lens / near the centre).

    offset_deg is the camera-to-lidar rotational alignment, added at the end.
    """
    intr = intrinsics if intrinsics is not None else load_intrinsics()

    if intr is not None and cy is not None:
        # undistort just this point; returns normalized coords (x = X/Z on the
        # undistorted pinhole plane). bearing = atan(x_norm), left positive.
        pt = np.array([[[float(cx), float(cy)]]], dtype=np.float64)
        und = cv2.fisheye.undistortPoints(pt, intr["K"], intr["D"])
        x_norm = float(und[0, 0, 0])
        return -math.degrees(math.atan(x_norm)) + offset_deg

    # pinhole fallback (no calibration yet)
    f_px = (frame_w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    dx = cx - frame_w / 2.0
    return -math.degrees(math.atan(dx / f_px)) + offset_deg


def build_mask(hsv_img, ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv_img, np.array(lo, dtype=np.uint8),
                        np.array(hi, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# --------------------------------------------------------------------------
# floor-contact filter: a pillar STANDS ON the white mat
# --------------------------------------------------------------------------
#
# Colour alone cannot tell a pillar from a red shirt behind the wall or a green
# banner in the hall. What a pillar has and they do not is contact with the mat:
# directly under its bottom edge (i.e. just in front of it, towards the car) the
# camera sees white floor, and that white floor is the same floor the car is
# standing on — nothing wall-dark in between. So each colour blob must pass:
#
#   contact  the strip of rows just under the blob is mostly white mat
#   linked   that white connects to the floor in front of the car through
#            pixels that are not wall-dark. Orange/blue corner lines and other
#            pillars do not break the link; the black walls do, which is what
#            rejects anything seen over or beyond them.
#
# A blob whose base is at or below the bottom of the view (a very close pillar)
# is accepted: nothing but a pillar can be that close. Thresholds live in
# config.json -> "floor" and are tuned from the dashboard.

FLOOR_DEFAULTS = {
    "enabled": True,
    "white_s_max": 70,       # white mat: saturation at most this ...
    "white_v_min": 120,      # ... and brightness at least this (HSV, 0-255)
    "strip_px": 10,          # height of the contact strip under a blob, rows
    "min_white_frac": 0.4,   # share of that strip that must be white mat
    "linked": True,          # also require the link to the floor in front
    "dark_v_max": 50,        # wall-dark: brightness at most this
    "ignore_bottom_px": 0,   # bottom rows hidden by the car's own body
}
FLOOR_STRIP_GAP_PX = 2       # skip the blob's own soft bottom edge
FLOOR_SEED_ROWS = 8          # "floor in front of the car" = white in these rows
FLOOR_MIN_STRIP_ROWS = 3     # fewer visible strip rows = base below the view

REASON_BELOW_VIEW = "base below view"
REASON_NO_FLOOR = "no white floor under it"
REASON_NOT_LINKED = "not linked to the floor in front (wall between?)"


def floor_params(cfg_floor):
    """config.json "floor" block merged over FLOOR_DEFAULTS."""
    p = dict(FLOOR_DEFAULTS)
    if cfg_floor:
        p.update({k: v for k, v in cfg_floor.items() if k in FLOOR_DEFAULTS})
    return p


def build_floor_mask(hsv_img, fp):
    """White mat: low saturation and bright. 0/255 uint8."""
    m = cv2.inRange(hsv_img,
                    np.array([0, 0, int(fp["white_v_min"])], dtype=np.uint8),
                    np.array([180, int(fp["white_s_max"]), 255], dtype=np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class FloorContext:
    """One frame's floor analysis, shared by every blob in that frame."""

    def __init__(self, hsv_img, fp):
        self.fp = fp
        self.h, self.w = hsv_img.shape[:2]
        self.bottom = max(1, self.h - int(fp["ignore_bottom_px"]))
        self.floor = build_floor_mask(hsv_img, fp)
        self.labels = None      # connected regions of not-wall-dark pixels
        self.seeds = None       # the region(s) holding the floor in front
        if fp["linked"]:
            notdark = (hsv_img[:, :, 2] > int(fp["dark_v_max"])).astype(np.uint8)
            _, self.labels = cv2.connectedComponents(notdark, connectivity=4)
            r0 = max(0, self.bottom - FLOOR_SEED_ROWS)
            white = self.floor[r0:self.bottom] > 0
            seeds = np.unique(self.labels[r0:self.bottom][white])
            seeds = seeds[seeds != 0]
            # No white at all in front (e.g. nose to a wall): the link cannot
            # be judged, so only the contact test applies this frame.
            self.seeds = seeds if seeds.size else None

    def strip(self, x, y, w, h):
        """(x0, y0, x1, y1) of the contact strip under a bounding box."""
        y0 = y + h + FLOOR_STRIP_GAP_PX
        y1 = min(y0 + int(self.fp["strip_px"]), self.bottom)
        m = int(w * 0.2)                       # central 60 % of the width
        x0, x1 = x + m, x + w - m
        if x1 <= x0:
            x0, x1 = x, x + w
        return x0, y0, x1, y1

    def check(self, x, y, w, h):
        """(ok, reason, white_frac). reason is '' for a plain pass."""
        x0, y0, x1, y1 = self.strip(x, y, w, h)
        if y1 - y0 < FLOOR_MIN_STRIP_ROWS:
            return True, REASON_BELOW_VIEW, None
        white = self.floor[y0:y1, x0:x1] > 0
        frac = float(white.mean()) if white.size else 0.0
        if frac < float(self.fp["min_white_frac"]):
            return False, REASON_NO_FLOOR, frac
        if self.labels is not None and self.seeds is not None:
            if not np.isin(self.labels[y0:y1, x0:x1][white], self.seeds).any():
                return False, REASON_NOT_LINKED, frac
        return True, "", frac

    def linked_floor(self):
        """White floor that is linked to the floor in front (for viewers)."""
        if self.labels is None or self.seeds is None:
            return self.floor
        return np.where(np.isin(self.labels, self.seeds) & (self.floor > 0),
                        255, 0).astype(np.uint8)


def detect_blobs(frame_rgb, hsv_cfg, min_area, masks_out=None, floor=None,
                 rejected_out=None, context_out=None):
    """Returns pixel-space blobs: {colour, x, y, w, h, cx, cy, area}.

    floor: config.json's "floor" block. When given and enabled, a blob only
    counts if it stands on the white mat (see FloorContext); the others are
    appended to rejected_out (if a list), each with a "reason". Accepted blobs
    carry "floor" ("on floor" / "base below view") and "white_frac".

    If masks_out is a dict it receives, by colour name, the mask the blobs were
    found in — with the floor filter on, only the pixels of accepted blobs —
    and, under "FLOOR", the white-floor mask. context_out (a dict) receives the
    frame's FloorContext under "floor", for viewers."""
    hsv_img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    fp = floor_params(floor) if floor is not None else None
    ctx = FloorContext(hsv_img, fp) if (fp is not None and fp["enabled"]) else None
    if context_out is not None:
        context_out["floor"] = ctx
    if masks_out is not None and ctx is not None:
        masks_out["FLOOR"] = ctx.floor
    blobs = []
    for colour, ranges in hsv_cfg.items():
        mask = build_mask(hsv_img, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        kept = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            b = {"colour": colour, "x": x, "y": y, "w": w, "h": h,
                 "cx": x + w / 2.0, "cy": y + h / 2.0, "area": area}
            if ctx is not None:
                ok, reason, frac = ctx.check(x, y, w, h)
                b["white_frac"] = frac
                if not ok:
                    b["reason"] = reason
                    if rejected_out is not None:
                        rejected_out.append(b)
                    continue
                b["floor"] = reason or "on floor"
            blobs.append(b)
            kept.append(c)
        if masks_out is not None:
            if ctx is None:
                masks_out[colour] = mask
            else:
                pillar = np.zeros_like(mask)
                cv2.drawContours(pillar, kept, -1, 255, cv2.FILLED)
                masks_out[colour] = cv2.bitwise_and(pillar, mask)
    return blobs


def blobs_to_obstacles(blobs, frame_w, hfov_deg, offset_deg=0.0):
    """Colour + bearing. Uses fisheye undistortion when calibrated (needs cy).
    Distance left inf for fusion to fill."""
    return [
        Obstacle(
            color=b["colour"],
            bearing_deg=px_to_bearing(b["cx"], frame_w, hfov_deg,
                                      cy=b["cy"], offset_deg=offset_deg),
            distance_mm=float("inf"),
            confidence=min(1.0, b["area"] / 5000.0),
        )
        for b in blobs
    ]


def open_camera(width=FRAME_W, height=FRAME_H, hflip=True, vflip=True):
    from picamera2 import Picamera2
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

class CameraThread(threading.Thread):
    def __init__(self, shared: SharedState):
        super().__init__(name="CameraThread", daemon=True)
        self.shared = shared
        self._stop = threading.Event()
        self._cam = None
        cfg = load_config()
        self.hsv_cfg = cfg["hsv"]
        self.hfov_deg = cfg["hfov_deg"]
        self.min_area = cfg["min_blob_area"]
        self.swap_rb = cfg.get("swap_rb", False)
        self.offset_deg = cfg.get("camera_offset_deg", 0.0)
        self.floor = cfg.get("floor", {})     # floor-contact filter (see above)
        load_intrinsics(cfg)          # warm the cache from the same config read

    def run(self):
        try:
            self._cam = open_camera()
            while not self._stop.is_set():
                frame = grab_rgb(self._cam, self.swap_rb)
                blobs = detect_blobs(frame, self.hsv_cfg, self.min_area,
                                     floor=self.floor)
                obstacles = blobs_to_obstacles(blobs, FRAME_W, self.hfov_deg,
                                               offset_deg=self.offset_deg)
                self.shared.set_camera(
                    CameraResult(timestamp=time.time(),
                                 obstacles=obstacles, ok=True))
                time.sleep(PUBLISH_PERIOD)
        except Exception as e:
            print(f"[CameraThread] fatal: {type(e).__name__}: {e}")
        finally:
            try:
                if self._cam is not None:
                    self._cam.stop()
            except Exception:
                pass
            print("[CameraThread] stopped")

    def stop(self):
        self._stop.set()
