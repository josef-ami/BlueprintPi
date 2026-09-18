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


def detect_blobs(frame_rgb, hsv_cfg, min_area):
    """Returns pixel-space blobs: {colour, x, y, w, h, cx, cy, area}."""
    hsv_img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    blobs = []
    for colour, ranges in hsv_cfg.items():
        mask = build_mask(hsv_img, ranges)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            blobs.append({"colour": colour, "x": x, "y": y, "w": w, "h": h,
                          "cx": x + w / 2.0, "cy": y + h / 2.0, "area": area})
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


def open_camera(width=FRAME_W, height=FRAME_H):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}))
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
        load_intrinsics(cfg)          # warm the cache from the same config read

    def run(self):
        try:
            self._cam = open_camera()
            while not self._stop.is_set():
                frame = grab_rgb(self._cam, self.swap_rb)
                blobs = detect_blobs(frame, self.hsv_cfg, self.min_area)
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
