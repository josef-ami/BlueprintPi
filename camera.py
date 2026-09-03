"""
Camera adapter.

Detection logic lives in MODULE-LEVEL functions so that both the robot
(CameraThread) and the calibration dashboard run the exact same code path.
The dashboard passes live slider values; the robot passes values loaded from
config.json at startup. Nothing here is duplicated between the two.

The camera's job is COLOR + BEARING. Distance stays inf; fusion fills it
from the lidar.
"""

import json
import os
import threading
import time

import cv2
import numpy as np

from worldstate import SharedState, CameraResult, Obstacle

FRAME_W = 640
FRAME_H = 480
PUBLISH_PERIOD = 0.033  # ~30 Hz cap

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config.json")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def load_config(path=CONFIG_PATH):
    """Read tuning values written by the dashboard. Raises if missing/invalid."""
    with open(path, "r") as f:
        return json.load(f)


def save_config(cfg, path=CONFIG_PATH):
    """Write tuning values (used by the dashboard's Save)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, path)   # atomic; never leaves a half-written config


# --------------------------------------------------------------------------
# detection — shared by robot and dashboard
# --------------------------------------------------------------------------

def px_to_bearing(cx, frame_w, hfov_deg):
    """Blob centre-x -> bearing in degrees. + = left of forward."""
    norm = (cx - frame_w / 2.0) / (frame_w / 2.0)   # -1 left edge .. +1 right
    return -norm * (hfov_deg / 2.0)


def build_mask(hsv_img, ranges):
    """OR together every (lo, hi) range for one colour, then clean it up."""
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv_img, np.array(lo, dtype=np.uint8),
                        np.array(hi, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        return np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)    # kill speckle
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # fill holes
    return mask


def detect_blobs(frame_rgb, hsv_cfg, min_area):
    """
    Returns a list of dicts, one per accepted blob:
      {colour, x, y, w, h, cx, cy, area}
    Pixel space only — no bearing yet, so callers can apply their own HFOV.
    """
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
            blobs.append({
                "colour": colour, "x": x, "y": y, "w": w, "h": h,
                "cx": x + w / 2.0, "cy": y + h / 2.0, "area": area,
            })
    return blobs


def blobs_to_obstacles(blobs, frame_w, hfov_deg):
    """Convert pixel blobs into Obstacle records (distance filled by fusion)."""
    return [
        Obstacle(
            color=b["colour"],
            bearing_deg=px_to_bearing(b["cx"], frame_w, hfov_deg),
            distance_mm=float("inf"),
            confidence=min(1.0, b["area"] / 5000.0),
        )
        for b in blobs
    ]


def open_camera(width=FRAME_W, height=FRAME_H):
    """Start Picamera2 and return it. Only ONE process may hold the camera."""
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}))
    cam.start()
    time.sleep(0.5)   # let auto-exposure settle
    return cam


def grab_rgb(cam, swap_rb=False):
    """
    One frame as RGB. Picamera2's 'RGB888' channel order varies by version,
    so swap_rb is a config toggle — flip it if red and blue look swapped.
    """
    frame = cam.capture_array()
    if swap_rb:
        frame = frame[:, :, ::-1]
    return np.ascontiguousarray(frame)


# --------------------------------------------------------------------------
# robot-side thread
# --------------------------------------------------------------------------

class CameraThread(threading.Thread):
    def __init__(self, shared: SharedState):
        super().__init__(name="CameraThread", daemon=True)
        self.shared = shared
        self._stop = threading.Event()
        self._cam = None
        cfg = load_config()                       # read once at startup
        self.hsv_cfg = cfg["hsv"]
        self.hfov_deg = cfg["hfov_deg"]
        self.min_area = cfg["min_blob_area"]
        self.swap_rb = cfg.get("swap_rb", False)

    def run(self):
        try:
            self._cam = open_camera()
            while not self._stop.is_set():
                frame = grab_rgb(self._cam, self.swap_rb)
                blobs = detect_blobs(frame, self.hsv_cfg, self.min_area)
                obstacles = blobs_to_obstacles(blobs, FRAME_W, self.hfov_deg)
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
