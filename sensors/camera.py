"""
Camera adapter — COLOUR ONLY.

In the lidar-first architecture the camera no longer produces obstacles or
geometry. Its single job: report which colours are visible and at what
bearing, so fusion can stamp a colour onto the obstacles the LIDAR detected.

detect_blobs / build_mask stay module-level and unchanged because the
calibration dashboard still uses them to visualise masks. What changed is the
thread's OUTPUT: ColorDetection (colour + bearing), never distance.
"""

import json
import os
import threading
import time

import cv2
import numpy as np

import math

from worldstate import SharedState, CameraResult, ColorDetection

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
# detection helpers — shared with the dashboard
# --------------------------------------------------------------------------

def px_to_bearing(cx, frame_w, hfov_deg):
    """Blob centre-x -> bearing in degrees. + = left of forward.
    Uses a pinhole/tangent model rather than linear interpolation,
    so it stays accurate toward the edges of the frame."""
    half_w = frame_w / 2.0
    hfov_rad = math.radians(hfov_deg)
    f_x = half_w / math.tan(hfov_rad / 2.0)

    pixel_offset = cx - half_w
    bearing_rad = math.atan(pixel_offset / f_x)
    return -math.degrees(bearing_rad)


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


def blobs_to_detections(blobs, frame_w, hfov_deg):
    """Colour + bearing only — no distance, no obstacle semantics."""
    return [
        ColorDetection(
            color=b["colour"],
            bearing_deg=px_to_bearing(b["cx"], frame_w, hfov_deg),
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

    def run(self):
        try:
            self._cam = open_camera()
            while not self._stop.is_set():
                frame = grab_rgb(self._cam, self.swap_rb)
                blobs = detect_blobs(frame, self.hsv_cfg, self.min_area)
                detections = blobs_to_detections(blobs, FRAME_W, self.hfov_deg)
                self.shared.set_camera(
                    CameraResult(timestamp=time.time(),
                                 detections=detections, ok=True))
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
