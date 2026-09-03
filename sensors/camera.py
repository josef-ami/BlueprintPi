"""
Camera adapter.

A plain (non-async) thread: capture a frame, threshold in HSV, extract colored
blobs, convert each blob's horizontal position to a bearing, publish a
CameraResult. Distance is left as inf here — the fusion step fills real distance
from the lidar. The camera's job is COLOR + BEARING, not range.
"""

import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from worldstate import SharedState, CameraResult, Obstacle

FRAME_W = 640
FRAME_H = 480
HFOV_DEG = 62.0            # horizontal field of view of your lens; MEASURE THIS
MIN_BLOB_AREA = 300       # px, reject specks; tune on the field
PUBLISH_PERIOD = 0.033    # ~30 Hz cap

# HSV ranges — PLACEHOLDERS. Calibrate against the actual pillars under
# competition lighting (rules even say you get calibration time). Red wraps
# around H=0 so it needs two ranges.
HSV = {
    "RED":   [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "GREEN": [((40, 80, 60), (85, 255, 255))],
    "MAGENTA": [((140, 80, 80), (165, 255, 255))],
}


def _px_to_bearing(cx):
    """Map blob center-x (0..FRAME_W) to bearing in deg, + = left of forward."""
    # image x increases to the right; forward is center. Left should be positive.
    norm = (cx - FRAME_W / 2) / (FRAME_W / 2)   # -1 (left edge) .. +1 (right edge)
    return -norm * (HFOV_DEG / 2)


class CameraThread(threading.Thread):
    def __init__(self, shared: SharedState):
        super().__init__(name="CameraThread", daemon=True)
        self.shared = shared
        self._stop = threading.Event()
        self._cam = None

    def run(self):
        self._cam = Picamera2()
        cfg = self._cam.create_preview_configuration(
            main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(0.5)   # let auto-exposure settle
        try:
            while not self._stop.is_set():
                frame = self._cam.capture_array()          # RGB
                obstacles = self._detect(frame)
                self.shared.set_camera(
                    CameraResult(timestamp=time.time(), obstacles=obstacles, ok=True)
                )
                time.sleep(PUBLISH_PERIOD)
        except Exception as e:
            print(f"[CameraThread] fatal: {type(e).__name__}: {e}")
        finally:
            try:
                self._cam.stop()
            except Exception:
                pass
            print("[CameraThread] stopped")

    def _detect(self, frame_rgb):
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        found = []
        for color, ranges in HSV.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            # clean the mask so contours stop breathing
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < MIN_BLOB_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w / 2
                found.append(Obstacle(
                    color=color,
                    bearing_deg=_px_to_bearing(cx),
                    distance_mm=float("inf"),   # filled by fusion from lidar
                    confidence=min(1.0, area / 5000.0),
                ))
        return found

    def stop(self):
        self._stop.set()
