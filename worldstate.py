"""
Shared state that the producer threads (camera, lidar) write to and the
main fusion loop reads from. The lock-guarded snapshot() is the single
integration seam between the threads.

Conventions (fixed once, enforced everywhere):
  - distances in mm, angles/bearings in degrees
  - lidar frame: 0 deg = robot forward, increasing CCW (adjust in lidar.py
    if your mount differs — see MOUNT_OFFSET_DEG there)
  - "no return at this angle" is represented by float('inf'), never None
"""

import threading
from dataclasses import dataclass, field


@dataclass
class Obstacle:
    color: str          # "RED" | "GREEN" | "MAGENTA" | "UNKNOWN"
    bearing_deg: float  # + = left of forward
    distance_mm: float  # inf if camera-only (no lidar range yet)
    confidence: float = 1.0


@dataclass
class LidarObstacle:
    """
    A discrete object the lidar clustered out of the filtered point cloud.
    Geometry comes entirely from the lidar; `color` is filled in later by
    matching against camera colour detections (UNKNOWN until then).
    """
    bearing_deg: float      # centre bearing of the cluster, + = left of forward
    distance_mm: float      # nearest point in the cluster
    width_deg: float        # angular span of the cluster
    point_count: int        # how many returns formed it
    color: str = "UNKNOWN"  # filled by fusion from the camera


@dataclass
class ColorDetection:
    """Camera output in the new split: a colour seen at a bearing. No distance."""
    color: str          # "RED" | "GREEN" | "MAGENTA"
    bearing_deg: float  # + = left of forward
    confidence: float = 1.0


@dataclass
class CameraResult:
    timestamp: float
    detections: list = field(default_factory=list)  # list[ColorDetection]
    ok: bool = True


@dataclass
class LidarResult:
    timestamp: float
    # raw 360-element array kept for the dashboard / debugging; value mm or inf
    ranges: list = field(default_factory=lambda: [float("inf")] * 360)
    # raw per-angle signal quality (0-63), parallel to ranges; for live tuning
    qualities: list = field(default_factory=lambda: [0] * 360)
    # discrete obstacles clustered from the FILTERED point cloud
    obstacles: list = field(default_factory=list)   # list[LidarObstacle]
    ok: bool = True


class SharedState:
    """
    Holds only the single most-recent result from each producer.
    Overwritten each cycle — never queued — so readers always get 'now'.
    Critical sections are tiny: producers swap in a reference, the reader
    copies two references out. All heavy work happens outside the lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._camera = None   # CameraResult | None
        self._lidar = None    # LidarResult | None

    def set_camera(self, result: CameraResult):
        with self._lock:
            self._camera = result

    def set_lidar(self, result: LidarResult):
        with self._lock:
            self._lidar = result

    def snapshot(self):
        """Return (camera_result, lidar_result) — either may be None early on."""
        with self._lock:
            return self._camera, self._lidar
