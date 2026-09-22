"""
Shared state — camera-first architecture.

Camera detects obstacles (colour + bearing). Lidar supplies a 360-degree
range array. Fusion attaches a distance to each camera obstacle by looking
up the lidar range at that obstacle's bearing.

Conventions (fixed once, enforced everywhere):
  - distances in mm, bearings in degrees
  - bearing: 0 = robot forward, + = left of forward
  - lidar frame: index = integer degree, 0 = forward, increasing CCW
  - "no return / no reading" is float('inf'), never None
"""

import threading
from dataclasses import dataclass, field


@dataclass
class Obstacle:
    color: str          # "RED" | "GREEN" | "MAGENTA" | "UNKNOWN"
    bearing_deg: float  # + = left of forward
    distance_mm: float  # inf until fusion fills it from the lidar
    confidence: float = 1.0


@dataclass
class CameraResult:
    timestamp: float
    obstacles: list = field(default_factory=list)   # list[Obstacle], dist may be inf
    ok: bool = True


@dataclass
class LidarResult:
    timestamp: float
    ranges: list = field(default_factory=lambda: [float("inf")] * 360)
    qualities: list = field(default_factory=lambda: [0] * 360)   # for the dashboard
    ok: bool = True


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._camera = None
        self._lidar = None

    def set_camera(self, result: CameraResult):
        with self._lock:
            self._camera = result

    def set_lidar(self, result: LidarResult):
        with self._lock:
            self._lidar = result

    def snapshot(self):
        with self._lock:
            return self._camera, self._lidar
