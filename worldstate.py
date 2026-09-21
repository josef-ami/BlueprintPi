"""
Shared state - what the sensor threads publish and the control loop reads.

Conventions (fixed once, enforced everywhere):
  - distances in mm, angles in degrees
  - bearing / yaw: 0 = robot forward, + = LEFT of forward
  - lidar frame: index = integer degree, 0 = forward, increasing anticlockwise
  - car frame: x forward, y left, origin at the LiDAR
  - lane frame: `along` down the straight, `lat` + = LEFT of the lane centre
  - "no return / no reading" is float('inf') for ranges, None for a pick

The camera says WHICH colour and in WHICH direction. The LiDAR says how far,
and independently reports objects whose colour nothing has named yet. Neither
one decides anything; control/ does.
"""

import threading
from dataclasses import dataclass, field


def wrap180(angle):
    """Fold an angle into (-180, 180]. The conventions above only make sense
    if every angle difference goes through here."""
    return (angle + 180.0) % 360.0 - 180.0


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class Pillar:
    """One accepted blob: what the camera knows about it on its own."""
    colour: str               # "RED" | "GREEN"
    bearing_deg: float        # + = left of forward
    err_px: int               # centre x - 320 in a 640 px frame, + = right
    area: int                 # in detection pixels (320x240)
    box: tuple = None         # (x0, y0, x1, y1) in full-frame pixels
    white_frac: float = None  # mat fraction in the contact strip, for the UI
    x_mm: float = None        # car frame, filled by locate_pillar()
    y_mm: float = None


@dataclass
class CameraResult:
    timestamp: float
    pillars: list = field(default_factory=list)   # list[Pillar], largest first
    best: object = None                           # the largest, or None
    second: object = None                         # the next-largest, or None
    ok: bool = True
    seq: int = 0                                  # +1 per detection pass
    space: str = "HSV"                            # which colour space named it


@dataclass
class LidarResult:
    timestamp: float
    ranges: list = field(default_factory=lambda: [float("inf")] * 360)
    qualities: list = field(default_factory=lambda: [0] * 360)
    rev: int = 0
    ok: bool = True


@dataclass
class WallFit:
    """What the two 45 deg cone fits found this revolution."""
    left_mm: float = None        # perpendicular distance to the left wall
    right_mm: float = None
    yaw_deg: float = None        # car yaw to the walls, + = pointing LEFT
    left_line: tuple = None      # (m, c) of y = m x + c in the LiDAR frame
    right_line: tuple = None

    def both(self):
        return self.left_mm is not None and self.right_mm is not None


class SharedState:
    """One lock, three slots. Producers replace; consumers snapshot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._camera = None
        self._lidar = None

    def set_camera(self, result):
        with self._lock:
            self._camera = result

    def set_lidar(self, result):
        with self._lock:
            self._lidar = result

    def snapshot(self):
        with self._lock:
            return self._camera, self._lidar
