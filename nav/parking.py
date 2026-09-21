"""
nav/parking.py - locate the parallel-parking bay in the mat frame.

The bay is two magenta blocks (200 x 20 x 100 mm) standing against the OUTER
wall of the start straight, 1.5 * car_len apart. In a LiDAR scan those blocks
are non-wall returns sitting within ~200 mm of an outer wall - much closer to
the wall than any traffic-sign seat (seats are >=300 mm off the outer wall), so
they separate cleanly from pillars by their distance to the wall.

detect() takes the non-seat ("unclassified") detections that WorldBelief has
already map-subtracted and seat-filtered, decides which side they hug and where
along that side, and returns an arena.ParkingBay. A camera magenta bearing can
be passed to disambiguate, but geometry alone is usually enough.
"""

from __future__ import annotations

import numpy as np

from . import arena

NEAR_WALL_MM = 260.0        # a return this close to an outer wall is a block/edge
MIN_ALONG_SEP = 150.0       # two clusters at least this far apart = the two ends


def _side_and_along(x: float, y: float):
    """Nearest outer wall for point (x, y) -> (side, along, wall_gap_mm)."""
    h = arena.OUTER_HALF
    gaps = {"N": h - y, "S": y + h, "E": h - x, "W": x + h}
    side = min(gaps, key=lambda k: gaps[k])
    along = x if side in ("N", "S") else y
    return side, along, gaps[side]


def detect(unclassified, car_len_mm: float = 175.0, cam_magenta_bearing=None):
    """unclassified: [(x, y, n, w)] mat frame. Returns arena.ParkingBay | None."""
    hugging = {}
    for (x, y, *_rest) in unclassified:
        side, along, gap = _side_and_along(x, y)
        if gap <= NEAR_WALL_MM:
            hugging.setdefault(side, []).append(along)
    if not hugging:
        return None
    # pick the side with the most wall-hugging returns
    side = max(hugging, key=lambda k: len(hugging[k]))
    alongs = sorted(hugging[side])
    if len(alongs) >= 2 and (alongs[-1] - alongs[0]) >= MIN_ALONG_SEP:
        center = 0.5 * (alongs[0] + alongs[-1])
    else:
        center = float(np.mean(alongs))
    return arena.parking_bay(side, center, car_len_mm)
