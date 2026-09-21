"""
nav/geom.py - the arena, from the rulebook. Nothing here is measured or
learned; it is all fixed by WRO Future Engineers 2026 and known before the
round starts.

    racetrack inner size      3000 x 3000 mm  (mat is 3200 x 3200)
    distance between borders  1000 mm (+/- 10 for the International Final)
    => internal block          1000 x 1000 mm, centred
    wall height               100 mm, both exterior and interior
    traffic sign              50 x 50 x 100 mm

MAT FRAME
    origin at the centre of the field, x east, y north, millimetres.
    The LiDAR is at 55 mm above the mat, below the 100 mm wall tops, so it
    sees both walls and signs for their full height.

If your measured corridor differs from 1000 mm (the rules allow +/-10), set
CORRIDOR here before anything else imports this module - everything
downstream is derived from it.
"""

from __future__ import annotations

import math

import numpy as np

def _measured(path=None):
    """Arena dimensions MEASURED from the mat, if they have been.

    calibrationmat.py --build writes arena_cal.json from the accumulated laps.
    The rulebook figures below are the fallback, not the truth: a mat can be
    built a few tens of mm off, and a wrong corridor width biases every
    map-based pose. Measuring beat assuming here - this mat came out
    2990 / 1090 / 950 against the rulebook's 3000 / 1000 / 1000.
    """
    import json
    import os
    p = path or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "arena_cal.json")
    try:
        with open(p, "r", encoding="utf-8") as f:
            c = json.load(f)
        return float(c["outer_mm"]), float(c["corridor_mm"])
    except Exception:                                   # noqa: BLE001
        return 3000.0, 1000.0                           # rulebook default


OUTER, CORRIDOR = _measured()   # racetrack inner size, distance between borders
INNER = OUTER - 2 * CORRIDOR    # internal block (falls out of the other two)
MID = (OUTER + INNER) / 4.0     # mid-corridor square half-size, 1000 mm
WALL_H = 100.0
PILLAR = 50.0


def _square_segments(side: float) -> np.ndarray:
    h = side / 2.0
    c = [(-h, -h), (h, -h), (h, h), (-h, h)]
    return np.array([[c[i][0], c[i][1], c[(i + 1) % 4][0], c[(i + 1) % 4][1]]
                     for i in range(4)], dtype=np.float64)


WALL_SEGMENTS = np.vstack([_square_segments(OUTER), _square_segments(INNER)])


def raycast(px: float, py: float, angles_rad: np.ndarray,
            segs: np.ndarray, max_range: float) -> np.ndarray:
    """Distance from (px, py) along each angle to the nearest segment.

    Solves P + t*d = A + u*(B-A) for every (ray, segment) pair and keeps
    t > 0, 0 <= u <= 1. Used only at start-up, to measure the corridor width
    at each station of the reference line.
    """
    dx = np.cos(angles_rad)[:, None]
    dy = np.sin(angles_rad)[:, None]
    ax, ay = segs[:, 0][None, :], segs[:, 1][None, :]
    ex = (segs[:, 2] - segs[:, 0])[None, :]
    ey = (segs[:, 3] - segs[:, 1])[None, :]
    wx, wy = ax - px, ay - py
    denom = dx * ey - dy * ex
    safe = np.abs(denom) > 1e-12
    den = np.where(safe, denom, 1.0)
    t = (wx * ey - wy * ex) / den
    u = (wx * dy - wy * dx) / den
    hit = safe & (t > 1.0) & (u >= 0.0) & (u <= 1.0) & (t < max_range)
    return np.where(hit, t, np.inf).min(axis=1)
