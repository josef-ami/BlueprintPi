"""
nav/arena.py - the SEMANTIC map on top of the wall map in geom.py.

geom.py knows the walls (two concentric squares, fixed by the rulebook).
This module adds the two things a wall map does not carry but perception and
planning both need:

    1. SEATS  - the finite set of places a traffic sign is allowed to stand.
                The rulebook draws 50x50 mm sign seats in the straight
                sections; a sign is always centred on one of them. Knowing
                the seats turns "is there a pillar somewhere ahead" into the
                far easier "is seat N3 occupied, and if so what colour".
    2. PARKING - the parallel-parking bay: two magenta blocks against the
                outer wall of the start straight, 1.5 * car length apart.

FRAME (identical to geom.py / localize.py)
    origin at the centre of the field, x east, y north, millimetres.
    Seat (x, y) is the centre of the sign's 50x50 footprint on the mat.

    Per straight side we use a local (along, lat) description and map it to
    mat (x, y):
        along - position down the corridor, 0 at the side's mid-point
        lat   - across the corridor, 0 at the corridor centre-line,
                + towards the OUTER wall, - towards the inner block.

WHAT IS PINNED vs WHAT IS PARAMETRIC
    The wall geometry (OUTER/INNER/CORRIDOR) is exact from geom.py.
    The seat grid below (ALONG_STATIONS x LAT_OFFSETS) is a reasonable
    superset of the WRO FE sign positions but the EXACT station coordinates
    must be verified against the official field drawing before a competition.
    Perception never hard-depends on them: a detected pillar is snapped to the
    nearest seat within SNAP_TOL_MM, and if none is close it is still reported
    as a free detection - so refining these numbers later cannot break the
    pipeline, it only sharpens the "which seat" label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geom import OUTER, INNER, CORRIDOR, PILLAR, WALL_SEGMENTS, raycast

# corridor centre-line offset from the field centre (mm): 1000 for the
# standard 3000/1000 field.
CORRIDOR_CENTER = (INNER / 2.0 + OUTER / 2.0) / 2.0        # = 1000.0
HALF_SIDE = INNER / 2.0                                    # straight half-length along, = 500
OUTER_HALF = OUTER / 2.0                                   # = 1500

# ---- seat grid (PARAMETRIC - verify against the official field drawing) ----
# stations down each straight (along, mm) and lanes across it (lat, mm).
ALONG_STATIONS = (-300.0, 0.0, 300.0)
LAT_OFFSETS = (-190.0, 190.0)     # two lanes, well inside the +/-420 mm limit
SEAT_SIZE = PILLAR                # 50 mm
EVAL_CIRCLE_D = 85.0              # rulebook evaluation circle around a seat
SNAP_TOL_MM = 160.0              # a detection this close to a seat IS that seat

# sides of the loop, in loop order. `sign` orients `along` so it always runs
# the same rotational way round the track (not needed for a static seat, kept
# for planner use later).
SIDES = ("N", "E", "S", "W")


def _place(side: str, along: float, lat: float) -> tuple[float, float]:
    """(along, lat) on a given side -> mat (x, y)."""
    cc = CORRIDOR_CENTER
    if side == "N":
        return (along, cc + lat)
    if side == "S":
        return (along, -cc - lat)
    if side == "E":
        return (cc + lat, along)
    if side == "W":
        return (-cc - lat, along)
    raise ValueError(side)


@dataclass(frozen=True)
class Seat:
    id: str
    x: float
    y: float
    side: str
    along: float
    lat: float


def build_seats() -> list[Seat]:
    seats: list[Seat] = []
    for side in SIDES:
        i = 0
        for a in ALONG_STATIONS:
            for l in LAT_OFFSETS:
                x, y = _place(side, a, l)
                seats.append(Seat(f"{side}{i}", x, y, side, a, l))
                i += 1
    return seats


SEATS: list[Seat] = build_seats()
_SEAT_XY = np.array([[s.x, s.y] for s in SEATS], dtype=np.float64)


def nearest_seat(x: float, y: float, tol: float = SNAP_TOL_MM):
    """Return (Seat, dist_mm) for the closest seat within tol, else (None, d)."""
    d = np.hypot(_SEAT_XY[:, 0] - x, _SEAT_XY[:, 1] - y)
    k = int(np.argmin(d))
    return (SEATS[k], float(d[k])) if d[k] <= tol else (None, float(d[k]))


def square_segments(cx: float, cy: float, size: float) -> np.ndarray:
    """The four wall segments of an axis-aligned square footprint, for the
    ray-caster (used to render pillars/parking into a synthetic scan)."""
    h = size / 2.0
    c = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    return np.array([[c[i][0], c[i][1], c[(i + 1) % 4][0], c[(i + 1) % 4][1]]
                     for i in range(4)], dtype=np.float64)


def pillar_segments(x: float, y: float) -> np.ndarray:
    return square_segments(x, y, SEAT_SIZE)


# ------------------------------- parking --------------------------------
MAGENTA_LONG = 200.0     # block long side (mm)
MAGENTA_THICK = 20.0     # block thickness (mm)


@dataclass(frozen=True)
class ParkingBay:
    side: str
    along_center: float
    car_len_mm: float
    # geometry in the mat frame
    length: float                 # 1.5 * car length, the gap between blocks
    left_block: np.ndarray        # 4 segments
    right_block: np.ndarray
    rect: tuple                   # (x0, y0, x1, y1) drivable bay, for the UI


def parking_bay(side: str, along_center: float, car_len_mm: float) -> ParkingBay:
    """A parallel-parking bay against the OUTER wall of `side`.

    The bay is `1.5 * car_len` long down the corridor and 200 mm deep from the
    outer wall inward. The two magenta blocks cap its ends.
    """
    L = 1.5 * car_len_mm
    depth = 200.0
    a0, a1 = along_center - L / 2.0, along_center + L / 2.0
    # magenta blocks cap each end of the bay
    left = _block_segments(side, a0 - MAGENTA_THICK / 2.0)
    right = _block_segments(side, a1 + MAGENTA_THICK / 2.0)
    # drivable rectangle (mat frame), depth from outer wall inward
    if side in ("N", "S"):
        ys = (OUTER_HALF - depth, OUTER_HALF) if side == "N" else (-OUTER_HALF, -OUTER_HALF + depth)
        rect = (a0, ys[0], a1, ys[1])
    else:
        xs = (OUTER_HALF - depth, OUTER_HALF) if side == "E" else (-OUTER_HALF, -OUTER_HALF + depth)
        rect = (xs[0], a0, xs[1], a1)
    return ParkingBay(side, along_center, car_len_mm, L, left, right, rect)


def outer_face_lat(side: str) -> float:
    # lat of the outer wall face relative to corridor centre
    return OUTER_HALF - CORRIDOR_CENTER


def _block_segments(side: str, along: float) -> np.ndarray:
    """A magenta block: long side (200) runs across the corridor from the
    outer wall inward, thickness (20) along the corridor at `along`."""
    lat_out = outer_face_lat(side)
    lat_in = lat_out - MAGENTA_LONG
    # two along edges +/- half thickness, two lat ends
    a0, a1 = along - MAGENTA_THICK / 2.0, along + MAGENTA_THICK / 2.0
    p = [_place(side, a0, lat_in), _place(side, a1, lat_in),
         _place(side, a1, lat_out), _place(side, a0, lat_out)]
    return np.array([[p[i][0], p[i][1], p[(i + 1) % 4][0], p[(i + 1) % 4][1]]
                     for i in range(4)], dtype=np.float64)


# --------------------------- convenience for the UI ---------------------
def wall_segments() -> np.ndarray:
    return WALL_SEGMENTS


def seat_list() -> list[dict]:
    return [{"id": s.id, "x": s.x, "y": s.y, "side": s.side} for s in SEATS]


if __name__ == "__main__":
    print(f"{len(SEATS)} seats; corridor centre {CORRIDOR_CENTER} mm")
    for s in SEATS[:6]:
        print(f"  {s.id}: ({s.x:.0f}, {s.y:.0f})")
    bay = parking_bay("S", 0.0, 175.0)
    print(f"parking bay S len={bay.length:.0f} rect={tuple(round(v) for v in bay.rect)}")
