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


class ParkingTracker:
    """Accumulates parking sightings instead of believing the latest frame.

    The bay is bolted to the mat for the whole round, so a detection that moves
    between frames is noise, not the bay moving. Publishing the newest detection
    every frame made it jump between walls whenever scan noise near some other
    outer wall happened to form a plausible pair.

    So candidates vote, exactly as PillarMap does for signs: a bay is published
    only after CONFIRM sightings, its position is refined by a running average
    rather than replaced, and once confirmed a rival must beat it clearly before
    it can take over.
    """

    MATCH_MM = 250.0      # same bay if it is this close, along the same wall
    CONFIRM = 4           # sightings before a bay is published at all
    TAKEOVER = 3          # a rival must lead by this many hits to displace it

    def __init__(self):
        self.cands = []           # [{side, center, car_len, hits}]
        self.best = None          # arena.ParkingBay once confirmed

    def update(self, bay):
        """Fold in one frame's detection (or None). Returns the believed bay."""
        if bay is not None:
            for c in self.cands:
                if c["side"] == bay.side and \
                        abs(c["center"] - bay.along_center) < self.MATCH_MM:
                    k = 1.0 / min(c["hits"] + 1, 12)      # running average
                    c["center"] += k * (bay.along_center - c["center"])
                    c["car_len"] += k * (bay.car_len_mm - c["car_len"])
                    c["hits"] += 1
                    break
            else:
                self.cands.append({"side": bay.side,
                                   "center": bay.along_center,
                                   "car_len": bay.car_len_mm, "hits": 1})

        if not self.cands:
            return self.best
        top = max(self.cands, key=lambda c: c["hits"])
        if top["hits"] < self.CONFIRM:
            return self.best                     # not yet trustworthy
        if self.best is not None and top["side"] != self.best.side:
            cur = next((c["hits"] for c in self.cands
                        if c["side"] == self.best.side), 0)
            if top["hits"] < cur + self.TAKEOVER:
                return self.best                 # not a clear enough win
        self.best = arena.parking_bay(top["side"], top["center"], top["car_len"])
        return self.best

    @property
    def confirmed(self):
        return self.best is not None


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
