"""
perception/selfmask.py - blank out the bearings where the car sees ITSELF.

WHY THIS EXISTS
    Two scans taken at different places on the mat were compared bearing by
    bearing. Anything close in BOTH cannot be the arena - it has to be the car.
    That found, on this robot:

        138-232 deg   4-16 mm    the chassis: the whole rear arc is blocked
        54-76 deg     ~180 mm    structure on the left
        297-323 deg   ~170-200   structure on the right

    The two ~180 mm structures matter most. They are the right angular width
    and stand-off to pass the tower test as a ~44 mm sign, so they manufacture
    a phantom pillar on every single scan, for ever. They also enter the scan
    matcher as points that are nowhere near any wall, dragging the match score
    down and pulling the pose with them.

    Masking them is not cosmetic: it removes a permanent false pillar AND
    should lift the match score, because the matcher stops being asked to
    explain returns that belong to the robot.

CALIBRATE, DO NOT GUESS
    The bands are specific to one build. tools/calibrate_self.py derives them
    from real scans (move the car between captures so only the car stays
    constant) and writes them to config.json under lidar.blind_deg.
"""

from __future__ import annotations

import json
import os

import numpy as np

# NOTHING is masked by default, on purpose.
#
# The obvious candidate was this car's rear arc (138-232 deg), which returns
# 4-16 mm where the chassis blocks the beam. But every consumer already gates
# at 80-90 mm, so those returns are discarded anyway - masking them buys
# nothing and only adds a way to lose real objects.
#
# Two side bands near +/-55 deg (~180 mm) also repeated across captures and
# looked like structure. Their range differed by ~30 mm between captures, and
# rigid structure repeats exactly: they were a traffic sign placed close to the
# car. Masking them WOULD have deleted a correct 60 mm detection.
#
# So bands are opt-in, from config.json (lidar.blind_deg), and are meant to be
# derived by tools/calibrate_self.py ON AN EMPTY MAT - never guessed by eye.
DEFAULT_BANDS = []
SELF_MAX_MM = 260.0          # a return closer than this can plausibly be the car

_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "config.json")


def load_bands(path=_CFG):
    """Blind bands from config.json (lidar.blind_deg), else the defaults."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        bands = cfg.get("lidar", {}).get("blind_deg")
        if bands:
            return [tuple(b) for b in bands]
    except Exception:                                   # noqa: BLE001
        pass
    return list(DEFAULT_BANDS)


def apply(ranges, bands=None):
    """Return a copy of the scan with the car's own returns removed."""
    r = np.asarray(ranges, dtype=np.float64).copy()
    n = len(r)
    for lo, hi in (load_bands() if bands is None else bands):
        lo, hi = int(lo) % n, int(hi) % n
        if lo <= hi:
            r[lo:hi + 1] = np.inf
        else:                                           # wraps through 0
            r[lo:] = np.inf
            r[:hi + 1] = np.inf
    return r


def calibrate(scans, close_mm=SELF_MAX_MM, min_frac=0.6, join=3):
    """Bearings that are close in MOST scans are the car, not the arena.

    Pass scans taken at DIFFERENT places on the mat - that is what makes the
    arena vary while the car stays put. Returns [(lo, hi)] degree bands.
    """
    scans = [np.asarray(s, dtype=np.float64) for s in scans]
    if not scans:
        return []
    n = len(scans[0])
    hits = np.zeros(n)
    for s in scans:
        hits += (np.isfinite(s) & (s < close_mm)).astype(float)
    close = [i for i in range(n) if hits[i] / len(scans) >= min_frac]
    if not close:
        return []
    bands, cur = [], [close[0]]
    for i in close[1:]:
        if i - cur[-1] <= join:
            cur.append(i)
        else:
            bands.append((cur[0], cur[-1]))
            cur = [i]
    bands.append((cur[0], cur[-1]))
    return bands
