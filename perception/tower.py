"""
perception/tower.py - free-standing "tower" test on a raw scan.

CREDIT
    The valley test is the LiDAR tower-detection idea published by Team LazyGo
    (WRO 2025, github.com/a-n-m-noor/lazygo_wro2025). A traffic sign makes a
    VALLEY in the scan: the readings step closer, run across the object, then
    step back out. Check the depth of that valley, and convert its angular
    width to a physical size with s = r*theta. Roughly 50 mm wide and a valley
    deeper than ~200 mm is a sign; anything else is not.

WHY IT EARNS ITS PLACE HERE
    Everything else in this package subtracts the KNOWN map first, so it can
    only be as good as the pose. This test uses no map and no pose at all - it
    is pure scan shape - so it stays valid exactly when localization is shaky,
    which is when the false pillars appear.

    It also separates signs from the parking blocks on geometry rather than on
    position. A block is flush against the outer wall, so whatever is behind it
    is barely 200 mm further away and its valley is SHALLOW. A sign stands
    ~310 mm clear of both walls, so its valley is DEEP. The two stop looking
    alike without needing to know where the car is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_DEPTH_MM = 200.0     # a valley shallower than this is not a free object
SIZE_MIN_MM = 25.0       # a 50 mm sign, with slack for grazing views
SIZE_MAX_MM = 95.0
MIN_RANGE_MM = 90.0
MAX_RANGE_MM = 2600.0
MAX_WIDTH_DEG = 60       # wider than this is a wall, not an object
MAX_GAP = 3              # dropouts tolerated inside one object


@dataclass(frozen=True)
class Tower:
    bearing_deg: float       # sensor frame, 0 = forward, + = left
    range_mm: float          # to the near face
    size_mm: float           # from s = r * theta
    depth_mm: float          # how far it stands clear of the background
    width_deg: float
    i0: int                  # first and last scan index of the valley
    i1: int


MERGE_MM = 120.0         # across a dropout, this close in range = same object


def _valid(r):
    return np.isfinite(r) & (r > MIN_RANGE_MM) & (r < MAX_RANGE_MM)


def segments(r, ok, n, max_gap=MAX_GAP, merge_mm=MERGE_MM):
    """Split the scan into objects, tolerating dropouts.

    A real RPLidar drops rays AT the edges of things, so on live data there is
    usually no sharp adjacent step to find - the sign is separated from the
    wall by a run of inf, not by a 300 mm jump between neighbouring bins. So
    group the valid returns instead: start a new group when the angular gap is
    too wide, or when the range jumps across that gap.
    """
    idxs = [i for i in range(n) if ok[i]]
    if not idxs:
        return []
    groups, cur = [], [idxs[0]]
    for i in idxs[1:]:
        gap = i - cur[-1]
        jump = abs(float(r[i]) - float(r[cur[-1]]))
        if gap > max_gap + 1 or jump > merge_mm:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    # stitch across 0 deg if the ends belong together
    if len(groups) > 1:
        a, b = groups[-1], groups[0]
        if (b[0] + n - a[-1]) <= max_gap + 1 and \
                abs(float(r[b[0]]) - float(r[a[-1]])) <= merge_mm:
            groups[0] = a + b
            groups.pop()
    return groups


def _find_by_segments(r, ok, n, min_depth, size_min, size_max):
    """An object is a segment that sits closer than BOTH of its neighbours."""
    groups = segments(r, ok, n)
    if len(groups) < 3:
        return []
    means = [float(np.mean(r[g])) for g in groups]
    out = []
    for k, g in enumerate(groups):
        prev_m = means[(k - 1) % len(groups)]
        next_m = means[(k + 1) % len(groups)]
        near = float(np.min(r[g]))
        depth = min(prev_m, next_m) - near
        if depth < min_depth:
            continue
        span = (g[-1] - g[0]) % n + 1
        if span > MAX_WIDTH_DEG:
            continue
        size = means[k] * math.radians(span)
        if not (size_min <= size <= size_max):
            continue
        mid = g[len(g) // 2]
        out.append(Tower(bearing_deg=float(mid if mid <= 180 else mid - 360),
                         range_mm=near, size_mm=size, depth_mm=depth,
                         width_deg=float(span), i0=g[0], i1=g[-1]))
    return out


def find_towers(ranges, min_depth=MIN_DEPTH_MM,
                size_min=SIZE_MIN_MM, size_max=SIZE_MAX_MM):
    """Valleys in the scan that are the right size to be a traffic sign.

    Returns [Tower], nearest first. No map, no pose.
    """
    r = np.asarray(ranges, dtype=np.float64)
    n = len(r)
    ok = _valid(r)
    if ok.sum() < 8:
        return []

    out = _find_by_segments(r, ok, n, min_depth, size_min, size_max)
    if out:
        out.sort(key=lambda t: t.range_mm)
        return out

    # fallback: the sharp-adjacent-step form, for clean/simulated scans
    out = []
    for a in range(n):
        b = (a + 1) % n
        if not (ok[a] and ok[b]):
            continue
        # LEADING EDGE: the reading steps sharply CLOSER
        if r[a] - r[b] < min_depth:
            continue

        # Run across the object until it steps sharply back OUT.
        # Dropouts happen mid-object on a real scanner (a dark or grazing
        # facet returns nothing), so tolerate a few rather than abandoning the
        # valley - a single inf inside the object used to lose the whole sign.
        seg = [b]
        closed = False
        nxt = None
        gaps = 0
        w = 1
        while w <= MAX_WIDTH_DEG:
            j = (b + w) % n
            if not ok[j]:
                gaps += 1
                if gaps > MAX_GAP:
                    break
                w += 1
                continue
            if r[j] - r[seg[-1]] >= min_depth:      # TRAILING EDGE
                closed = True
                nxt = j
                break
            seg.append(j)
            gaps = 0
            w += 1
        if not closed:
            continue

        vals = r[seg]
        near = float(np.min(vals))
        back = min(float(r[a]), float(r[nxt]))  # background on either shoulder
        depth = back - near
        if depth < min_depth:
            continue

        w_deg = (seg[-1] - b) % n + 1           # angular span, gaps included
        size = float(np.mean(vals)) * math.radians(w_deg)   # s = r * theta
        if not (size_min <= size <= size_max):
            continue

        mid = seg[len(seg) // 2]
        out.append(Tower(bearing_deg=float(mid if mid <= 180 else mid - 360),
                         range_mm=near, size_mm=size, depth_mm=depth,
                         width_deg=float(w_deg), i0=seg[0], i1=seg[-1]))
    out.sort(key=lambda t: t.range_mm)
    return out


def tower_points(ranges, pose, towers):
    """Mat-frame (x, y) of each tower's centre face, for cross-checking against
    the seat grid."""
    x0, y0, th = pose
    pts = []
    for t in towers:
        a = math.radians(t.bearing_deg) + th
        pts.append((x0 + t.range_mm * math.cos(a),
                    y0 + t.range_mm * math.sin(a)))
    return pts
