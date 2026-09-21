"""
perception/fit.py - systematic start-up fit of the known map to one scan.

THE PRINCIPLE
    Do not trust a guessed pose. Generate every pose the arena allows and score
    each against three INDEPENDENT cues, then take the best total:

      1. WALLS    correlative match of the scan against the wall likelihood
                  field. Strong, but 4-fold ambiguous (a rotated arena is the
                  same arena) and weak along a corridor.
      2. SEATS    subtract the walls; every leftover cluster must sit on a legal
                  traffic-sign seat. An obstacle 900 mm ahead is only consistent
                  with the poses that put a seat 900 mm ahead - THIS is what
                  pins the along-corridor coordinate that walls cannot.
      3. PARKING  two short lines close together hugging an OUTER wall are the
                  magenta blocks, and they exist in exactly one straight.

    Walls say "you are in a corridor, this far from each wall, facing this way".
    Seats say "and you are this far along it". Parking says "and it is this
    straight". Together they are far stronger than any one of them.

WHAT STILL CANNOT BE KNOWN
    Which of the four corridors, in absolute terms, when the scene is fully
    symmetric. Parking narrows 4 -> 2. The last 180 deg needs motion or an
    explicit start. fit() reports the ambiguity instead of hiding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from nav.localize import DistanceField, ScanMatcher
from nav.pillarmap import extract
from nav import arena

# scoring weights: walls are the backbone, seats/parking break the ties
W_SEAT = 0.35
W_PARK = 0.25
SEAT_SIGMA = arena.SNAP_TOL_MM        # how tightly a cluster must hit a seat
TOP_K = 10                            # wall candidates re-scored with semantics

# parking pattern: two short lines hugging an outer wall
PARK_WALL_GAP_MM = 280.0              # "hugging" the outer wall
PARK_SEP_MIN_MM = 140.0               # 1.5 * car_len, for plausible car lengths
PARK_SEP_MAX_MM = 520.0


def canonical_start(side="S", cw=True):
    """Mid-corridor pose at the middle of a given straight, facing the driving
    direction. Because the mat is 4-fold symmetric, naming the straight is a
    free choice; this just gives a consistent frame."""
    cc = arena.CORRIDOR_CENTER
    pos = {"N": (0.0, cc), "E": (cc, 0.0), "S": (0.0, -cc), "W": (-cc, 0.0)}
    cw_head = {"N": 0.0, "E": -90.0, "S": 180.0, "W": 90.0}
    th = cw_head[side] + (0.0 if cw else 180.0)
    th = (th + 180.0) % 360.0 - 180.0
    x, y = pos[side]
    return (x, y, math.radians(th))


# --------------------------------------------------------------------------
def corridor_distances(ranges, half_deg=8):
    """The three numbers a human reads off the scan: how far to the wall ahead,
    and to the wall on each side. Sensor frame, 0 = forward, + = left."""
    r = np.asarray(ranges, dtype=np.float64)

    def sector(centre):
        idx = [(centre + d) % 360 for d in range(-half_deg, half_deg + 1)]
        v = r[idx]
        v = v[np.isfinite(v) & (v > 80)]
        return float(np.min(v)) if v.size else None

    return {"front": sector(0), "left": sector(90),
            "right": sector(270), "back": sector(180)}


def residual_clusters(ranges, pose, field):
    """Scan minus the known walls = the obstacles. [(x, y, n, width_mm)]."""
    return extract(list(ranges), pose, field)


def _wall_gap(x, y):
    """Distance from (x, y) to the nearest OUTER wall."""
    h = arena.OUTER_HALF
    return min(h - y, y + h, h - x, x + h)


def _along_on_side(x, y):
    """Coordinate along the nearest outer wall, for the parking pattern."""
    h = arena.OUTER_HALF
    gaps = {"N": h - y, "S": y + h, "E": h - x, "W": x + h}
    side = min(gaps, key=lambda k: gaps[k])
    return side, (x if side in ("N", "S") else y)


def find_parking(clusters, car_len_mm=175.0):
    """Two clusters hugging the same outer wall, 1.5 car-lengths apart, are the
    magenta blocks. Returns (ParkingBay, {ids of clusters used}) or (None, set()).
    """
    hug = []
    for i, c in enumerate(clusters):
        x, y = c[0], c[1]
        if _wall_gap(x, y) <= PARK_WALL_GAP_MM:
            side, along = _along_on_side(x, y)
            hug.append((i, side, along))
    best = None
    for a in range(len(hug)):
        for b in range(a + 1, len(hug)):
            ia, sa, aa = hug[a]
            ib, sb, ab = hug[b]
            if sa != sb:
                continue
            sep = abs(aa - ab)
            if PARK_SEP_MIN_MM <= sep <= PARK_SEP_MAX_MM:
                if best is None or sep < best[0]:
                    best = (sep, sa, 0.5 * (aa + ab), {ia, ib})
    if best is None:
        return None, set()
    sep, side, centre, used = best
    # the gap between the blocks IS the bay; derive the car length it implies
    return arena.parking_bay(side, centre, sep / 1.5), used


def score_seats(clusters, skip=frozenset()):
    """How well the leftover obstacles land on legal seats.

    Returns (score 0..1, hits, orphans). A cluster that snaps tightly to a seat
    is evidence FOR this pose; one floating in open corridor is evidence against.
    """
    used = [c for i, c in enumerate(clusters) if i not in skip]
    if not used:
        return 0.0, 0, 0
    vals, hits, orphans = [], 0, 0
    for (x, y, *_rest) in used:
        seat, d = arena.nearest_seat(x, y, tol=1e9)
        vals.append(math.exp(-(d / SEAT_SIGMA) ** 2))
        if d <= arena.SNAP_TOL_MM:
            hits += 1
        else:
            orphans += 1
    return float(np.mean(vals)), hits, orphans


# --------------------------------------------------------------------------
@dataclass
class FitResult:
    pose: tuple
    side: str
    wall: float
    seat: float
    park: float
    total: float
    hits: int
    orphans: int
    parking: object = None
    clusters: list = dc_field(default_factory=list)
    ambiguous_with: list = dc_field(default_factory=list)   # equal-scoring rotations
    notes: list = dc_field(default_factory=list)

    def explain(self):
        out = [f"pose ({self.pose[0]:.0f}, {self.pose[1]:.0f}, "
               f"{math.degrees(self.pose[2]):.1f} deg) on straight {self.side}",
               f"  walls   {self.wall:.2f}",
               f"  seats   {self.seat:.2f}  ({self.hits} on-seat, {self.orphans} orphan)",
               f"  parking {self.park:.2f}" + ("  FOUND" if self.parking else ""),
               f"  TOTAL   {self.total:.2f}"]
        if self.ambiguous_with:
            out.append("  ambiguous with: " + ", ".join(self.ambiguous_with))
        out += ["  " + n for n in self.notes]
        return "\n".join(out)


def fit(ranges, field: DistanceField, matcher: ScanMatcher,
        cw=None, car_len_mm=175.0, sensor_ahead=0.0,
        sides=("N", "E", "S", "W"), along_step=50, along_span=700):
    """Two-pass systematic fit.

    Pass A sweeps every straight, both driving senses, and every along-corridor
    offset, scoring on WALLS only (cheap). Pass B takes the best few and adds
    the SEAT and PARKING cues, which is what actually decides.
    """
    r = list(ranges)
    senses = (True, False) if cw is None else (cw,)

    # ---- pass A: wall score over all hypotheses ----
    cand = []
    for side in sides:
        for sense in senses:
            bx, by, bth = canonical_start(side, sense)
            ax, ay = math.cos(bth), math.sin(bth)
            for d in range(-along_span, along_span + 1, along_step):
                sc, x, y, th = matcher.match(r, bx + d * ax, by + d * ay, bth,
                                             sensor_ahead)
                if sc > 0.0:
                    cand.append((sc, side, (x, y, th)))
    if not cand:
        return None
    cand.sort(key=lambda c: c[0], reverse=True)

    # de-duplicate near-identical poses, keep the best TOP_K distinct ones
    picked = []
    for sc, side, pose in cand:
        if all(math.hypot(pose[0] - p[2][0], pose[1] - p[2][1]) > 120.0
               or side != p[1] for p in picked):
            picked.append((sc, side, pose))
        if len(picked) >= TOP_K:
            break

    # ---- pass B: add the semantic cues ----
    results = []
    for sc, side, pose in picked:
        clusters = residual_clusters(r, pose, field)
        bay, used = find_parking(clusters, car_len_mm)
        seat, hits, orphans = score_seats(clusters, skip=used)
        park = 1.0 if bay is not None else 0.0
        total = sc + W_SEAT * seat + W_PARK * park
        results.append(FitResult(pose=pose, side=side, wall=sc, seat=seat,
                                 park=park, total=total, hits=hits,
                                 orphans=orphans, parking=bay,
                                 clusters=clusters))
    results.sort(key=lambda f: f.total, reverse=True)
    best = results[0]

    # be honest about rotations that scored the same
    for other in results[1:]:
        if abs(other.total - best.total) < 0.02 and other.side != best.side:
            best.ambiguous_with.append(other.side)
    if not best.clusters:
        best.notes.append("no obstacles in view - along-track rests on the "
                          "wall/corner geometry alone")
    if best.parking is None:
        best.notes.append("parking not in view - straight identity unconfirmed")
    return best
