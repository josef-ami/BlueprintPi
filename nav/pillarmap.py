"""
pillarmap.py - traffic signs as a persistent GLOBAL map, not per-frame tracks.

THE IDEA
    Every LiDAR return either belongs to a wall, whose position is known from
    the rulebook, or it does not. Subtract the known walls from the scan and
    what is left IS the traffic signs. No corridor heuristics, no cluster
    width guessing against a fitted line that a sign might have corrupted in
    the first place - the map is exact, so the residual is clean.

WHY THIS FIXES THE AFTER-CORNER FAILURE
    The current firmware stores a pillar as (along, lat) in a LANE frame that
    is destroyed and rebuilt at every corner, so turningStep() has no choice
    but to clearTracks(): the coordinates would be meaningless afterwards.
    That is why a pillar sitting 250 mm past a corner has to be rediscovered
    from zero, needs TRACK_CONFIRM fresh sightings at 10 Hz, and is often
    still unconfirmed when the car reaches it.

    Here a sign is stored at (x, y) in the MAT frame. A corner does not
    change a mat-frame coordinate. A sign seen from 1.5 m away on the
    previous straight is already in the map, already confirmed, and already
    has a colour when the car comes round the corner. Nothing is thrown away
    and nothing needs re-confirming.

    The same property means the map only improves: by lap 2 the car has seen
    most of the course and is driving a known problem.

COLOUR
    The camera no longer has to locate anything. Position comes from the
    LiDAR, which is accurate and has 360 degrees of coverage; the camera only
    has to answer "what colour is the thing over there", which it can do from
    a single bearing. That removes the camera's ~885 mm range limit and its
    +/-48 deg FoV from the POSITION problem, and leaves them affecting only
    the labelling problem - where being late is recoverable, because an
    unlabelled sign is still a known obstacle to be avoided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geom import PILLAR

RED, GREEN, UNKNOWN = 1, 0, 2

WALL_REJECT_MM = 75.0        # a return this close to a mapped wall IS the wall
MAX_RANGE_MM = 2600.0
MIN_RANGE_MM = 90.0
CLUSTER_GAP_MM = 70.0
CLUSTER_MAX_W_MM = 130.0
CLUSTER_MIN_PTS = 2
MATCH_MM = 150.0
CONFIRM_HITS = 3
FACE_TO_CENTRE_MM = PILLAR / 2.0


@dataclass
class GlobalPillar:
    x: float
    y: float
    color: int = UNKNOWN
    hits: int = 0
    colour_votes: dict = field(default_factory=dict)
    last_seen_s: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.hits >= CONFIRM_HITS

    def vote(self, c: int):
        self.colour_votes[c] = self.colour_votes.get(c, 0) + 1
        best = max(self.colour_votes.items(), key=lambda kv: kv[1])
        # a single stray frame must not repaint a sign that has been seen
        # consistently; require a clear margin to switch
        if best[0] != self.color:
            cur = self.colour_votes.get(self.color, 0)
            if best[1] >= cur + 2:
                self.color = best[0]


def extract(ranges, pose, field_):
    """Map-subtracted sign detections, in the mat frame.

    pose  : (x, y, th)
    field_: DistanceField
    returns [(x, y, n_points, width_mm)]
    """
    x0, y0, th = pose
    r = np.asarray(ranges, dtype=np.float64)
    bins = np.arange(360)
    ok = np.isfinite(r) & (r > MIN_RANGE_MM) & (r < MAX_RANGE_MM)
    if ok.sum() < 5:
        return []
    idx = bins[ok]
    d = r[ok]
    a = np.radians(idx.astype(np.float64)) + th
    px = x0 + d * np.cos(a)
    py = y0 + d * np.sin(a)

    # --- subtract the known map ---
    keep = field_.dist(px, py) > WALL_REJECT_MM
    if keep.sum() < CLUSTER_MIN_PTS:
        return []
    idx, px, py, d = idx[keep], px[keep], py[keep], d[keep]

    # --- cluster what is left ---
    brk = np.ones(len(idx), dtype=bool)
    if len(idx) > 1:
        gap_bin = np.diff(idx) > 3
        gap_xy = np.hypot(np.diff(px), np.diff(py)) > CLUSTER_GAP_MM
        brk[1:] = gap_bin | gap_xy
    groups = np.split(np.arange(len(idx)), np.nonzero(brk)[0][1:])

    # wrap 359 -> 0
    if len(groups) > 1 and idx[0] + 360 - idx[-1] <= 3 and \
            math.hypot(px[0] - px[-1], py[0] - py[-1]) < CLUSTER_GAP_MM:
        groups[0] = np.concatenate([groups[-1], groups[0]])
        groups.pop()

    out = []
    for g in groups:
        if len(g) < CLUSTER_MIN_PTS:
            continue
        cx, cy = px[g], py[g]
        w = math.hypot(cx.max() - cx.min(), cy.max() - cy.min())
        if w > CLUSTER_MAX_W_MM:
            continue
        mx, my = float(cx.mean()), float(cy.mean())
        # the LiDAR sees the near face; push the centre away from the sensor
        vx, vy = mx - x0, my - y0
        n = math.hypot(vx, vy)
        if n < 1.0:
            continue
        out.append((mx + FACE_TO_CENTRE_MM * vx / n,
                    my + FACE_TO_CENTRE_MM * vy / n, len(g), w))
    return out


class PillarMap:
    def __init__(self):
        self.pillars: list[GlobalPillar] = []
        # Bumped whenever something the PLAN depends on changes: a sign
        # becoming confirmed, or a confirmed sign changing colour. Position
        # refinements of a few mm are deliberately NOT counted - re-solving a
        # 300 ms plan ten times a second because a centroid moved 2 mm is
        # what made the replan the most expensive thing in the stack.
        self.version = 0

    def update(self, dets, t_s: float, pose_healthy: bool = True):
        """Fold detections into the map. Nothing is added while the pose is
        untrusted - a sign entered at a wrong global position is worse than a
        missing one, because it persists."""
        if not pose_healthy:
            return
        for (x, y, n, w) in dets:
            hit = None
            best = MATCH_MM
            for p in self.pillars:
                dd = math.hypot(p.x - x, p.y - y)
                if dd < best:
                    best, hit = dd, p
            if hit is None:
                self.pillars.append(GlobalPillar(x=x, y=y, hits=1, last_seen_s=t_s))
            else:
                was = hit.confirmed
                k = 1.0 / min(hit.hits + 1, 12)
                hit.x += k * (x - hit.x)
                hit.y += k * (y - hit.y)
                hit.hits += 1
                hit.last_seen_s = t_s
                if hit.confirmed and not was:
                    self.version += 1

    def label(self, cam_x, cam_y, cam_th, detections, tol_deg=9.0,
              max_mm=1400.0):
        """Attach camera colours to mapped signs by bearing.

        detections: [(colour, bearing_deg)] in the camera frame, + = left.
        The camera only has to point at the right thing - it never has to
        say how far away it is.
        """
        for (col, bearing) in detections:
            if col == UNKNOWN:
                continue
            want = cam_th + math.radians(bearing)
            # Among the signs on this bearing, the NEAREST one is what the
            # camera saw - anything behind it is occluded. Picking the
            # smallest angular error instead lets a distant sign steal the
            # label from the one actually in view, which is how a green ends
            # up mapped as red.
            bp, bd = None, 1e9
            for p in self.pillars:
                dx, dy = p.x - cam_x, p.y - cam_y
                dist = math.hypot(dx, dy)
                if dist > max_mm or dist < 60:
                    continue
                da = abs(math.degrees(
                    (math.atan2(dy, dx) - want + math.pi) % (2 * math.pi) - math.pi))
                # allow the sign's own angular half-width on top of the
                # pointing tolerance, so a close sign is not rejected for
                # being wide
                allow = tol_deg + math.degrees(math.atan2(25.0, max(dist, 60.0)))
                if da <= allow and dist < bd:
                    bd, bp = dist, p
            if bp is not None:
                before = bp.color
                bp.vote(col)
                if bp.color != before and bp.confirmed:
                    self.version += 1

    def confirmed(self):
        return [p for p in self.pillars if p.confirmed]

    def prune(self, t_s: float, unconfirmed_ttl: float = 3.0):
        self.pillars = [p for p in self.pillars
                        if p.confirmed or (t_s - p.last_seen_s) < unconfirmed_ttl]
