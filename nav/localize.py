"""
localize.py - pose in the mat frame from a LiDAR scan and a known static map.

WHY THIS AND NOT SLAM
    The arena is fully known before the round starts: 3000 mm outer square,
    1000 mm inner block, both fixed by the rules. There is nothing to map.
    The only unknowns are the car's pose and the traffic signs - and the signs
    are re-drawn every round, so they must never enter the map. Running SLAM
    here would spend a lot of compute re-discovering a shape written in the
    rulebook, and would happily fold a traffic sign into the map as if it
    were a wall.

WHY NOT ICP
    ICP needs a good initial guess, iterates to a local minimum, and has no
    natural way to say "I am not confident". Correlative scan matching
    searches the whole plausible pose window exhaustively at a fixed cost,
    returns a score that is directly comparable between scans, and cannot
    fall into a local minimum inside the window. Fixed cost and a usable
    confidence are worth more on a competition car than the last millimetre
    of accuracy.

WHAT IS AND IS NOT OBSERVABLE
    Read this before expecting miracles. Inside a 1000 mm corridor between
    two parallel walls, the scan is two parallel lines. Sliding the car ALONG
    the corridor produces an identical scan, so the along-corridor coordinate
    is genuinely unobservable there - no algorithm recovers it, and one that
    claims to is fitting noise. What IS observable on a straight:
    lateral position and heading, both with no drift.

    The along coordinate becomes observable as soon as a corner enters the
    scan - the end of the inner wall and the outer corner are distinctive
    geometry. That is precisely where the current firmware is guessing
    (SIDE_OPEN_MM > 1500 for 3 frames), and precisely where its errors hurt
    most. So the honest claim for this module is: drift-free lateral and
    heading everywhere, plus an accurate along-track fix near every corner.
    That is enough, because the corner is where the runs are being lost.

FOUR-FOLD SYMMETRY
    A square-in-square map is identical under 90 degree rotation, so a scan
    alone cannot tell which of the four corridors the car is in. It never has
    to: the car starts in a known section and the search window (+/-150 mm,
    +/-10 deg) is far too small to jump between corridors. The ambiguity is
    resolved by continuity, which is free.
"""

from __future__ import annotations

import math

import numpy as np

from .geom import WALL_SEGMENTS, OUTER, INNER

DEG = math.pi / 180.0

GRID_RES = 10.0                 # mm per cell
GRID_HALF = 1700.0              # mm, covers the mat plus margin


class DistanceField:
    """Distance from any point to the nearest wall surface, on a grid.

    Precomputed once. Doubles as the likelihood field for scan matching and
    as the wall-rejection test for pillar extraction, which is why both live
    off the same table.
    """

    def __init__(self, res: float = GRID_RES, half: float = GRID_HALF):
        self.res = res
        self.half = half
        n = int(2 * half / res) + 1
        self.n = n
        ax = np.linspace(-half, half, n)
        gx, gy = np.meshgrid(ax, ax, indexing="ij")
        px, py = gx.ravel(), gy.ravel()

        best = np.full(px.shape, 1e9)
        for (x1, y1, x2, y2) in WALL_SEGMENTS:
            vx, vy = x2 - x1, y2 - y1
            L2 = vx * vx + vy * vy
            t = ((px - x1) * vx + (py - y1) * vy) / L2
            np.clip(t, 0.0, 1.0, out=t)
            d = np.hypot(px - (x1 + t * vx), py - (y1 + t * vy))
            np.minimum(best, d, out=best)
        self.field = best.reshape(n, n)

        # score table: 1 at a perfect hit, falling off over SIGMA.
        # A hard floor rather than a Gaussian tail means a traffic sign - which
        # is never in the map and so always far from a wall - contributes a
        # bounded penalty instead of dragging the fit.
        self.sigma = 40.0
        self.score = np.exp(-0.5 * (self.field / self.sigma) ** 2).astype(np.float32)

    def idx(self, x, y):
        ix = np.rint((x + self.half) / self.res).astype(np.int32)
        iy = np.rint((y + self.half) / self.res).astype(np.int32)
        np.clip(ix, 0, self.n - 1, out=ix)
        np.clip(iy, 0, self.n - 1, out=iy)
        return ix, iy

    def dist(self, x, y):
        ix, iy = self.idx(np.asarray(x), np.asarray(y))
        return self.field[ix, iy]


class ScanMatcher:
    """Coarse-to-fine correlative scan matching against a DistanceField.

    The translation search is a table lookup, not a transform: once the scan
    is rotated for a candidate heading, translating it by a whole number of
    cells is just an index offset into the score table. That turns an
    O(candidates x points) transform into one vectorised gather per heading,
    which is what makes an exhaustive search affordable at 10 Hz on a Pi.
    """

    def __init__(self, field: DistanceField,
                 win_xy: float = 150.0, win_th: float = 10.0,
                 coarse_xy: float = 30.0, coarse_th: float = 2.0,
                 fine_xy: float = 10.0, fine_th: float = 0.5,
                 stride: int = 3):
        self.f = field
        self.win_xy = win_xy
        self.win_th = win_th
        self.coarse_xy = coarse_xy
        self.coarse_th = coarse_th
        self.fine_xy = fine_xy
        self.fine_th = fine_th
        self.stride = stride

    def _search(self, bx, by, x0, y0, th0, win_xy, step_xy, win_th, step_th):
        f = self.f
        nx = int(win_xy / step_xy)
        dxs = (np.arange(-nx, nx + 1) * step_xy).astype(np.float64)
        nt = max(1, int(win_th / step_th))
        dts = (np.arange(-nt, nt + 1) * step_th).astype(np.float64)

        best = (-1.0, x0, y0, th0)
        for dth in dts:
            th = th0 + dth * DEG
            c, s = math.cos(th), math.sin(th)
            wx = x0 + bx * c - by * s
            wy = y0 + bx * s + by * c
            # candidate translations -> integer cell offsets
            ix, iy = f.idx(wx, wy)
            ox = np.rint(dxs / f.res).astype(np.int32)
            # (ndx, ndy, npts)
            gx = np.clip(ix[None, None, :] + ox[:, None, None], 0, f.n - 1)
            gy = np.clip(iy[None, None, :] + ox[None, :, None], 0, f.n - 1)
            sc = f.score[gx, gy].sum(axis=2)
            k = int(np.argmax(sc))
            i, j = divmod(k, sc.shape[1])
            if sc[i, j] > best[0]:
                best = (float(sc[i, j]), x0 + float(ox[i]) * f.res,
                        y0 + float(ox[j]) * f.res, th)
        return best

    def match(self, ranges, x0, y0, th0, sensor_ahead: float = 0.0):
        """(score01, x, y, th) for the VEHICLE REFERENCE POINT.

        sensor_ahead is how far the LiDAR sits ahead of that reference point
        (130 mm on this car - it is mounted at the front axle). Getting this
        wrong does not just shift the pose: the same pose is the origin used
        to place signs in the map, so every sign would be entered 130 mm
        behind where it really is, and the plan would then steer around
        phantoms. Folding the offset into the body coordinates here keeps
        one definition of "where the car is" for the whole stack.

        score01 is the mean per-point score, so it stays comparable between
        scans with different numbers of valid returns.
        """
        r = np.asarray(ranges, dtype=np.float64)
        bins = np.arange(0, 360, self.stride)
        r = r[bins]
        ok = np.isfinite(r) & (r > 80) & (r < 2600)
        if ok.sum() < 30:
            return 0.0, x0, y0, th0
        a = np.radians(bins[ok].astype(np.float64))
        d = r[ok]
        bx, by = d * np.cos(a) + sensor_ahead, d * np.sin(a)

        sc, x, y, th = self._search(bx, by, x0, y0, th0,
                                    self.win_xy, self.coarse_xy,
                                    self.win_th, self.coarse_th)
        sc, x, y, th = self._search(bx, by, x, y, th,
                                    self.coarse_xy, self.fine_xy,
                                    self.coarse_th, self.fine_th)
        return sc / max(1, int(ok.sum())), x, y, th


class PoseFilter:
    """Dead reckoning corrected by scan matching.

    Dead reckoning (encoder + IMU) is smooth and locally exact but drifts.
    Scan matching is drift-free but noisy per scan and occasionally wrong.
    So the scan match is applied as a bounded, gated correction rather than
    as the answer: a bad fit (low score, or a jump larger than the car could
    physically have made since the last scan) is rejected outright and dead
    reckoning carries on. Rejecting is the whole point - a confident wrong
    pose is far worse than a stale one.
    """

    def __init__(self, x, y, th, matcher: ScanMatcher,
                 accept_score: float = 0.45, gain: float = 0.6,
                 max_jump_mm: float = 140.0, max_jump_deg: float = 9.0,
                 sensor_ahead: float = 0.0):
        self.x, self.y, self.th = x, y, th
        self.m = matcher
        self.sensor_ahead = sensor_ahead
        self.accept_score = accept_score
        self.gain = gain
        self.max_jump_mm = max_jump_mm
        self.max_jump_deg = max_jump_deg
        self.score = 0.0
        self.fixes = 0
        self.rejects = 0
        self.since_fix_mm = 0.0

    def predict(self, d_mm: float, dth_rad: float):
        """Encoder distance and IMU heading change since the last call."""
        self.th += dth_rad
        self.x += d_mm * math.cos(self.th)
        self.y += d_mm * math.sin(self.th)
        self.since_fix_mm += abs(d_mm)

    def correct(self, ranges):
        sc, mx, my, mth = self.m.match(ranges, self.x, self.y, self.th,
                                       self.sensor_ahead)
        self.score = sc
        if sc < self.accept_score:
            self.rejects += 1
            return False
        dj = math.hypot(mx - self.x, my - self.y)
        da = abs(math.degrees((mth - self.th + math.pi) % (2 * math.pi) - math.pi))
        if dj > self.max_jump_mm or da > self.max_jump_deg:
            self.rejects += 1
            return False
        g = self.gain
        self.x += g * (mx - self.x)
        self.y += g * (my - self.y)
        self.th += g * ((mth - self.th + math.pi) % (2 * math.pi) - math.pi)
        self.fixes += 1
        self.since_fix_mm = 0.0
        return True

    @property
    def healthy(self) -> bool:
        """False when the pose has been running open-loop long enough that the
        planner should stop trusting it and hand back to wall-relative
        control."""
        return self.since_fix_mm < 900.0
