"""
perception/state.py - the world BELIEF: one object that folds the pose fix, the
per-seat pillar state (present / empty / colour), and the parking bay into a
single snapshot the dashboard streams and a planner will later read.

It reuses the existing pieces rather than re-implementing them:
    - nav.localize.ScanMatcher / DistanceField  -> the pose
    - nav.pillarmap.extract + PillarMap          -> free pillar detections,
      confirmation voting, and camera-colour labelling
and adds the semantic layer the user asked for on top:
    - each rulebook SEAT carries a log-odds occupancy belief, so a seat the
      LiDAR has clearly seen PAST becomes "empty", a seat with a confirmed
      pillar on it becomes "red"/"green", and an unseen seat stays "unknown".

Belief update is incremental ("update the matrix every time new info comes"):
call update() with each scan (and any camera detections) and the seat states
sharpen over time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as dc_field

import numpy as np

from nav.localize import DistanceField, ScanMatcher, PoseFilter
from nav.pillarmap import PillarMap, extract, RED, GREEN, UNKNOWN
from nav import arena, parking as parking_mod
from perception.tower import find_towers, tower_points

_COLOR_NAME = {RED: "red", GREEN: "green", UNKNOWN: "unknown"}

MIN_VALID = 30       # a scan with fewer usable returns is a spin-up/dropout frame


def valid_count(ranges):
    r = np.asarray(ranges, dtype=np.float64)
    return int((np.isfinite(r) & (r > 80) & (r < 2600)).sum())

SEAT_RADIUS_MM = 60.0        # how close a return must be to the seat centre
CLEAR_MARGIN_MM = 70.0       # seen this far past a seat -> nothing there
OCC_MAX_RANGE_MM = 2600.0
LOGODDS_HIT = 0.85
LOGODDS_MISS = 0.55
LOGODDS_CLAMP = 4.0
OCC_DECIDE = 1.2             # |log-odds| above this = a decided seat
TOWER_MATCH_MM = 130.0       # a seat detection must coincide with a valley


@dataclass
class SeatBelief:
    seat: arena.Seat
    logodds: float = 0.0          # + = occupied, - = empty
    color: int = UNKNOWN

    @property
    def state(self) -> str:
        if self.logodds > OCC_DECIDE:
            return _COLOR_NAME.get(self.color, "pillar") if self.color != UNKNOWN else "pillar"
        if self.logodds < -OCC_DECIDE:
            return "empty"
        return "unknown"

    @property
    def confidence(self) -> float:
        return float(min(1.0, abs(self.logodds) / LOGODDS_CLAMP))


from perception.fit import (canonical_start, fit as _fit, corridor_distances,
                            find_parking as fit_parking,
                            find_parking_points as fit_parking_points,
                            belongs_to_parking)  # noqa: E402


class WorldBelief:
    def __init__(self, field: DistanceField | None = None,
                 sensor_ahead: float = 0.0):
        self.field = field or DistanceField()
        self.matcher = ScanMatcher(self.field)
        self.sensor_ahead = sensor_ahead
        self.pillars = PillarMap()
        self.seats = {s.id: SeatBelief(s) for s in arena.SEATS}
        # wide matcher for the first fix / static drop: the car may be anywhere
        # along a 1 m straight, so a +/-150 mm window (fine for tracking) can't
        # reach it. +/-700 mm covers a whole straight section.
        self.wide = ScanMatcher(self.field, win_xy=700.0, win_th=15.0,
                                coarse_xy=40.0, coarse_th=3.0,
                                fine_xy=10.0, fine_th=0.5)
        self.filter = PoseFilter(0.0, 0.0, 0.0, self.matcher,
                                 sensor_ahead=sensor_ahead)
        self.score = 0.0
        self.parking = None            # arena.ParkingBay once located
        self.unclassified = []         # non-seat detections (parking/noise)
        self.last_fit = None           # perception.fit.FitResult from fit_start
        self.towers = []               # free-standing valleys, mat frame
        self._t0 = time.time()

    @property
    def pose(self):
        return (self.filter.x, self.filter.y, self.filter.th)

    def auto_init(self, ranges, cw=True, sides=("N", "E", "S", "W")):
        """Self-localize with NO numeric guess - only the driving direction.

        A single static scan of a bare square-in-square corridor is 4-fold
        ambiguous: the walls look identical in all four corridors. The parking
        bay breaks that tie because it exists in only ONE straight. So we try
        the canonical start pose of each candidate straight, and prefer the one
        whose scan actually yields two magenta blocks hugging the outer wall.
        Score breaks any remaining tie. Returns (side, score, parking_seen).
        """
        if valid_count(ranges) < MIN_VALID:
            return None, 0.0, False
        r = list(ranges)
        cands = []
        for side in sides:
            bx, by, bth = canonical_start(side, cw)
            ax, ay = math.cos(bth), math.sin(bth)
            # sweep along the corridor, try both driving senses (front/back)
            seeds = [(bx + d * ax, by + d * ay, bth + dh)
                     for d in range(-700, 701, 100) for dh in (0.0, math.pi)]
            sc, x, y, th = self._grid_match(r, seeds)
            dets = extract(r, (x, y, th), self.field)
            unc = [d for d in dets if arena.nearest_seat(d[0], d[1])[0] is None]
            bay = parking_mod.detect(unc)
            ahead = 0
            if bay is not None:
                # is the parking in the car's forward hemisphere? (the parking
                # narrows 4 corridors to 2; "placed in front of the bay, facing
                # the driving direction" breaks the remaining 180 deg flip)
                cx, cy = 0.5 * (bay.rect[0] + bay.rect[2]), 0.5 * (bay.rect[1] + bay.rect[3])
                fwd = (cx - x) * math.cos(th) + (cy - y) * math.sin(th)
                ahead = 1 if fwd > 0 else 0
            cands.append((1 if bay is not None else 0, ahead, sc, side, (x, y, th), bay))
        # prefer: parking seen -> parking ahead -> best wall-match score
        cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
        park, ahead, sc, side, (x, y, th), bay = cands[0]
        self.filter.x, self.filter.y, self.filter.th = x, y, th
        self.score = sc
        if bay is not None:
            self.parking = bay
        return side, sc, bool(park)

    @property
    def healthy(self) -> bool:
        return self.filter.healthy

    # ---- localization ----
    def fit_start(self, ranges, cw=None, car_len_mm=175.0,
                  sides=("N", "E", "S", "W")):
        """Systematic start-up fit: score every allowed pose on walls + seats +
        parking and take the best. This is the recommended first fix - it needs
        no guess, works out the driving direction, and uses the obstacles
        themselves to pin the along-corridor position.

        Returns the fit.FitResult (or None if the scan was unusable).
        """
        if valid_count(ranges) < MIN_VALID:
            return None
        res = _fit(ranges, self.field, self.matcher, cw=cw,
                   car_len_mm=car_len_mm, sensor_ahead=self.sensor_ahead,
                   sides=sides)
        if res is None:
            return None
        self.filter.x, self.filter.y, self.filter.th = res.pose
        self.score = res.wall
        if res.parking is not None:
            self.parking = res.parking
        self.last_fit = res
        return res

    def _grid_match(self, ranges, seeds):
        """Tight correlative match from every seed; return the best (sc,x,y,th).
        A dense grid of tight matches is far more reliable on a partial/occluded
        real scan than one match with a huge window (which lands in local maxima).
        """
        r = list(ranges)
        best = (-1.0, 0.0, 0.0, 0.0)
        for (x, y, th) in seeds:
            sc, X, Y, T = self.matcher.match(r, x, y, th, self.sensor_ahead)
            if sc > best[0]:
                best = (sc, X, Y, T)
        return best

    def global_init(self, ranges, guess=None):
        """First fix from a rough `guess`. The unobservable axis is ALONG the
        corridor, so sweep seeds down the heading direction (+/-700 mm) and let
        each tight match recover lateral + heading. This tolerates a large
        along-track error in the guess, which a single window cannot."""
        if valid_count(ranges) < MIN_VALID:
            return self.pose, 0.0
        gx, gy, gth = guess if guess is not None else self.pose
        seeds = [(gx + d * math.cos(gth), gy + d * math.sin(gth), gth)
                 for d in range(-700, 701, 100)]
        sc, x, y, th = self._grid_match(ranges, seeds)
        self.score = sc
        if sc > 0.0:
            self.filter.x, self.filter.y, self.filter.th = x, y, th
        return self.pose, sc

    def track(self, ranges):
        """LiDAR-only frame update: an ungated correlative match seeded at the
        current pose. This is the primary estimator here - pose comes from the
        LiDAR (and camera colour) ALONE, deliberately without IMU/encoder.

        It is honest about what a scan can give: lateral offset and heading are
        pinned every frame; the along-corridor coordinate is only observable
        near a corner or when a mapped pillar is in view, so mid-straight it is
        rough and can lag. That roughness is expected - the encoder + IMU are
        meant to sharpen it later (see predict()), not to be needed for it.

        Uses the TIGHT matcher window around the current pose so a good static
        lock is only nudged by noise, not dragged across the corridor. The wide
        window is only for the first fix (global_init / auto_init).
        """
        if valid_count(ranges) < MIN_VALID:
            return self.pose, self.score       # skip spin-up/dropout frames
        x0, y0, th0 = self.pose
        sc, x, y, th = self.matcher.match(list(ranges), x0, y0, th0,
                                          self.sensor_ahead)
        self.score = sc
        if sc > 0.0:
            self.filter.x, self.filter.y, self.filter.th = x, y, th
        return self.pose, sc

    # ---- optional LATER sharpening from IMU + encoder (not used by default) --
    def predict(self, d_mm, dth_rad):
        """OPTIONAL, for later: dead-reckon between scans from encoder distance
        + IMU heading change. Not part of the LiDAR/camera-only estimate; wire
        it in once the encoder/IMU are trusted, then use correct() instead of
        track() to fold the scan in as a gated correction on top of odometry."""
        self.filter.predict(d_mm, dth_rad)

    def correct(self, ranges):
        """Gated scan-match correction, for use AFTER predict() when odometry
        is available. Rejects low score / physically impossible jumps."""
        ok = self.filter.correct(list(ranges))
        self.score = self.filter.score
        return ok

    def localize(self, ranges, guess=None):
        """Convenience: global_init when a guess is supplied (first/static
        fix), else a LiDAR-only track()."""
        if guess is not None:
            return self.global_init(ranges, guess)
        return self.track(ranges)

    # ---- semantic update ----
    def update(self, ranges, cam_dets=None, t_s=None, pose_healthy=True):
        """Fold one scan (+ optional camera colour detections) into the belief.

        cam_dets: [(colour_int, bearing_deg)] in the camera frame, + = left.
        """
        t_s = time.time() - self._t0 if t_s is None else t_s
        r = np.asarray(ranges, dtype=np.float64)

        # 1) free pillar detections -> confirmation map (existing machinery).
        # In this map-based design a legal sign only ever stands on a SEAT, so
        # keep detections that snap to one; the rest (parking blocks, noise)
        # are held separately for the parking detector and debugging.
        dets_all = extract(list(r), self.pose, self.field)

        # PARKING IS CLAIMED FIRST. The two magenta blocks sit against the outer
        # wall and, seen end-on, look exactly like a pair of small pillars - so
        # if seats got first refusal they would swallow the blocks and the bay
        # would never be found. Claim the parking pattern, then let the seats
        # have whatever is genuinely left.
        car_len = self.parking.car_len_mm if self.parking else 175.0
        bay = fit_parking_points(list(r), self.pose, self.field, car_len)
        if bay is None:                      # fall back to the centroid pattern
            bay, _used = fit_parking(dets_all, car_len)
        if bay is not None and pose_healthy:
            self.parking = bay
        rest = [d for d in dets_all
                if not belongs_to_parking(d[0], d[1], self.parking)]

        # A seat detection must ALSO be a free-standing valley in the raw scan
        # (LazyGo's tower test: depth > 200 mm, s = r*theta ~ 50 mm). That test
        # uses no map and no pose, so it still holds when the pose is shaky -
        # which is exactly when map-subtraction invents pillars out of wall
        # residue. It is also what keeps the parking blocks out: flush to the
        # wall, their valley is too shallow to qualify.
        tpts = tower_points(list(r), self.pose, find_towers(list(r)))
        self.towers = tpts

        def _free_standing(d):
            return any(math.hypot(d[0] - tx, d[1] - ty) < TOWER_MATCH_MM
                       for (tx, ty) in tpts)

        dets = [d for d in rest
                if arena.nearest_seat(d[0], d[1])[0] is not None
                and _free_standing(d)]
        self.unclassified = [d for d in rest
                             if arena.nearest_seat(d[0], d[1])[0] is None]
        self.pillars.update(dets, t_s, pose_healthy=pose_healthy)
        if cam_dets:
            x0, y0, th = self.pose
            self.pillars.label(x0, y0, th, cam_dets)

        # 2) per-seat occupancy log-odds from this scan
        self._update_seats(r)


        # 3) attach colours + occupancy from confirmed pillars to their seats
        for p in self.pillars.confirmed():
            seat, d = arena.nearest_seat(p.x, p.y)
            if seat is not None:
                sb = self.seats[seat.id]
                sb.logodds = min(LOGODDS_CLAMP, max(sb.logodds, OCC_DECIDE + 0.5))
                if p.color != UNKNOWN:
                    sb.color = p.color

        # 4) one sign per line (see arena.Seat.station). A card never places two
        # signs abreast - the car would have to pass one on the right and the
        # other on the left at the same instant. So if a line shows two, only the
        # stronger can be real; the loser is pushed back to 'unknown'.
        self._enforce_one_per_station()
        return self.snapshot(include_scan=False)

    def _enforce_one_per_station(self):
        for station, seats in arena.SEATS_BY_STATION.items():
            occupied = [self.seats[s.id] for s in seats
                        if self.seats[s.id].logodds > OCC_DECIDE]
            if len(occupied) < 2:
                continue
            occupied.sort(key=lambda sb: sb.logodds, reverse=True)
            for loser in occupied[1:]:
                loser.logodds = 0.0          # back to unknown, not 'empty'
                loser.color = UNKNOWN

    def _update_seats(self, r):
        x0, y0, th = self.pose
        finite = np.isfinite(r)
        if finite.sum() < 10:
            return
        deg = np.arange(360)
        for sb in self.seats.values():
            s = sb.seat
            dx, dy = s.x - x0, s.y - y0
            dist = math.hypot(dx, dy)
            if dist > OCC_MAX_RANGE_MM or dist < 120.0:
                continue
            bearing = (math.degrees(math.atan2(dy, dx) - th)) % 360.0
            # look at the few LiDAR bins around the seat bearing
            lo, hi = int(bearing) - 2, int(bearing) + 3
            bins = [(b % 360) for b in range(lo, hi)]
            vals = r[bins]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            near = float(np.min(vals))
            if abs(near - dist) <= SEAT_RADIUS_MM:
                sb.logodds = min(LOGODDS_CLAMP, sb.logodds + LOGODDS_HIT)   # something at the seat
            elif near > dist + CLEAR_MARGIN_MM:
                sb.logodds = max(-LOGODDS_CLAMP, sb.logodds - LOGODDS_MISS)  # saw past -> empty
            # else: occluded (closer return) -> no evidence

    def set_parking(self, bay):
        self.parking = bay

    # ---- output ----
    def snapshot(self, ranges=None, include_scan=True):
        x, y, th = self.pose
        th_deg = (math.degrees(th) + 180.0) % 360.0 - 180.0
        seats = []
        for sb in self.seats.values():
            seats.append({
                "id": sb.seat.id, "x": round(sb.seat.x, 1), "y": round(sb.seat.y, 1),
                "state": sb.state, "color": _COLOR_NAME.get(sb.color, "unknown"),
                "logodds": round(sb.logodds, 2), "conf": round(sb.confidence, 2),
            })
        snap = {
            "pose": {"x": round(x, 1), "y": round(y, 1), "th_deg": round(th_deg, 2)},
            "score": round(self.score, 3),
            "seats": seats,
            "pillars": [{"x": round(p.x, 1), "y": round(p.y, 1),
                         "color": _COLOR_NAME.get(p.color, "unknown")}
                        for p in self.pillars.confirmed()],
            "arena": {"outer": arena.OUTER, "inner": arena.INNER,
                      "corridor": arena.CORRIDOR},
        }
        if self.parking is not None:
            snap["parking"] = {"side": self.parking.side,
                               "rect": [round(v, 1) for v in self.parking.rect]}
        if self.last_fit is not None:
            f = self.last_fit
            snap["fit"] = {"side": f.side, "wall": round(f.wall, 2),
                           "seat": round(f.seat, 2), "park": round(f.park, 2),
                           "total": round(f.total, 2),
                           "hits": f.hits, "orphans": f.orphans,
                           "ambiguous": f.ambiguous_with}
        if ranges is not None:
            d = corridor_distances(ranges)
            snap["dist"] = {k: (round(v) if v is not None else None)
                            for k, v in d.items()}
        if include_scan and ranges is not None:
            snap["scan"] = scan_points(self.pose, ranges)
        return snap


def scan_points(pose, ranges, stride=2, max_range=OCC_MAX_RANGE_MM):
    """World-frame (x, y) of the scan, for drawing it on the mat."""
    x0, y0, th = pose
    r = np.asarray(ranges, dtype=np.float64)
    idx = np.arange(0, 360, stride)
    r = r[idx]
    ok = np.isfinite(r) & (r > 80) & (r < max_range)
    a = np.radians(idx[ok].astype(np.float64)) + th
    px = x0 + r[ok] * np.cos(a)
    py = y0 + r[ok] * np.sin(a)
    return [[round(float(a), 1), round(float(b), 1)] for a, b in zip(px, py)]
