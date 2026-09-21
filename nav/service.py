"""
nav/service.py - the thread that turns LiDAR scans into a steering command.

SPLIT OF WORK, AND WHY
    on_telemetry()  runs in obstacleRound.py's 50 Hz loop. Dead-reckons the
                    pose from the STM32's heading and encoder, then evaluates
                    the tracker against the current plan. ~60 us. Safe there.

    the thread      waits for a new LiDAR revolution, then does the expensive
                    work: correlative scan matching (~15-30 ms), map-subtracted
                    sign extraction, and - only when the sign map has actually
                    changed - a replan (~200 ms).

    A replan must NEVER run in the sensor loop. At 200 ms it would stall the
    link past the firmware's LIDAR_STALE_MS of 200 and the car would drop to
    its stale-lidar behaviour mid-corner.

WHAT IT SENDS THE FIRMWARE
    navOk      1 when the pose is trustworthy AND a plan exists
    crossMm    cross-track error to the planned line, + = car LEFT of it
    hdgErrDd   heading error to the line tangent, deci-degrees, + = turn left
    curvUm     planned curvature a short preview ahead, 1e-6 per mm, signed
    navDone    1 when three laps of arc length have been travelled

    The firmware needs no map, no plan and no geometry - just three numbers
    and a validity bit. If navOk drops for any reason the firmware falls
    straight back to its own wall-relative planner, continuously, with no
    mode change to get wrong.

HANDOVER AT THE START
    Direction (clockwise vs anticlockwise) is resolved from the POSE, by
    watching whether the station index rises or falls - not from the floor
    colour sensor. That needs ~250 mm of motion, so navOk is false for the
    first stretch and the firmware drives on its own law until then.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from .geom import PILLAR
from .localize import DistanceField, ScanMatcher, PoseFilter
from .pillarmap import PillarMap, extract
from .planner import Path, solve, planned_line, track, LAP_MM, _nearest

DEG = math.pi / 180.0

NAV_NONE = (0, 0, 0, 0, 0)          # navOk, cross, hdgErrDd, curvUm, navDone


class NavService(threading.Thread):
    TARGET_LAPS = 3

    def __init__(self, lidar, ticks_per_mm: float = 1.4853,
                 lidar_ahead_mm: float = 130.0, camera_ahead_mm: float = 150.0,
                 half_w_mm: float = 57.1, start_xy=(0.0, 0.0), start_th=0.0,
                 log=print):
        super().__init__(name="NavService", daemon=True)
        self.lidar = lidar
        self.ticks_per_mm = ticks_per_mm
        self.lidar_ahead = lidar_ahead_mm
        self.camera_ahead = camera_ahead_mm
        self.half_w = half_w_mm
        self.log = log

        self.field = DistanceField()
        self.matcher = ScanMatcher(self.field)
        self.pose = PoseFilter(start_xy[0], start_xy[1], start_th, self.matcher,
                               sensor_ahead=lidar_ahead_mm)
        self.map = PillarMap()

        self.path = None
        self.clockwise = None
        self.lat = None
        self.pts = self.kappa = self.line_th = None
        self.info = {}

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_rev = -1
        self._last_ticks = None
        self._last_head = None
        self._map_version = -1
        self._last_plan_t = -1e9
        self._track_i = None
        self._cam = []

        self._dir_dist = 0.0
        self._dir_votes = 0
        self._dir_last_i = None
        self._s_travelled = 0.0
        self._last_i = None

        self.state = "INIT"
        self.done = False
        self._fields = NAV_NONE
        self.n_matches = 0
        self.n_rejects = 0
        self.n_replans = 0
        self.last_solve_ms = 0.0

    # ------------------------------------------------ called from the 50 Hz loop
    def on_telemetry(self, heading_deg: float, enc_ticks: int):
        """Dead-reckon and evaluate the tracker. Cheap; safe in the main loop."""
        if self._last_ticks is None:
            self._last_ticks = enc_ticks
            self._last_head = heading_deg * DEG
            return
        d_mm = (enc_ticks - self._last_ticks) / self.ticks_per_mm
        self._last_ticks = enc_ticks
        h = heading_deg * DEG
        dth = (h - self._last_head + math.pi) % (2 * math.pi) - math.pi
        self._last_head = h

        with self._lock:
            self.pose.predict(d_mm, dth)
            if self.state == "INIT":
                self._fields = NAV_NONE
                return
            if self.state == "DIRECTION":
                self._resolve_direction(d_mm)
                self._fields = NAV_NONE
                return
            if self.pts is None or not self.pose.healthy:
                self._fields = NAV_NONE
                return

            i, _ = self.path.project(self.pose.x, self.pose.y)
            if self._last_i is not None:
                step = (i - self._last_i) % self.path.n
                if step > self.path.n // 2:
                    step -= self.path.n
                self._s_travelled += step * self.path.ds
            self._last_i = i
            if self._s_travelled >= self.TARGET_LAPS * LAP_MM:
                self.done = True

            delta, self._track_i, cross = track(
                self.path, self.pts, self.kappa, self.line_th,
                self.pose.x, self.pose.y, self.pose.th,
                wheelbase=135.85, hint=self._track_i)

            j = self._track_i
            lt = self.line_th[j]
            hdg = (lt - self.pose.th + math.pi) % (2 * math.pi) - math.pi
            k = self.kappa[(j + 7) % len(self.kappa)]
            self._fields = (1,
                            int(round(max(-9000.0, min(9000.0, cross)))),
                            int(round(max(-1800.0, min(1800.0,
                                                       math.degrees(hdg) * 10)))),
                            int(round(max(-30000.0, min(30000.0, k * 1e6)))),
                            1 if self.done else 0)

    def on_camera(self, detections):
        """detections: [(colour, bearing_deg)], colour 1=red 0=green, +=left."""
        if detections:
            with self._lock:
                self._cam = list(detections)

    def wire_fields(self):
        return self._fields

    def status(self):
        return (f"nav {self.state} pose=({self.pose.x:.0f},{self.pose.y:.0f},"
                f"{math.degrees(self.pose.th):.0f}) q={self.pose.score:.2f} "
                f"signs={len(self.map.confirmed())} laps={self._s_travelled/LAP_MM:.2f} "
                f"fix/rej={self.n_matches}/{self.n_rejects} "
                f"solve={self.last_solve_ms:.0f}ms")

    # --------------------------------------------------------------- the thread
    def run(self):
        while not self._stop.is_set():
            rev = self.lidar.rev
            if rev == self._last_rev:
                time.sleep(0.005)
                continue
            self._last_rev = rev
            try:
                ranges = list(self.lidar._ranges)
            except Exception:
                time.sleep(0.02)
                continue
            self._scan(ranges)

    def _scan(self, ranges):
        t = time.monotonic()
        if self.state == "INIT":
            self._global_init(ranges)
            return

        with self._lock:
            ok = self.pose.correct(ranges)
            if ok:
                self.n_matches += 1
            else:
                self.n_rejects += 1
            px, py, pth = self.pose.x, self.pose.y, self.pose.th
            healthy = self.pose.healthy
            cam = list(self._cam)
            self._cam = []

        if self.state != "RUN":
            return

        c, s = math.cos(pth), math.sin(pth)
        lx, ly = px + self.lidar_ahead * c, py + self.lidar_ahead * s
        dets = extract(ranges, (lx, ly, pth), self.field)
        with self._lock:
            self.map.update(dets, t, pose_healthy=healthy)
            self.map.prune(t)
            if cam:
                self.map.label(px + self.camera_ahead * c,
                               py + self.camera_ahead * s, pth, cam)
        self._maybe_replan(t)

    def _global_init(self, ranges):
        """Wide search for the initial pose.

        The square-in-square map is four-fold symmetric, so there are four
        equally good answers. That is harmless - they differ by a 90 degree
        rotation of the whole world, every sign is mapped into whichever
        frame is picked, and nothing downstream can tell the difference.
        """
        wide = ScanMatcher(self.field, win_xy=700.0, win_th=25.0,
                           coarse_xy=50.0, coarse_th=4.0,
                           fine_xy=10.0, fine_th=0.5, stride=2)
        sc, x, y, th = wide.match(ranges, self.pose.x, self.pose.y,
                                  self.pose.th, self.lidar_ahead)
        # match() already returns a MEAN per-point score in [0, 1]. Dividing
        # by the point count again made this test unreachable, so global
        # initialisation silently never ran. The simulator did not catch it
        # because it seeds the pose at the true start position; on the real
        # car that pose is precisely the unknown.
        if sc > 0.35:
            with self._lock:
                self.pose.x, self.pose.y, self.pose.th = x, y, th
            self.state = "DIRECTION"
            self.log(f"[nav] located at ({x:.0f},{y:.0f}) "
                     f"{math.degrees(th):.0f} deg, q={sc:.2f}")

    def _resolve_direction(self, d_mm):
        ccw = Path(False, half_w=self.half_w)
        i, _ = ccw.project(self.pose.x, self.pose.y)
        if self._dir_last_i is not None:
            d = (i - self._dir_last_i) % ccw.n
            if d > ccw.n // 2:
                d -= ccw.n
            if abs(d) >= 1:
                self._dir_votes += 1 if d > 0 else -1
        self._dir_last_i = i
        self._dir_dist += abs(d_mm)
        if self._dir_dist > 220.0 and abs(self._dir_votes) >= 3:
            self.clockwise = self._dir_votes < 0
            self.path = Path(self.clockwise, half_w=self.half_w)
            self._last_i, _ = self.path.project(self.pose.x, self.pose.y)
            self.lat = np.zeros(self.path.n)
            self.pts, self.kappa, self.line_th = planned_line(self.path, self.lat)
            self.state = "RUN"
            self.log(f"[nav] direction {'CW' if self.clockwise else 'CCW'} "
                     f"(from the pose, not the colour sensor)")

    def _maybe_replan(self, t):
        """Re-solve only when the sign map has actually changed.

        Periodic replanning re-solved an identical problem nearly every call
        and cost ~200 ms a time. Event-driven, with a slow heartbeat so the
        anchor cannot drift far from the line being followed.
        """
        if self.path is None:
            return
        changed = self.map.version != self._map_version
        stale = (t - self._last_plan_t) > 3.0
        if not (changed or stale):
            return
        self._map_version = self.map.version
        self._last_plan_t = t

        with self._lock:
            pil = self.map.confirmed()
            anchor = self.path.project(self.pose.x, self.pose.y)
        t0 = time.monotonic()
        lat, info = solve(self.path, pil, half_w=self.half_w, anchor=anchor)
        pts, kap, lth = planned_line(self.path, lat)
        self.last_solve_ms = 1000.0 * (time.monotonic() - t0)
        with self._lock:
            self.lat, self.info = lat, info
            self.pts, self.kappa, self.line_th = pts, kap, lth
            self.n_replans += 1

    def stop(self):
        self._stop.set()
