"""
perception/nav.py - the moving navigator.

Start with the systematic fit (walls + seats + parking, zero operator input),
then run LiDAR-only tracking frame by frame, and keep the semantic map growing
as the car drives.

STATE MACHINE
    INIT   -> no pose yet. Run the full fit until it clears LOCK_TOTAL.
    LOCKED -> tracking. Cheap tight scan-match every frame (~1.4 ms), and the
              seat/parking belief is updated only while locked.
    LOST   -> the match score collapsed or the pose jumped further than the car
              could physically have moved. Stop updating the map and re-fit.

WHY RECOVERY IS CONSTRAINED
    A re-fit is free to land on any of the four symmetric corridors. If it
    picked a different one, every pillar already in the map would silently move
    to the wrong place. So recovery is restricted to the straight we were last
    in (and its neighbours) and to the direction already established - it
    re-finds the car inside the frame it has been building, instead of choosing
    a new frame.

WHY THE MAP IS FROZEN WHILE LOST
    A sign entered at a wrong global position persists and poisons the plan.
    pillarmap already refuses updates when the pose is untrusted; this makes
    that explicit at the navigator level.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field as dc_field

from perception.state import WorldBelief
from perception import fit as fitmod
from nav import arena

LOCK_TOTAL = 0.55        # minimum fit total to accept a lock
LOST_SCORE = 0.45        # tracking score below this is a bad frame
LOST_FRAMES = 5          # consecutive bad frames before declaring LOST
MAX_STEP_MM = 300.0      # further than the car could move between scans

# Vehicle geometry - these MUST match nav/service.py or the whole map shifts.
# The LiDAR sits ahead of the vehicle reference point, and that same point is
# the origin used to place signs: get it wrong and every pillar is entered that
# far from where it really is, which then snaps onto the wrong seat.
LIDAR_AHEAD_MM = 130.0
# The camera sits ON TOP of the LiDAR on this car, so they share an origin.
# That makes camera-bearing -> LiDAR-pillar association exact: no parallax
# term, the two sensors disagree only by noise. (nav/service.py assumes 150 mm
# for a different mounting - do not copy that number here.)
CAMERA_AHEAD_MM = LIDAR_AHEAD_MM
TICKS_PER_MM = 1.4853
ODO_FRESH_S = 0.5        # telemetry older than this = no odometry available


def section_of(x, y):
    """Which of the eight track sections a pose is in: a straight (N/E/S/W) or
    a corner (NE/SE/SW/NW)."""
    h = arena.INNER / 2.0
    if abs(x) <= h:
        return "N" if y > 0 else "S"
    if abs(y) <= h:
        return "E" if x > 0 else "W"
    return ("N" if y > 0 else "S") + ("E" if x > 0 else "W")


def straight_of(section):
    """The straight a section belongs to, or None for a corner."""
    return section if section in ("N", "E", "S", "W") else None


_NEIGHBOURS = {"N": ("N", "E", "W"), "S": ("S", "E", "W"),
               "E": ("E", "N", "S"), "W": ("W", "N", "S")}


@dataclass
class NavState:
    state: str
    pose: tuple
    score: float
    direction: str
    section: str
    corners: int
    laps: int
    healthy: bool
    fit: object = None
    notes: list = dc_field(default_factory=list)


class Navigator:
    def __init__(self, car_len_mm=175.0, sensor_ahead=LIDAR_AHEAD_MM,
                 sides=("N", "E", "S", "W"), cw=None,
                 ticks_per_mm=TICKS_PER_MM, camera_ahead=CAMERA_AHEAD_MM):
        self.wb = WorldBelief(sensor_ahead=sensor_ahead)
        self.car_len_mm = car_len_mm
        self.ticks_per_mm = ticks_per_mm
        self.camera_ahead = camera_ahead
        # odometry (optional): fed from the STM32's IMU heading + encoder
        self._last_ticks = None
        self._last_head = None
        self._odo_t = 0.0
        self._cam = []
        self.sides = sides
        self.cw_hint = cw
        self.state = "INIT"
        self.direction = "?"
        self.section = None
        self.straight = None
        self.corners = 0
        self.laps = 0
        self.fit = None
        self._bad = 0
        self._prev_pose = None
        self._seen_straights = []

    # ------------------------------------------------- inputs from the car
    def on_telemetry(self, heading_deg, enc_ticks, t=None):
        """IMU heading (deg) + encoder ticks from the STM32, at its loop rate.

        Dead-reckons between scans. Cheap (~tens of us) so it is safe in the
        50 Hz control loop. Once this is flowing, tracking switches from
        'ungated re-match every scan' to 'predict then gated correction',
        which is what stops the pose wandering when a scan is partial.
        """
        t = time.time() if t is None else t
        h = math.radians(heading_deg)
        if self._last_ticks is None:
            self._last_ticks, self._last_head, self._odo_t = enc_ticks, h, t
            return
        d_mm = (enc_ticks - self._last_ticks) / self.ticks_per_mm
        dth = (h - self._last_head + math.pi) % (2 * math.pi) - math.pi
        self._last_ticks, self._last_head, self._odo_t = enc_ticks, h, t
        if self.state == "LOCKED":
            self.wb.predict(d_mm, dth)

    def on_camera(self, detections):
        """detections: [(colour, bearing_deg)] with 1=red, 0=green, + = left."""
        if detections:
            self._cam = list(detections)

    @property
    def has_odometry(self):
        return (time.time() - self._odo_t) < ODO_FRESH_S

    # ---------------------------------------------------------------- step
    def step(self, ranges, cam_dets=None):
        if cam_dets:
            self.on_camera(cam_dets)
        if self.state in ("INIT", "LOST"):
            self._try_fix(ranges)
        else:
            self._track(ranges)

        if self.state == "LOCKED":
            cam, self._cam = self._cam, []
            self.wb.update(ranges, cam_dets=cam or None)
            self._advance_sections()
        return self.snapshot()

    # ------------------------------------------------------------ internals
    def _try_fix(self, ranges):
        if self.state == "INIT":
            sides, cw = self.sides, self.cw_hint
        else:
            # recover INSIDE the frame we have been building
            sides = _NEIGHBOURS.get(self.straight, self.sides)
            cw = (self.direction == "CW") if self.direction in ("CW", "CCW") \
                else self.cw_hint
        res = self.wb.fit_start(ranges, cw=cw, car_len_mm=self.car_len_mm,
                                sides=sides)
        if res is None or res.total < LOCK_TOTAL:
            return
        self.fit = res
        if self.direction == "?":
            self.direction = res.direction
        self.state = "LOCKED"
        self._bad = 0
        self._prev_pose = self.wb.pose
        self._advance_sections()

    def _track(self, ranges):
        prev = self.wb.pose
        if self.has_odometry:
            # odometry already moved the pose in on_telemetry(); the scan is a
            # GATED correction on top, which rejects a bad match instead of
            # following it. This is the stable mode.
            self.wb.correct(ranges)
            score = self.wb.score
        else:
            _pose, score = self.wb.track(ranges)
        jumped = (self._prev_pose is not None and
                  math.hypot(self.wb.pose[0] - prev[0],
                             self.wb.pose[1] - prev[1]) > MAX_STEP_MM)
        if score < LOST_SCORE or jumped:
            self._bad += 1
        else:
            self._bad = 0
        if self._bad >= LOST_FRAMES:
            self.state = "LOST"
        self._prev_pose = self.wb.pose

    def _advance_sections(self):
        x, y, _th = self.wb.pose
        sec = section_of(x, y)
        if sec != self.section:
            self.section = sec
            st = straight_of(sec)
            if st is not None and st != self.straight:
                if self.straight is not None:
                    self.corners += 1
                self.straight = st
                self._seen_straights.append(st)
                if len(self._seen_straights) > 4:
                    self._seen_straights.pop(0)
                if self.corners and self.corners % 4 == 0 and \
                        len(set(self._seen_straights)) == 4:
                    self.laps = self.corners // 4

    # -------------------------------------------------------------- output
    def snapshot(self, ranges=None):
        snap = self.wb.snapshot(ranges=ranges)
        snap.update({"nav": {"state": self.state, "direction": self.direction,
                             "section": self.section, "corners": self.corners,
                             "laps": self.laps,
                             "odometry": self.has_odometry}})
        return snap

    def state_tuple(self):
        return NavState(state=self.state, pose=self.wb.pose,
                        score=self.wb.score, direction=self.direction,
                        section=self.section, corners=self.corners,
                        laps=self.laps, healthy=self.state == "LOCKED",
                        fit=self.fit)
