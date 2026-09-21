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
from dataclasses import dataclass, field as dc_field

from perception.state import WorldBelief
from perception import fit as fitmod
from nav import arena

LOCK_TOTAL = 0.55        # minimum fit total to accept a lock
LOST_SCORE = 0.45        # tracking score below this is a bad frame
LOST_FRAMES = 5          # consecutive bad frames before declaring LOST
MAX_STEP_MM = 300.0      # further than the car can move between scans


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
    def __init__(self, car_len_mm=175.0, sensor_ahead=0.0,
                 sides=("N", "E", "S", "W"), cw=None):
        self.wb = WorldBelief(sensor_ahead=sensor_ahead)
        self.car_len_mm = car_len_mm
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

    # ---------------------------------------------------------------- step
    def step(self, ranges, cam_dets=None):
        if self.state in ("INIT", "LOST"):
            self._try_fix(ranges)
        else:
            self._track(ranges)

        if self.state == "LOCKED":
            self.wb.update(ranges, cam_dets=cam_dets)
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
                             "laps": self.laps}})
        return snap

    def state_tuple(self):
        return NavState(state=self.state, pose=self.wb.pose,
                        score=self.wb.score, direction=self.direction,
                        section=self.section, corners=self.corners,
                        laps=self.laps, healthy=self.state == "LOCKED",
                        fit=self.fit)
