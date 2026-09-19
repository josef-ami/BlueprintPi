"""
supervisor.py — the Pi-side avoidance lifecycle.

Kept out of obstacle_lap.py so it imports with no camera, no lidar and no
serial port: the state machine below is the part most worth testing off-robot,
and tests/test_supervisor.py does exactly that.

It owns exactly one decision — which manoeuvre, if any, the STM32 should be
running right now — and nothing else. All the geometry is in solver.py; all the
driving is in ObstacleLap.cpp.
"""

import math

from .percept_link import ST_BOOT, ST_TURN90, ST_RECOVER, ST_FINISH
from .solver import AvoidCfg, solve, AVOID_NONE, AVOID_TRACK, AVOID_COMMIT, wrap180

LOST_GRACE_TICKS = 3            # missed detections tolerated before committing
COMMIT_TIMEOUT_S = 5.0          # a frozen leg that never ends is a bug, not a plan


class AvoidSupervisor:
    """
    Pi-side avoidance lifecycle. Owns exactly one thing: which manoeuvre, if any,
    the STM32 should be running right now.

        IDLE ──pillar confirmed──> TRACK ──close / lost──> COMMIT ──leg done──> IDLE
                                     │                                   (refractory)
                                     └──solver says already clear──> IDLE

    TRACK re-solves every tick and the STM32 re-bases its odometry on each
    refresh, so the turn-in lag corrects itself. COMMIT freezes the answer and
    the leg runs out on odometry alone — which is what lets the car finish the
    manoeuvre after the pillar has left the camera's field of view.
    """

    def __init__(self, cfg: AvoidCfg, confirm_ticks=3, refractory_mm=150.0):
        self.cfg = cfg
        self.confirm_ticks = confirm_ticks
        self.refractory_mm = refractory_mm
        self._refractory_until = -1e12
        self.reset()

    def reset(self, refractory_from_odo=None):
        self.action = AVOID_NONE
        self.color = ""
        self.heading_abs = 0.0     # absolute heading the manoeuvre holds
        self.leg_mm = 0.0
        self.commit_odo = 0.0
        self.commit_t = 0.0
        self._confirm = 0
        self._lost = 0
        if refractory_from_odo is not None:
            self._refractory_until = refractory_from_odo + self.refractory_mm

    # ---- the one call per tick ----

    def tick(self, obstacles, telem, now, left_mm, right_mm):
        """Returns (action, color, target_heading_deg, leg_remaining_mm, note)."""
        # The STM32 is doing something we must not interrupt.
        if telem.state in (ST_BOOT, ST_TURN90, ST_RECOVER, ST_FINISH):
            if self.action != AVOID_NONE:
                self.reset(refractory_from_odo=telem.odo_mm)
            return AVOID_NONE, "", 0.0, 0.0, "stm32 busy"

        if self.action == AVOID_COMMIT:
            return self._run_commit(telem, now)

        return self._seek(obstacles, telem, now, left_mm, right_mm)

    # ---- frozen leg: pure odometry, no perception involved ----

    def _run_commit(self, telem, now):
        remaining = self.leg_mm - (telem.odo_mm - self.commit_odo)
        if remaining <= 0.0:
            self.reset(refractory_from_odo=telem.odo_mm)
            return AVOID_NONE, "", 0.0, 0.0, "leg complete"
        if now - self.commit_t > COMMIT_TIMEOUT_S:
            self.reset(refractory_from_odo=telem.odo_mm)
            return AVOID_NONE, "", 0.0, 0.0, "leg timed out"
        return (AVOID_COMMIT, self.color, self.heading_abs, remaining,
                f"commit {remaining:.0f}mm left")

    # ---- looking for, or tracking, a pillar ----

    def _seek(self, obstacles, telem, now, left_mm, right_mm):
        # Just-passed pillars sit behind us but a stray blob or a second
        # detection of the same one must not re-arm immediately.
        if self.action == AVOID_NONE and telem.odo_mm < self._refractory_until:
            return AVOID_NONE, "", 0.0, 0.0, "refractory"

        pillar = self.pick(obstacles)

        if pillar is None:
            self._lost += 1
            if self.action == AVOID_TRACK and self._lost >= LOST_GRACE_TICKS:
                # It left the frame mid-approach. Freeze what we last knew and
                # drive it out blind — this is the normal way a pass ends.
                return self._freeze(telem, now, "pillar out of view")
            if self.action != AVOID_TRACK:
                self._confirm = 0
                return AVOID_NONE, "", 0.0, 0.0, "no pillar"
            return (AVOID_TRACK, self.color, self.heading_abs, self.leg_mm,
                    "holding last solution")

        self._lost = 0
        side_free = right_mm if pillar.color == "RED" else left_mm
        sol = solve(pillar.color, pillar.bearing_deg, pillar.distance_mm,
                    self.cfg, side_free_mm=side_free)

        if sol.action == AVOID_NONE:
            if self.action == AVOID_TRACK:
                self.reset(refractory_from_odo=telem.odo_mm)
            self._confirm = 0
            return AVOID_NONE, "", 0.0, 0.0, sol.reason

        # One frame of a mis-classified colour must not start a manoeuvre.
        if self.action == AVOID_NONE:
            self._confirm += 1
            if self._confirm < self.confirm_ticks:
                return AVOID_NONE, "", 0.0, 0.0, f"confirming {self._confirm}"

        self.color = pillar.color
        self.heading_abs = wrap180(telem.heading_deg + sol.theta_deg)
        self.leg_mm = sol.leg_mm

        if sol.action == AVOID_COMMIT:
            return self._freeze(telem, now, "close range")

        self.action = AVOID_TRACK
        return AVOID_TRACK, self.color, self.heading_abs, self.leg_mm, sol.reason

    def _freeze(self, telem, now, why):
        self.action = AVOID_COMMIT
        self.commit_odo = telem.odo_mm
        self.commit_t = now
        return (AVOID_COMMIT, self.color, self.heading_abs, self.leg_mm,
                f"freeze ({why}) {self.leg_mm:.0f}mm")

    def pick(self, obstacles):
        """Nearest RED/GREEN with a real range, inside the working cone."""
        best = None
        for o in obstacles:
            if o.color not in ("RED", "GREEN"):
                continue
            if not math.isfinite(o.distance_mm):
                continue
            if o.distance_mm > self.cfg.engage_mm:
                continue
            if abs(o.bearing_deg) > self.cfg.max_bearing_deg:
                continue
            if best is None or o.distance_mm < best.distance_mm:
                best = o
        return best
