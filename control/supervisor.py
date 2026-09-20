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

from .percept_link import ST_BOOT, ST_TURN90, ST_RECOVER, ST_FINISH, ST_STOPPED
from .solver import AvoidCfg, solve, AVOID_NONE, AVOID_TRACK, AVOID_COMMIT, wrap180

LOST_GRACE_TICKS = 3            # missed detections tolerated before releasing

# Sent as leg_mm on every TRACK frame. The firmware only uses it as a backstop
# cap while it is NOT re-basing (link stale, or TRACK stops arriving) — while
# TRACK keeps arriving it re-bases every tick and the cap never engages. It is
# never a target to count down to; see the AVOID state comment in
# ObstacleLap.cpp. Sized to match the firmware's own AVOID_MAX_LEG_CM so the
# two backstops agree if this one is ever the one that's live.
BACKSTOP_LEG_MM = 1500.0


class AvoidSupervisor:
    """
    Pi-side avoidance lifecycle. Owns exactly one thing: which manoeuvre, if
    any, the STM32 should be running right now — and there are only two:

        NONE    nothing to do; hold the lane heading.
        TRACK   hold heading_abs. Re-solved every tick while the pillar is far
                enough for the geometry to be trustworthy; held UNCHANGED once
                it isn't (see below), until released.

    Release back to NONE happens the instant any of these is true — never on
    a distance or a timer:
      - solve() says the pillar is already clear (bearing past the keep-out
        cone on the correct side) — checked every tick, near or far;
      - the pillar has been out of view for LOST_GRACE_TICKS ticks;
      - solve() rejects the range outright (e.g. below min_range_mm — the
        pillar is now too close to size up, which only happens once it is
        beside or behind the car).

    Once close (solve() returns AVOID_COMMIT — not enough range left for the
    tangent solve to stay sane) TRACK is still what goes out on the wire, but
    heading_abs stops being updated: the last good heading is held as-is.
    solve() keeps running every tick regardless, purely to catch the
    already-clear and rejected-range releases above — that's what makes this
    self-terminating with no leg, no odometry countdown and no timer.
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
        self._confirm = 0
        self._lost = 0
        if refractory_from_odo is not None:
            self._refractory_until = refractory_from_odo + self.refractory_mm

    # ---- the one call per tick ----

    def tick(self, obstacles, telem, now, left_mm, right_mm):
        """Returns (action, color, target_heading_deg, leg_mm, note)."""
        # The STM32 is doing something we must not interrupt.
        if telem.state in (ST_BOOT, ST_TURN90, ST_RECOVER, ST_FINISH, ST_STOPPED):
            if self.action != AVOID_NONE:
                self.reset(refractory_from_odo=telem.odo_mm)
            return AVOID_NONE, "", 0.0, 0.0, "stm32 busy"

        return self._seek(obstacles, telem, now, left_mm, right_mm)

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
                # Out of view. Release now — the firmware's own heading PID
                # closes the gap back to the lane heading; there is nothing
                # left here to run out on a distance.
                self.reset(refractory_from_odo=telem.odo_mm)
                return AVOID_NONE, "", 0.0, 0.0, "pillar out of view - released"
            if self.action != AVOID_TRACK:
                self._confirm = 0
                return AVOID_NONE, "", 0.0, 0.0, "no pillar"
            return (AVOID_TRACK, self.color, self.heading_abs, BACKSTOP_LEG_MM,
                    "holding last solution (grace)")

        self._lost = 0
        side_free = right_mm if pillar.color == "RED" else left_mm
        sol = solve(pillar.color, pillar.bearing_deg, pillar.distance_mm,
                    self.cfg, side_free_mm=side_free)

        if sol.action == AVOID_NONE:
            # Already clear, or the range is unusable (e.g. too close to size
            # up any more — which only happens once the pillar is beside or
            # behind us). Either way: release.
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
        if sol.action == AVOID_TRACK:
            # Far enough that the tangent solve is still trustworthy: adopt it.
            self.heading_abs = wrap180(telem.heading_deg + sol.theta_deg)
        # else (AVOID_COMMIT from the solver: close range) — keep whatever
        # heading_abs already holds. Not re-solving here is what avoids the
        # near-degenerate large-angle answer the geometry gives up close;
        # the already-clear check above is what eventually releases it.

        self.action = AVOID_TRACK
        return AVOID_TRACK, self.color, self.heading_abs, BACKSTOP_LEG_MM, sol.reason

    def snapshot(self):
        """Read-only view of the lifecycle internals, for the dashboard."""
        return {
            "action": self.action,
            "color": self.color,
            "heading_abs": self.heading_abs,
            "confirm": self._confirm,
            "confirm_ticks": self.confirm_ticks,
            "lost": self._lost,
            "lost_grace": LOST_GRACE_TICKS,
            "refractory_until": self._refractory_until,
            "refractory_mm": self.refractory_mm,
        }

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
