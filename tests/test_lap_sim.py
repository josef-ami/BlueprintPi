"""
A whole run, end to end.

The unit tests check each rule; this checks that the rules compose - that the
car actually gets through twelve corners and stops, rather than deadlocking in
a state that never exits. It is a kinematic toy, not a physics model: the
point is the sequencing, not the trajectory.

The world is a rectangular lap. The car integrates a bicycle model at the
heading the FSM asks for, the walls are computed from where it is, and the
corner arrives when it has driven far enough down a straight.
"""

import math

import pytest

from control.fsm import (COLOUR_BLUE, COLOUR_NONE, COLOUR_ORANGE, Ctx, FSM,
                         State)
from control.intent import SteerMode
from control.link import S_ARC_DONE, Telemetry
from control.mapper import CommandMapper, MEASURED_RADIUS_MM
from worldstate import wrap180

TPM = 1.4853
STRAIGHT_MM = 2400.0          # how long each straight is
CORRIDOR = 1000.0

# The STM32's, not the Pi's: the arc's inner loop stayed on the firmware, so
# this is the firmware's default, mirrored here because this file is
# pretending to BE the firmware.
FW_TURN_STOP_DEG = 15.0


class World:
    """A lap made of straights. The car's position is tracked along and
    across the current straight; a corner resets both."""

    def __init__(self, p, straight_mm=STRAIGHT_MM, first_mm=None):
        self.p = p
        self.straight = straight_mm
        # On the real mat the car starts partway down the start section, so
        # the first segment A is shorter than a full straight L. That gap is
        # exactly what the final straight runs.
        self.first = first_mm if first_mm is not None else straight_mm * 0.55
        self.n_straights = 0
        self.along = 0.0          # down the current straight
        self.lat = 0.0            # + = left of centre
        self.heading = 0.0        # absolute
        self.lane = 0.0           # the current straight's direction
        self.odo = 0
        self.corners = 0

    def step(self, intent, dt, fsm):
        """Advance the car by one tick under `intent`."""
        speed_mm_s = 400.0 * (intent.speed_pwm / 60.0)
        if intent.mode is SteerMode.STOP or intent.speed_pwm == 0:
            return
        v = speed_mm_s * (-1.0 if intent.reverse else 1.0)

        if intent.mode is SteerMode.HEADING_HOLD:
            err = wrap180(intent.target_heading_deg - self.heading)
            # the STM32's P loop, rate-limited the way the servo is
            self.heading += max(-6.0, min(6.0, err * 0.35))
        elif intent.mode is SteerMode.ARC:
            err = wrap180(intent.target_heading_deg - self.heading)
            rate = math.degrees(abs(v) / MEASURED_RADIUS_MM) * dt
            self.heading += math.copysign(min(rate, abs(err)), err)
        elif intent.mode is SteerMode.DIRECT:
            rate = math.degrees(v / MEASURED_RADIUS_MM) * dt
            self.heading += rate * (intent.steer_deg / 35.0)

        d = v * dt
        yaw = math.radians(wrap180(self.heading - self.lane))
        self.along += d * math.cos(yaw)
        self.lat += d * math.sin(yaw)
        self.lat = max(-480.0, min(480.0, self.lat))
        self.odo += int(d * TPM)

    def corner_here(self):
        return self.along >= (self.first if self.n_straights == 0
                              else self.straight)

    def turn(self, fsm):
        """The world's side of a corner: a new straight, rotated."""
        self.corners += 1
        self.n_straights += 1
        self.lane = wrap180(self.lane - (90.0 if fsm.clockwise else -90.0))
        self.along = 0.0
        self.lat = 0.0

    # ---- what the sensors would report ----

    def cones(self):
        c = math.cos(math.radians(wrap180(self.heading - self.lane)))
        c = max(0.2, abs(c))
        return ((CORRIDOR / 2 - self.lat) / c, (CORRIDOR / 2 + self.lat) / c)

    def sides(self, fsm):
        """The single beams. The inner wall ends at the corner."""
        left, right = self.cones()
        if self.corner_here():
            if fsm.turn_is_clockwise():
                right = 3000.0
            else:
                left = 3000.0
        return left, right

    def front(self):
        end = self.first if self.n_straights == 0 else self.straight
        return max(60.0, end + 500.0 - self.along)


def run_lap(p, ticks=40000, straight_mm=STRAIGHT_MM, first_mm=None):
    fsm = FSM(p, emit=lambda s: None)
    mapper = CommandMapper()
    w = World(p, straight_mm, first_mm)
    t = 0.0
    dt = 1.0 / 50.0
    arc_done_latch = False
    log = []

    fsm.step(_ctx(p, w, fsm, t, 0))           # arm
    fsm.request_start()

    for i in range(ticks):
        t += dt
        status = 0
        # The STM32 sets ARC_DONE when the arc is within TURN_STOP_DEG.
        if fsm.state is State.TURNING:
            err = abs(wrap180(fsm.turn_target - w.heading))
            if err < FW_TURN_STOP_DEG:
                status |= S_ARC_DONE
                arc_done_latch = True
        floor = COLOUR_ORANGE if (w.corner_here() and fsm.corner_count == 0
                                  and not fsm.dc_colour_armed) else COLOUR_NONE

        before = fsm.corner_count
        intent = mapper.map(fsm.step(_ctx(p, w, fsm, t, status, floor)))
        if fsm.corner_count != before:
            log.append((fsm.corner_count, round(w.along), round(w.lat)))
        # the world's corner happens when the FSM leaves the arc
        if before != fsm.corner_count or (
                fsm.state is State.DRIVE_TO_CORNER and w.corner_here()
                and fsm.corner_count > w.corners):
            w.turn(fsm)

        w.step(intent, dt, fsm)
        if fsm.state is State.FINISHED and fsm.corner_count > 0:
            return fsm, w, i, log
    return fsm, w, ticks, log


def _ctx(p, w, fsm, t, status, floor=COLOUR_NONE):
    left, right = w.sides(fsm)
    cl, cr = w.cones()
    tel = Telemetry(stamp=t, status=status, heading_deg=w.heading,
                    odo_ticks=w.odo, floor=floor)
    return Ctx(p=p, now=t, telem=tel, telem_fresh=True, lidar_live=True,
               lidar_dead=False, new_frame=True, new_rev=True,
               front_mm=w.front(), left_mm=left, right_mm=right,
               cone_left=cl if cl < 1100 else None,
               cone_right=cr if cr < 1100 else None,
               wall_ang=wrap180(w.heading - w.lane),
               ticks_per_mm=TPM)


# -------------------------------------------------------------- the runs

def test_a_full_run_finishes(p):
    fsm, w, ticks, log = run_lap(p)
    assert fsm.state is State.FINISHED, \
        f"stuck in {fsm.state.name} after {ticks} ticks"
    assert fsm.corner_count == p["TARGET_CORNERS"]
    assert len(log) == p["TARGET_CORNERS"]


def test_the_direction_locks_on_the_first_corner(p):
    fsm, w, ticks, log = run_lap(p)
    assert fsm.locked_colour == COLOUR_ORANGE
    assert fsm.clockwise


def test_the_lane_heading_tracks_the_lap(p):
    """Twelve corners of 90 deg clockwise is three full turns; the lane
    heading must come back to where it started. The levelling is what keeps
    it there - it pulls the IMU's idea of the lane onto the fitted walls once
    a revolution, so three laps of drift cannot accumulate."""
    fsm, w, ticks, log = run_lap(p)
    assert wrap180(fsm.lane_heading) == pytest.approx(0.0, abs=2.0)
    assert wrap180(fsm.lane_heading - w.lane) == pytest.approx(0.0, abs=2.0)


def test_the_car_stays_inside_the_lane(p):
    """A run that ends with the car against a wall the whole way would still
    'finish'. It should not."""
    fsm, w, ticks, log = run_lap(p)
    offsets = [abs(lat) for _, _, lat in log]
    limit = p.derived["LANE_LIMIT_MM"]
    assert max(offsets) < limit, \
        f"the car rode the wall into a corner: {offsets}"


def test_centring_holds_with_the_pre_corner_swing_off(p):
    """With the swing off and no pillars, the target is the lane centre the
    whole way round, so the offset at each corner should be small - not
    merely legal. This is the check that the centring law itself works; the
    test above only proves the car did not hit anything."""
    p.set("PRE_CORNER_SWING_MM", 0.0)
    fsm, w, ticks, log = run_lap(p)
    offsets = [abs(lat) for _, _, lat in log]
    assert sum(offsets) / len(offsets) < 60, \
        f"centring is not holding: {offsets}"


def test_the_pre_corner_swing_enters_on_the_side_opposite_the_exit(p):
    """Enter wide, leave tight. A 90 deg arc throws the car away from the
    side it started on, so the entry side is the opposite of the planned
    exit side - and with no pillar seen the planned exit is the OUTER side,
    which on a clockwise lap is positive. So the car should arrive at every
    corner held on the INNER (negative) side by about the swing distance.

    Getting this sign backwards is silent: the car still finishes, it just
    exits every corner pinned against the wrong wall, which is what the
    give-up logic then reads as 'the correct side is unreachable'.
    """
    fsm, w, ticks, log = run_lap(p)
    assert fsm.clockwise                      # orange lap: + offset = outer
    lats = [lat for _, _, lat in log]
    assert all(v < 0 for v in lats), \
        f"the swing went to the outer side, not the entry side: {lats}"

    swing = p["PRE_CORNER_SWING_MM"]
    assert all(0.2 * swing < abs(v) < 1.2 * swing for v in lats), \
        f"the swing is not being held: {lats} against {swing}"

    # ... and flipping the sign of the swing must flip the side.
    p.set("PRE_CORNER_SWING_MM", -swing)
    fsm2, w2, t2, log2 = run_lap(p)
    assert all(lat > 0 for _, _, lat in log2), \
        f"a negative swing did not move to the other side: {log2}"


def test_the_final_straight_is_shorter_than_a_full_one(p):
    """It runs L - A: the full straight minus the first segment."""
    fsm, w, ticks, log = run_lap(p)
    assert fsm.have_full_straight
    assert 0 < fsm.final_distance_cm < fsm.full_start_straight_cm


def test_a_stop_mid_run_halts_it(p):
    fsm = FSM(p, emit=lambda s: None)
    mapper = CommandMapper()
    w = World(p)
    t = 0.0
    fsm.step(_ctx(p, w, fsm, t, 0))
    fsm.request_start()
    for i in range(400):
        t += 0.02
        intent = mapper.map(fsm.step(_ctx(p, w, fsm, t, 0)))
        w.step(intent, 0.02, fsm)
        if i == 200:
            fsm.request_stop()
    assert fsm.state is State.FINISHED
    assert fsm.corner_count == 0


def test_a_shorter_lap_still_finishes(p):
    """The corner geometry should not depend on the straight being long."""
    fsm, w, ticks, log = run_lap(p, straight_mm=1600.0)
    assert fsm.state is State.FINISHED
    assert fsm.corner_count == p["TARGET_CORNERS"]


def run_lap_ccw(p, ticks=40000):
    """The same run, blue line, anticlockwise. The corner-exit sign argument
    depends on the direction, so both ways round have to work."""
    fsm = FSM(p, emit=lambda s: None)
    mapper = CommandMapper()
    w = World(p)
    t, dt = 0.0, 1.0 / 50.0
    fsm.step(_ctx(p, w, fsm, t, 0))
    fsm.request_start()
    for i in range(ticks):
        t += dt
        status = 0
        if fsm.state is State.TURNING and \
                abs(wrap180(fsm.turn_target - w.heading)) < FW_TURN_STOP_DEG:
            status |= S_ARC_DONE
        floor = (COLOUR_BLUE if (w.corner_here() and fsm.corner_count == 0
                                 and not fsm.dc_colour_armed) else COLOUR_NONE)
        before = fsm.corner_count
        intent = mapper.map(fsm.step(_ctx(p, w, fsm, t, status, floor)))
        if before != fsm.corner_count or (
                fsm.state is State.DRIVE_TO_CORNER and w.corner_here()
                and fsm.corner_count > w.corners):
            w.turn(fsm)
        w.step(intent, dt, fsm)
        if fsm.state is State.FINISHED and fsm.corner_count > 0:
            return fsm, w, i
    return fsm, w, ticks


def test_anticlockwise_also_finishes(p):
    fsm, w, ticks = run_lap_ccw(p)
    assert fsm.state is State.FINISHED, f"stuck in {fsm.state.name}"
    assert not fsm.clockwise, "blue must lock anticlockwise"
    assert fsm.corner_count == p["TARGET_CORNERS"]
    assert wrap180(fsm.lane_heading) == pytest.approx(0.0, abs=2.0)
