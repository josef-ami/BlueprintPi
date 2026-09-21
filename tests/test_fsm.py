"""
The state machine, against the behaviour ObstacleRound.cpp's states had.

A fake STM32 stands in for the telemetry: the FSM only ever learns the
heading, the odometry and the floor colour through TELEM, so a dataclass with
those three fields is a complete substitute and every test can put the car
exactly where it wants it.
"""

import math

import pytest

from control.fsm import (COLOUR_BLUE, COLOUR_NONE, COLOUR_ORANGE, Ctx, FSM,
                         MnvPhase, State)
from control.intent import SteerMode
from control.link import (S_ARC_DONE, S_RECOVERING, S_RECOVER_CAPPED,
                          Telemetry)
from worldstate import wrap180

TPM = 1.4853          # ticks per mm, the firmware's TICKS_PER_CM / 10


class Car:
    """A fake STM32 and a fake world, driven by the test."""

    def __init__(self, p):
        self.p = p
        self.fsm = FSM(p, emit=lambda s: None)
        self.t = 0.0
        self.odo = 0
        self.heading = 0.0
        self.floor = COLOUR_NONE
        self.status = 0
        self.front = 5000.0
        self.left = 500.0
        self.right = 500.0
        self.cone_left = 500.0
        self.cone_right = 500.0
        self.wall_ang = 0.0
        self.pillar_colour = None
        self.pillar_xy = None
        self.sec_colour = None
        self.sec_xy = None
        self.unknown_xy = None
        self.last = None

    def ctx(self, **kw):
        tel = Telemetry(stamp=self.t, status=self.status,
                        heading_deg=self.heading, odo_ticks=self.odo,
                        floor=self.floor)
        d = dict(p=self.p, now=self.t, telem=tel, telem_fresh=True,
                 lidar_live=True, lidar_dead=False, new_frame=True,
                 new_rev=True, front_mm=self.front, left_mm=self.left,
                 right_mm=self.right, cone_left=self.cone_left,
                 cone_right=self.cone_right, wall_ang=self.wall_ang,
                 pillar_xy=self.pillar_xy, pillar_colour=self.pillar_colour,
                 sec_xy=self.sec_xy, sec_colour=self.sec_colour,
                 unknown_xy=self.unknown_xy, ticks_per_mm=TPM)
        d.update(kw)
        return Ctx(**d)

    def step(self, n=1, advance_mm=0.0, **kw):
        for _ in range(n):
            self.t += 0.02
            self.odo += int(advance_mm * TPM)
            self.last = self.fsm.step(self.ctx(**kw))
        return self.last

    def go(self):
        # Arm first. WAIT_START discards a Start that arrives before it is
        # entered - the firmware does the same, so that a Start sent while the
        # car was still booting cannot launch it the moment it arms.
        self.step()
        self.fsm.request_start()
        return self.step()


# ------------------------------------------------------------ WAIT_START

def test_starts_armed_and_stopped(p):
    c = Car(p)
    out = c.step()
    assert c.fsm.state is State.WAIT_START
    assert out.mode is SteerMode.STOP


def test_start_refused_while_the_lidar_is_stale(p):
    c = Car(p)
    c.fsm.request_start()
    out = c.step(lidar_live=False)
    assert c.fsm.state is State.WAIT_START
    assert out.mode is SteerMode.STOP


def test_start_refused_without_telemetry(p):
    c = Car(p)
    c.fsm.request_start()
    c.step(telem_fresh=False)
    assert c.fsm.state is State.WAIT_START


def test_go_sets_the_lane_heading_to_where_the_car_points(p):
    c = Car(p)
    c.heading = 37.5
    c.go()
    assert c.fsm.state is State.DRIVE_TO_CORNER
    assert c.fsm.lane_heading == pytest.approx(37.5)


def test_go_zeroes_the_distance_baselines(p):
    c = Car(p)
    c.odo = 123456
    c.go()
    assert c.fsm._corner_odo == 123456
    assert c.fsm._since_corner_mm(c.ctx()) == pytest.approx(0.0)


# ----------------------------------------------------------- blind = stop

def test_no_telemetry_commands_stop(p):
    """Without TELEM there is no heading and no odometry. There is nothing
    useful a state machine can do blind."""
    c = Car(p)
    c.go()
    out = c.step(telem_fresh=False)
    assert out.mode is SteerMode.STOP
    assert "TELEM" in out.reason


# --------------------------------------------------------------- driving

def test_drive_holds_the_lane_heading_plus_the_planner(p):
    c = Car(p)
    c.go()
    out = c.step()
    assert out.mode is SteerMode.HEADING_HOLD
    assert out.speed_pwm == p["DRIVE_PWM"]
    assert out.target_heading_deg == pytest.approx(
        wrap180(c.fsm.target_heading + c.fsm.planner.lat_yaw_cmd))


def test_planner_is_live_from_the_first_tick_after_a_corner(p):
    """A pillar right after the corner has to be handled immediately, so the
    planner runs through the post-corner lockout - only the LEVELLING waits."""
    c = Car(p)
    c.go()
    c.step()
    assert c.fsm.planner_enabled
    assert not c.fsm.level_enabled, "levelling waits out the lockout"


def test_levelling_turns_on_past_the_lockout(p):
    c = Car(p)
    c.go()
    c.fsm.corner_count = 1
    c.step(n=2, advance_mm=p["POST_CORNER_LOCKOUT_CM"] * 10.0)
    assert c.fsm.level_enabled


def test_safety_distance_restarts_the_straight(p):
    """No corner within SEARCH_SAFETY_CM means the trigger is broken. The
    firmware clears `entered` and lets the handler start the straight over on
    the next tick, which is visible as the baseline moving."""
    c = Car(p)
    c.go()
    before = c.fsm._dc_base
    c.step(n=2, advance_mm=p["SEARCH_SAFETY_CM"] * 10.0)
    assert c.fsm.state is State.DRIVE_TO_CORNER
    assert c.fsm._dc_base != before, "the straight restarted"
    assert c.fsm._since_corner_mm(c.ctx()) > 0, "but the corner baseline did not"


# -------------------------------------------------------- corner trigger

def drive_until_locked(c, p):
    """Cross the orange line so the direction locks, the way lap 1 does."""
    c.go()
    c.step(n=2, advance_mm=100.0)
    c.floor = COLOUR_ORANGE
    c.step(n=3, advance_mm=10.0)          # debounce
    return c


def test_colour_gate_arms_and_sets_the_direction(p):
    c = Car(p)
    drive_until_locked(c, p)
    assert c.fsm.dc_colour_armed
    assert c.fsm.last_first_colour == COLOUR_ORANGE


def test_blue_locks_anticlockwise(p):
    c = Car(p)
    c.go()
    c.step(n=2, advance_mm=100.0)
    c.floor = COLOUR_BLUE
    c.step(n=3, advance_mm=10.0)
    c.floor = COLOUR_NONE
    c.right = c.left = 300.0
    c.step(n=p["SIDE_WALL_FRAMES"] + 2, advance_mm=10.0)
    c.left = p["SIDE_OPEN_MM"] + 500      # the LEFT side opens
    c.step(n=p["SIDE_OPEN_FRAMES"] + 1, advance_mm=10.0)
    assert c.fsm.dc_turn_armed
    assert not c.fsm.clockwise


def test_turn_needs_the_wall_to_have_been_seen_first(p):
    """A straight that starts inside a corner square sees 'open' before the
    inner wall even begins. Without the wall-seen proof that faked a second
    corner, so the side must read SHORT at least SIDE_WALL_FRAMES times
    before an opening is allowed to count."""
    c = Car(p)
    c.right = c.left = p["SIDE_OPEN_MM"] + 500   # open from the very first tick
    c.go()
    c.floor = COLOUR_ORANGE
    c.step(n=3, advance_mm=10.0)                 # arm the gate
    c.floor = COLOUR_NONE
    c.step(n=p["SIDE_OPEN_FRAMES"] + 3, advance_mm=10.0)
    assert c.fsm.side_wall_count == 0, "the inner wall was never seen"
    assert c.fsm.side_open_count == 0, "so an opening must not count"
    assert not c.fsm.dc_turn_armed


def test_turn_arms_after_the_wall_then_the_opening(p):
    c = Car(p)
    drive_until_locked(c, p)
    c.floor = COLOUR_NONE
    c.right = 300.0                       # the inner wall, clearly seen
    c.step(n=p["SIDE_WALL_FRAMES"] + 1, advance_mm=10.0)
    assert c.fsm.side_wall_count >= p["SIDE_WALL_FRAMES"]
    c.right = p["SIDE_OPEN_MM"] + 500     # and now it ends
    c.step(n=p["SIDE_OPEN_FRAMES"], advance_mm=10.0)
    assert c.fsm.dc_turn_armed
    assert c.fsm.clockwise


def test_a_yawed_car_cannot_trigger_a_corner(p):
    """Mid-swerve the side beam is not sideways and fakes an open corner."""
    c = Car(p)
    drive_until_locked(c, p)
    c.floor = COLOUR_NONE
    c.right = 300.0
    c.step(n=p["SIDE_WALL_FRAMES"] + 1, advance_mm=10.0)
    c.heading = p["TURN_TRIGGER_MAX_YAW"] + 10     # swerving hard
    c.right = p["SIDE_OPEN_MM"] + 500
    c.step(n=p["SIDE_OPEN_FRAMES"] + 2, advance_mm=10.0)
    assert c.fsm.side_open_count == 0
    assert not c.fsm.dc_turn_armed


def test_stale_lidar_resets_the_open_count(p):
    c = Car(p)
    drive_until_locked(c, p)
    c.floor = COLOUR_NONE
    c.right = 300.0
    c.step(n=p["SIDE_WALL_FRAMES"] + 1, advance_mm=10.0)
    c.right = p["SIDE_OPEN_MM"] + 500
    c.step(n=1, advance_mm=10.0)
    assert c.fsm.side_open_count == 1
    c.step(n=1, advance_mm=10.0, lidar_live=False)
    assert c.fsm.side_open_count == 0


def arm_the_corner(c, p):
    drive_until_locked(c, p)
    c.floor = COLOUR_NONE
    c.right = 300.0
    c.step(n=p["SIDE_WALL_FRAMES"] + 1, advance_mm=10.0)
    c.right = p["SIDE_OPEN_MM"] + 500
    c.step(n=p["SIDE_OPEN_FRAMES"], advance_mm=10.0)
    assert c.fsm.dc_turn_armed
    return c


def test_the_arc_waits_for_the_wall_ahead(p):
    """The side opening says the corner is HERE; it does not say the car is
    deep enough into the square to arc."""
    c = Car(p)
    arm_the_corner(c, p)
    c.front = p["TURN_FRONT_MM"] + 400        # wall ahead still far
    c.step(n=3, advance_mm=20.0)
    assert c.fsm.state is State.DRIVE_TO_CORNER
    c.front = p["TURN_FRONT_MM"] - 50         # now it is close
    c.step()
    assert c.fsm.state is State.TURNING


def test_the_arc_fires_anyway_at_the_backstop(p):
    """A stale or missing front reading must not let the car coast past the
    corner forever."""
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 5000.0                          # never gets close
    c.step(n=3, advance_mm=p["TURN_ARM_MAX_MM"])
    assert c.fsm.state is State.TURNING


def test_turn_delay_holds_the_arc_off(p):
    p.set("TURN_DELAY_MM", 400)
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 100.0                           # the front cue is satisfied
    c.step(n=2, advance_mm=50.0)
    assert c.fsm.state is State.DRIVE_TO_CORNER, "the delay has not run out"
    c.step(n=2, advance_mm=300.0)
    assert c.fsm.state is State.TURNING


# ------------------------------------------------------------- the arc

def test_turning_commands_an_arc_to_the_new_heading(p):
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 100.0
    out = c.step()
    assert c.fsm.state is State.TURNING
    out = c.step()
    assert out.mode is SteerMode.ARC
    # clockwise: lane heading goes from 0 to -90
    assert out.target_heading_deg == pytest.approx(-90.0)
    assert out.arc_lock == pytest.approx(c.fsm.turn_lock_cmd)


def test_arc_done_advances_and_rebases_the_lane(p):
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 100.0
    c.step(n=2)
    assert c.fsm.state is State.TURNING
    before = c.fsm.corner_count
    c.heading = -90.0
    c.status = S_ARC_DONE
    c.step(advance_mm=500.0)
    assert c.fsm.corner_count == before
    assert c.fsm.lane_heading == pytest.approx(-90.0)
    assert c.fsm.state is State.DRIVE_TO_CORNER
    assert c.fsm.planner.lane_along == pytest.approx(0.0)


def test_arc_has_an_odometry_backstop(p):
    """TURN_STOP_DEG is the STM32's to judge, but a car that never reaches it
    must still leave the arc."""
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 100.0
    c.step(n=2)
    assert c.fsm.state is State.TURNING
    c.step(n=2, advance_mm=p["TURN_CAP_CM"] * 10.0)
    assert c.fsm.state is State.DRIVE_TO_CORNER


def test_corner_count_increments_once_per_corner(p):
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 100.0
    c.step(n=3)
    assert c.fsm.corner_count == 1


def test_tracks_are_cleared_across_a_corner(p):
    """Anything seen mid-turn was in the old lane frame."""
    from control.planner import Track
    c = Car(p)
    arm_the_corner(c, p)
    c.fsm.planner.tracks.append(Track(colour="RED", lat=0.0, along=100.0,
                                      hits=9))
    c.front = 100.0
    c.step(n=2)
    assert c.fsm.planner.tracks == []


# ------------------------------------------------------- corner exit plan

def test_corner_exit_defaults_to_the_outer_side(p):
    p.set("CORNER_EXIT_MM", 150)
    c = Car(p)
    arm_the_corner(c, p)
    # clockwise -> outer is +1
    assert c.fsm.planner.corner_exit_cmd == pytest.approx(150.0)


def test_red_after_the_corner_aims_for_the_lane_right(p):
    """The passing rule is about the LANE, not the lap: red is passed on its
    right, which is the negative side of the lane whichever way round the car
    goes. Tying red to 'inner' was right clockwise and backwards the other
    way."""
    c = Car(p)
    drive_until_locked(c, p)
    c.fsm.planner.next_straight_colour = "RED"
    c.fsm._plan_corner_exit(c.ctx())
    assert c.fsm.planner.corner_exit_cmd == pytest.approx(
        -p["CORNER_EXIT_BIAS_MM"])


def test_green_after_the_corner_aims_for_the_lane_left(p):
    c = Car(p)
    drive_until_locked(c, p)
    c.fsm.planner.next_straight_colour = "GREEN"
    c.fsm._plan_corner_exit(c.ctx())
    assert c.fsm.planner.corner_exit_cmd == pytest.approx(
        p["CORNER_EXIT_BIAS_MM"])


def test_exit_side_is_direction_dependent(p):
    """The same colour is the INNER side one way round and the OUTER the
    other, which is what decides whether the arc is tightened."""
    c = Car(p)
    drive_until_locked(c, p)                  # clockwise
    c.fsm.planner.next_straight_colour = "RED"
    c.fsm._plan_corner_exit(c.ctx())
    cw_inner = c.fsm.turn_front_cmd == p["TURN_FRONT_INNER_MM"]

    c2 = Car(p)
    c2.go()
    c2.fsm.clockwise = False
    c2.fsm.locked_colour = COLOUR_BLUE
    c2.fsm.planner.next_straight_colour = "RED"
    c2.fsm._plan_corner_exit(c2.ctx())
    ccw_inner = c2.fsm.turn_front_cmd == p["TURN_FRONT_INNER_MM"]
    assert cw_inner != ccw_inner


def test_arc_adapt_can_be_switched_off(p):
    p.set("CORNER_ARC_ADAPT", False)
    c = Car(p)
    drive_until_locked(c, p)
    c.fsm.planner.next_straight_colour = "RED"
    c.fsm._plan_corner_exit(c.ctx())
    assert c.fsm.turn_front_cmd == pytest.approx(p["TURN_FRONT_MM"])
    assert c.fsm.turn_lock_cmd == pytest.approx(p["TURN_LOCK_FRACTION"])


# ---------------------------------------------------- the final straight

def test_final_straight_after_the_last_corner(p):
    p.set("TARGET_CORNERS", 1)
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 100.0
    c.step(n=2)
    c.heading = -90.0
    c.status = S_ARC_DONE
    c.step()
    assert c.fsm.state is State.FINAL_STRAIGHT


def test_final_straight_stops_after_its_distance(p):
    p.set("TARGET_CORNERS", 1)
    c = Car(p)
    arm_the_corner(c, p)
    c.front = 100.0
    c.step(n=2)
    c.heading = -90.0
    c.status = S_ARC_DONE
    c.step()
    c.status = 0
    c.fsm.final_distance_cm = 50.0
    out = c.step(n=2, advance_mm=300.0)
    assert c.fsm.state is State.FINISHED
    assert out.mode is SteerMode.STOP


def test_first_segment_is_measured_at_the_arm_point(p):
    c = Car(p)
    drive_until_locked(c, p)
    c.floor = COLOUR_NONE
    c.right = 300.0
    c.step(n=p["SIDE_WALL_FRAMES"] + 1, advance_mm=100.0)
    travelled = c.fsm._since_corner_cm(c.ctx())
    c.right = p["SIDE_OPEN_MM"] + 500
    c.step(n=p["SIDE_OPEN_FRAMES"], advance_mm=0.0)
    assert c.fsm.first_segment_cm == pytest.approx(travelled, rel=0.02)


# ------------------------------------------------------------ stop/finish

def test_stop_works_from_any_state(p):
    c = Car(p)
    c.go()
    c.step(n=3, advance_mm=100.0)
    c.fsm.request_stop()
    out = c.step()
    assert c.fsm.state is State.FINISHED
    assert out.mode is SteerMode.STOP


def test_finished_stays_stopped(p):
    c = Car(p)
    c.go()
    c.fsm.request_stop()
    c.step(n=5)
    assert c.fsm.state is State.FINISHED
    assert c.last.mode is SteerMode.STOP


def test_start_from_finished_rearms_and_runs(p):
    c = Car(p)
    c.go()
    c.fsm.request_stop()
    c.step()
    assert c.fsm.state is State.FINISHED
    c.fsm.request_start()
    # The firmware arms on this tick and honours the press on the next, so a
    # press that arrives while FINISHED cannot skip the armed state.
    c.step()
    assert c.fsm.state is State.WAIT_START
    c.step()
    assert c.fsm.state is State.DRIVE_TO_CORNER
    assert c.fsm.corner_count == 0


# ------------------------------------------------------------- recovery

def test_recovery_is_observed_and_suspends_the_planner(p):
    c = Car(p)
    c.go()
    c.step(n=2, advance_mm=100.0)
    c.status = S_RECOVERING
    c.step()
    assert c.fsm.state is State.RECOVER
    assert not c.fsm.planner_enabled
    assert not c.fsm.level_enabled
    assert c.fsm.colour_muted


def test_recovery_returns_to_the_state_it_interrupted(p):
    c = Car(p)
    c.go()
    c.step(n=2, advance_mm=100.0)
    assert c.fsm.state is State.DRIVE_TO_CORNER
    c.status = S_RECOVERING
    c.step(n=2)
    c.status = 0
    c.step()
    assert c.fsm.state is State.DRIVE_TO_CORNER


def test_colour_stays_muted_until_the_car_drives_back_past(p):
    c = Car(p)
    c.go()
    c.step(n=2, advance_mm=200.0)
    c.status = S_RECOVERING
    c.step()
    mute_from = c.fsm._colour_mute_from
    c.odo -= int(200 * TPM)                # reversed
    c.status = 0
    c.step()
    assert c.fsm.colour_muted, "still behind where it started reversing"
    c.step(n=2, advance_mm=300.0)
    assert not c.fsm.colour_muted
    assert c.odo >= mute_from


def test_recovery_never_fires_from_wait_start(p):
    c = Car(p)
    c.status = S_RECOVERING
    c.step(n=2)
    assert c.fsm.state is State.WAIT_START


# ------------------------------------------------------- 3-point corner

def test_maneuver_is_not_taken_by_default(p):
    c = Car(p)
    arm_the_corner(c, p)
    assert not c.fsm.dc_turn_maneuver
    c.front = 100.0
    c.step(n=2)
    assert c.fsm.state is State.TURNING


def test_maneuver_is_taken_after_an_inner_pass(p):
    p.set("USE_CORNER_MANEUVER", True)
    c = Car(p)
    drive_until_locked(c, p)
    c.floor = COLOUR_NONE
    c.fsm.planner.have_inner_pass = True
    c.fsm.planner.last_inner_pass_at = c.fsm.planner.lane_along
    c.right = 300.0
    c.step(n=p["SIDE_WALL_FRAMES"] + 1, advance_mm=10.0)
    c.right = p["SIDE_OPEN_MM"] + 500
    c.step(n=p["SIDE_OPEN_FRAMES"], advance_mm=10.0)
    assert c.fsm.dc_turn_maneuver
    c.front = 100.0
    c.step(n=2)
    assert c.fsm.state is State.CORNER_MANEUVER


def test_maneuver_phases_advance(p):
    p.set("USE_CORNER_MANEUVER", True)
    c = Car(p)
    drive_until_locked(c, p)
    c.floor = COLOUR_NONE
    c.fsm.planner.have_inner_pass = True
    c.fsm.planner.last_inner_pass_at = c.fsm.planner.lane_along
    c.right = 300.0
    c.step(n=p["SIDE_WALL_FRAMES"] + 1, advance_mm=10.0)
    c.right = p["SIDE_OPEN_MM"] + 500
    c.step(n=p["SIDE_OPEN_FRAMES"], advance_mm=10.0)
    c.front = 100.0
    c.step()                                       # the arc cue fires
    assert c.fsm.state is State.CORNER_MANEUVER
    c.front = p["MNV_DEEP_FRONT_MM"] + 400         # not deep into the corner yet
    c.step()
    assert c.fsm.mnv_phase is MnvPhase.SWING
    c.front = p["MNV_DEEP_FRONT_MM"] - 50          # deep enough
    c.step()
    assert c.fsm.mnv_phase is MnvPhase.ARC_FWD
    out = c.step()
    assert out.mode is SteerMode.DIRECT
    c.heading = -p["MNV_ARC_FWD_DEG"] - 5          # rotated far enough
    c.step()
    assert c.fsm.mnv_phase is MnvPhase.STOP


def test_maneuver_leg_is_measured_from_where_it_started(p):
    """Reversing at the opposite lock keeps rotating the car the SAME way, so
    a leg threshold measured against the total turned since the old lane is
    already exceeded the moment the reverse ends - every later forward leg
    then finished on its first pass without moving."""
    p.set("USE_CORNER_MANEUVER", True)
    c = Car(p)
    c.go()
    c.fsm.clockwise = True
    c.fsm.lane_heading = 0.0
    c.fsm.state = State.CORNER_MANEUVER
    c.fsm.entered = True
    c.fsm.mnv_new_lane = -90.0
    c.fsm.mnv_phase = MnvPhase.ARC_FWD
    c.fsm.mnv_legs = 1
    c.heading = -40.0                              # already turned 40 deg
    c.fsm.mnv_leg_start_deg = 40.0                 # this leg starts here
    c.step()
    assert c.fsm.mnv_phase is MnvPhase.ARC_FWD, "must not end without moving"
    c.heading = -40.0 - p["MNV_LEG_STEP_DEG"] - 1
    c.step()
    assert c.fsm.mnv_phase is MnvPhase.STOP


def test_maneuver_finishes_when_facing_the_new_lane(p):
    p.set("USE_CORNER_MANEUVER", True)
    c = Car(p)
    c.go()
    c.fsm.clockwise = True
    c.fsm.lane_heading = 0.0
    c.fsm.state = State.CORNER_MANEUVER
    c.fsm.entered = True
    c.fsm.mnv_new_lane = -90.0
    c.fsm.mnv_phase = MnvPhase.ARC_FWD
    c.heading = -89.0                              # within MNV_DONE_DEG
    c.step()
    assert c.fsm.state is State.DRIVE_TO_CORNER
    assert c.fsm.lane_heading == pytest.approx(-90.0)


# ---------------------------------------------------------- housekeeping

def test_snapshot_is_json_safe(p):
    import json
    c = Car(p)
    c.go()
    c.step(n=3, advance_mm=100.0)
    json.dumps(c.fsm.snapshot())


def test_turn_is_clockwise_uses_the_colour_before_the_lock(p):
    c = Car(p)
    c.go()
    assert c.fsm.locked_colour == COLOUR_NONE
    assert not c.fsm.turn_is_clockwise(), "no colour seen yet"
    c.fsm.last_first_colour = COLOUR_ORANGE
    assert c.fsm.turn_is_clockwise()
    c.fsm.locked_colour = COLOUR_BLUE
    c.fsm.clockwise = False
    assert not c.fsm.turn_is_clockwise(), "after the lock, the lock wins"


# ------------------------------------------------ the corner exit's shape
#
# The exit is a COMMANDED quantity, not a by-product. Measured in the sim on
# a bare lap, the lane offset just after each corner, + = outer, against a
# LANE_LIMIT of 398:
#
#     TURN_FRONT_MM  0.55   0.70   0.85   1.00   <- TURN_LOCK_FRACTION
#           500      +382   +339   +282   +241
#           600      +335   +241   +190   +150
#           750      +194   +108    +51    -27
#           900       +48    -41   -109   -160
#
# So firing the arc while the wall ahead is still far, and tightening it,
# walks the car across the whole corridor - and nothing the car does AFTER
# the corner has that authority. Which of the three settings gets used is
# therefore the single most consequential decision in the corner.

def plan_exit(p, colour, clockwise=True, secondary=True):
    """Run _plan_corner_exit with a next-straight colour in place, and return
    (corner_exit_cmd, turn_front_cmd, turn_lock_cmd)."""
    c = Car(p)
    c.fsm.clockwise = clockwise
    if colour is not None:
        if secondary:
            c.fsm.planner.sec_seen_colour = colour
            c.fsm.planner.sec_seen_at = 0.0
        else:
            c.fsm.planner.next_straight_colour = colour
    c.fsm._plan_corner_exit(c.ctx())
    return (c.fsm.planner.corner_exit_cmd, c.fsm.turn_front_cmd,
            c.fsm.turn_lock_cmd)


def test_no_colour_gives_the_middle_arc(p):
    """Unknown means exit near the lane centre, which keeps BOTH passing
    sides reachable - the one setting that cannot be wrong."""
    cmd, front, lock = plan_exit(p, None)
    assert cmd == pytest.approx(p["CORNER_EXIT_MM"])
    assert front == pytest.approx(p["TURN_FRONT_MM"])
    assert lock == pytest.approx(p["TURN_LOCK_FRACTION"])


def test_an_inner_exit_turns_early_and_tight(p):
    """Clockwise the inner wall is on the right, so a red next - passed on
    its right - needs the inner side, and the arc has to be the short one."""
    cmd, front, lock = plan_exit(p, "RED", clockwise=True)
    assert cmd < 0                                   # CW: negative = inner
    assert front == pytest.approx(p["TURN_FRONT_INNER_MM"])
    assert lock == pytest.approx(p["TURN_LOCK_INNER"])


def test_an_outer_exit_runs_on_and_turns_loose(p):
    cmd, front, lock = plan_exit(p, "GREEN", clockwise=True)
    assert cmd > 0                                   # CW: positive = outer
    assert front == pytest.approx(p["TURN_FRONT_OUTER_MM"])
    assert lock == pytest.approx(p["TURN_LOCK_OUTER"])


def test_the_passing_side_is_the_lane_not_the_lap(p):
    """Red is passed on its right, which is the right-hand side of the LANE,
    whichever way round the car is going. Whether that is the inner or the
    outer side then depends on the direction - so the arc a red asks for
    inverts between CW and CCW. Tying red to 'inner' is right for CW and
    exactly backwards for CCW."""
    cw_cmd, cw_front, _ = plan_exit(p, "RED", clockwise=True)
    ccw_cmd, ccw_front, _ = plan_exit(p, "RED", clockwise=False)
    assert cw_cmd == pytest.approx(ccw_cmd), "the LANE side does not change"
    assert cw_front == pytest.approx(p["TURN_FRONT_INNER_MM"])
    assert ccw_front == pytest.approx(p["TURN_FRONT_OUTER_MM"])


def test_the_wide_arc_is_off_by_default(p):
    """TURN_FRONT_OUTER_MM equal to TURN_FRONT_MM means the outer case just
    gets the ordinary arc. In sim a genuinely wider arc bought an outer-side
    green about 60 mm of clearance and cost 5 finished runs in 24 to extra
    outer-wall contacts, so it ships off."""
    assert p["TURN_FRONT_OUTER_MM"] == pytest.approx(p["TURN_FRONT_MM"])
    assert p["TURN_LOCK_OUTER"] == pytest.approx(p["TURN_LOCK_FRACTION"])


def test_the_wide_arc_can_be_turned_on(p):
    p.set("TURN_FRONT_OUTER_MM", 500.0)
    p.set("TURN_LOCK_OUTER", 0.55)
    _, front, lock = plan_exit(p, "GREEN", clockwise=True)
    assert front == pytest.approx(500.0)
    assert lock == pytest.approx(0.55)


def test_arc_adaptation_can_be_switched_off(p):
    """The exit OFFSET still gets commanded; only the arc stops adapting."""
    p.set("CORNER_ARC_ADAPT", False)
    cmd, front, lock = plan_exit(p, "RED", clockwise=True)
    assert cmd < 0
    assert front == pytest.approx(p["TURN_FRONT_MM"])
    assert lock == pytest.approx(p["TURN_LOCK_FRACTION"])


def test_the_second_pillar_and_the_cross_corner_sighting_agree(p):
    """Same colour, same plan - they differ only in where the colour came
    from, which is a log line, not a decision."""
    a = plan_exit(p, "RED", secondary=True)
    b = plan_exit(p, "RED", secondary=False)
    assert a == b
