"""
The mapper: the last thing between the FSM and the wire.

Its clamps are a second line of defence on top of the firmware's own. A bug
above it should produce a slow wrong turn, not a servo against its stop.
"""

import math

import pytest

from control.intent import ActionIntent, SteerMode
from control.mapper import (CommandMapper, KINEMATIC_RADIUS_MM,
                            MEASURED_RADIUS_MM, SPEED_MIN_MOVING,
                            STEER_LOCK_DEG, WHEELBASE_MM, radius_for_steer,
                            steer_for_lateral_shift, steer_for_radius)


def test_stop_stays_stopped():
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.STOP, speed_pwm=200,
                             steer_deg=99.0))
    assert out.mode is SteerMode.STOP
    assert out.speed_pwm == 0


def test_steer_is_clamped_to_the_mechanical_lock():
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.DIRECT, steer_deg=200.0,
                             speed_pwm=60))
    assert out.steer_deg == pytest.approx(STEER_LOCK_DEG)
    assert "clamped" in m.clamped
    out = m.map(ActionIntent(mode=SteerMode.DIRECT, steer_deg=-200.0,
                             speed_pwm=60))
    assert out.steer_deg == pytest.approx(-STEER_LOCK_DEG)


def test_a_creep_below_the_stall_is_raised():
    """Asking for a speed the motor cannot deliver just winds up the PID."""
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.HEADING_HOLD, speed_pwm=5))
    assert out.speed_pwm == SPEED_MIN_MOVING
    assert "stall" in m.clamped


def test_zero_speed_is_left_alone():
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.HEADING_HOLD, speed_pwm=0))
    assert out.speed_pwm == 0


def test_speed_is_clamped_to_the_ceiling():
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.HEADING_HOLD, speed_pwm=400))
    assert out.speed_pwm == 255


def test_arc_lock_is_clamped():
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.ARC, speed_pwm=60, arc_lock=5.0))
    assert out.arc_lock == pytest.approx(1.0)
    out = m.map(ActionIntent(mode=SteerMode.ARC, speed_pwm=60, arc_lock=0.0))
    assert out.arc_lock == pytest.approx(0.20)


def test_heading_is_wrapped():
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.HEADING_HOLD,
                             target_heading_deg=270.0, speed_pwm=60))
    assert out.target_heading_deg == pytest.approx(-90.0)
    out = m.map(ActionIntent(mode=SteerMode.HEADING_HOLD,
                             target_heading_deg=-450.0, speed_pwm=60))
    assert out.target_heading_deg == pytest.approx(-90.0)


def test_mapping_does_not_mutate_the_original():
    m = CommandMapper()
    original = ActionIntent(mode=SteerMode.DIRECT, steer_deg=200.0,
                            speed_pwm=60)
    m.map(original)
    assert original.steer_deg == 200.0


def test_reverse_and_reason_survive():
    m = CommandMapper()
    out = m.map(ActionIntent(mode=SteerMode.DIRECT, steer_deg=10.0,
                             speed_pwm=60, reverse=True, reason="3pt"))
    assert out.reverse
    assert out.reason == "3pt"


def test_clamped_is_empty_when_nothing_was_changed():
    m = CommandMapper()
    m.map(ActionIntent(mode=SteerMode.HEADING_HOLD, steer_deg=0.0,
                       speed_pwm=60, arc_lock=0.7))
    assert m.clamped == ""


# ------------------------------------------------------------- geometry

def test_steer_and_radius_are_inverses():
    for r in (250.0, 400.0, 1000.0):
        assert radius_for_steer(steer_for_radius(r)) == pytest.approx(r)


def test_steer_for_radius_saturates_at_the_lock():
    assert steer_for_radius(10.0) == pytest.approx(STEER_LOCK_DEG)
    assert steer_for_radius(-10.0) == pytest.approx(-STEER_LOCK_DEG)


def test_straight_wheels_are_an_infinite_radius():
    assert radius_for_steer(0.0) == float("inf")


def test_the_two_radii_disagree_and_that_is_deliberate():
    """The bicycle model assumes no tyre slip; at full lock on a mat there
    is plenty. The planner's reach estimate must use the MEASURED figure,
    because one that believes the car turns tighter than it does will commit
    to a pass it cannot make."""
    assert KINEMATIC_RADIUS_MM == pytest.approx(
        WHEELBASE_MM / math.tan(math.radians(STEER_LOCK_DEG)))
    assert MEASURED_RADIUS_MM > KINEMATIC_RADIUS_MM


def test_the_planner_default_is_the_measured_radius(p):
    assert p["TURN_RADIUS_MM"] == pytest.approx(MEASURED_RADIUS_MM)


def test_lateral_shift_is_signed_and_grows_with_the_shift():
    assert steer_for_lateral_shift(100.0, 500.0) > 0
    assert steer_for_lateral_shift(-100.0, 500.0) < 0
    assert steer_for_lateral_shift(200.0, 500.0) > \
        steer_for_lateral_shift(100.0, 500.0)
    assert steer_for_lateral_shift(100.0, 0.0) == 0.0
