"""
Geometry tests for control/solver.py. These run on a laptop with no hardware —
run them before every mat session, they cost two seconds and they catch the
sign error that would otherwise steer into the pillar instead of around it.

    cd BlueprintPi && python3 -m pytest tests/ -q
"""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.solver import (AvoidCfg, solve, AVOID_NONE, AVOID_TRACK,
                            AVOID_COMMIT, wrap180)

CFG = AvoidCfg(wall_guard=False)          # guard tested separately
D = CFG.clearance_mm                       # 57.1 + 25 + 40 = 122.1 mm


def face(r_axis):
    """Turn an axis range into the face range the lidar actually reports."""
    return r_axis - CFG.pillar_half_width_mm


# ---------------------------------------------------------------- clearance --

def test_clearance_is_half_car_plus_half_pillar_plus_margin():
    assert CFG.clearance_mm == pytest.approx(57.1 + 25.0 + 40.0)


# ------------------------------------------------------------------- sides --

def test_red_dead_ahead_steers_right():
    s = solve("RED", 0.0, face(700), CFG)
    assert s.action == AVOID_TRACK
    assert s.theta_deg < 0                      # negative = right
    assert s.delta_deg == pytest.approx(math.degrees(math.asin(D / 700)), abs=1e-6)


def test_green_dead_ahead_steers_left():
    s = solve("GREEN", 0.0, face(700), CFG)
    assert s.action == AVOID_TRACK
    assert s.theta_deg > 0


def test_theta_is_bearing_plus_signed_delta():
    for color, sign in (("RED", -1), ("GREEN", +1)):
        for b in (-20.0, -5.0, 0.0, 5.0, 20.0):
            s = solve(color, b, face(600), CFG)
            if s.action == AVOID_NONE:
                continue
            assert s.theta_deg == pytest.approx(b + sign * s.delta_deg, abs=1e-6)


# ------------------------------------------------------------ already clear --

def test_red_already_left_of_path_is_a_no_op():
    r = 600.0
    delta = math.degrees(math.asin(D / r))
    assert solve("RED", delta + 2.0, face(r), CFG).action == AVOID_NONE
    assert solve("RED", delta - 2.0, face(r), CFG).action != AVOID_NONE


def test_green_already_right_of_path_is_a_no_op():
    r = 600.0
    delta = math.degrees(math.asin(D / r))
    assert solve("GREEN", -delta - 2.0, face(r), CFG).action == AVOID_NONE
    assert solve("GREEN", -delta + 2.0, face(r), CFG).action != AVOID_NONE


def test_wrong_side_pillar_gets_a_bigger_correction():
    """A RED pillar already on our right must be crossed to, not drifted past."""
    near = solve("RED", 0.0, face(600), CFG)
    wrong = solve("RED", -15.0, face(600), CFG)
    assert abs(wrong.theta_deg) > abs(near.theta_deg)


# --------------------------------------------------------------------- leg --

def test_leg_is_the_tangent_length_plus_tail():
    r = 700.0
    s = solve("RED", 0.0, face(r), CFG)
    expect = math.sqrt(r * r - D * D) + CFG.tail_clear_mm
    assert s.leg_mm == pytest.approx(expect, abs=1e-6)


def test_leg_shrinks_as_the_pillar_gets_closer():
    legs = [solve("RED", 0.0, face(r), CFG).leg_mm for r in (900, 700, 500, 400)]
    assert legs == sorted(legs, reverse=True)


# ------------------------------------------------------------ track/commit --

def test_track_until_freeze_range_then_commit():
    assert solve("RED", 0.0, face(500), CFG).action == AVOID_TRACK
    assert solve("RED", 0.0, face(CFG.freeze_mm), CFG).action == AVOID_COMMIT
    assert solve("RED", 0.0, face(300), CFG).action == AVOID_COMMIT


def test_beyond_engage_range_is_silent():
    assert solve("RED", 0.0, face(CFG.engage_mm + 50), CFG).action == AVOID_NONE


# ---------------------------------------------------------------- rejects --

def test_non_pillar_colours_and_bad_ranges_are_rejected():
    assert solve("MAGENTA", 0.0, 500.0, CFG).action == AVOID_NONE
    assert solve("RED", 0.0, float("inf"), CFG).action == AVOID_NONE
    assert solve("RED", 0.0, 50.0, CFG).action == AVOID_NONE          # inside min_range
    assert solve("RED", 70.0, face(500), CFG).action == AVOID_NONE    # outside the cone


def test_inside_the_keepout_circle_saturates_instead_of_raising():
    # Unreachable with the shipped min_range_mm (the chassis floor rejects it
    # first), which is the point — but the arcsin must not blow up if it ever is.
    cfg = AvoidCfg(wall_guard=False, min_range_mm=30.0)
    s = solve("RED", 0.0, face(D * 0.9), cfg)
    assert s.action == AVOID_COMMIT
    assert math.isfinite(s.theta_deg)
    assert abs(s.theta_deg) < 90.0


def test_theta_is_clamped_to_max_heading():
    cfg = AvoidCfg(wall_guard=False, max_heading_deg=20.0)
    s = solve("RED", -30.0, face(300), cfg)
    assert abs(s.theta_deg) <= 20.0 + 1e-9


# ------------------------------------------------------------- wall guard --

def test_wall_guard_reduces_theta_when_the_wall_is_close():
    cfg = AvoidCfg(wall_guard=True)
    free = solve("RED", 0.0, face(600), cfg, side_free_mm=float("inf"))
    tight = solve("RED", 0.0, face(600), cfg, side_free_mm=250.0)
    assert abs(tight.theta_deg) < abs(free.theta_deg)


def test_wall_guard_refuses_when_there_is_no_room_at_all():
    cfg = AvoidCfg(wall_guard=True)
    s = solve("RED", 0.0, face(600), cfg, side_free_mm=100.0)
    assert s.action == AVOID_NONE
    assert "no room" in s.reason


# ------------------------------------------------------------------ wrap --

def test_wrap180():
    assert wrap180(190.0) == pytest.approx(-170.0)
    assert wrap180(-190.0) == pytest.approx(170.0)
    assert wrap180(180.0) == pytest.approx(180.0)
