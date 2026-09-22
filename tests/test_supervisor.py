"""
Lifecycle tests for control/supervisor.py — the TRACK -> COMMIT -> IDLE machine,
driven by fake obstacles and fake telemetry. No hardware, no serial port.

    cd BlueprintPi && python3 -m pytest tests/ -q
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.solver import AvoidCfg, AVOID_NONE, AVOID_TRACK, AVOID_COMMIT
from control.supervisor import AvoidSupervisor
from control.percept_link import ST_HEADING, ST_TURN90, ST_AVOID, ST_STOPPED

CFG = AvoidCfg(wall_guard=False)


@dataclass
class Obs:
    color: str
    bearing_deg: float
    distance_mm: float


@dataclass
class Tel:
    state: int = ST_HEADING
    heading_deg: float = 0.0
    odo_mm: float = 0.0


def sup():
    return AvoidSupervisor(CFG, confirm_ticks=3, refractory_mm=150.0)


def run(s, obstacles, telem, n=1, t0=100.0):
    """Tick n times against a static world; return the last decision."""
    out = None
    for i in range(n):
        out = s.tick(obstacles, telem, t0 + i * 0.02, float("inf"), float("inf"))
    return out


# ----------------------------------------------------------------- arming --

def test_a_single_frame_does_not_start_a_manoeuvre():
    s = sup()
    action, *_ = run(s, [Obs("RED", 0.0, 600.0)], Tel(), n=1)
    assert action == AVOID_NONE


def test_confirm_ticks_then_track():
    s = sup()
    action, color, heading, leg, _ = run(s, [Obs("RED", 0.0, 600.0)], Tel(), n=3)
    assert action == AVOID_TRACK
    assert color == "RED"
    assert heading < 0            # RED -> pass on its right -> steer right
    assert leg > 0


def test_heading_is_absolute_not_relative():
    s = sup()
    _, _, heading, _, _ = run(s, [Obs("RED", 0.0, 600.0)], Tel(heading_deg=90.0), n=3)
    # 90 deg lane heading plus a right-hand offset: still near 90, below it.
    assert 70.0 < heading < 90.0


# ---------------------------------------------------------------- commit ---

def test_close_range_commits():
    s = sup()
    action, *_ = run(s, [Obs("RED", 0.0, 250.0)], Tel(), n=3)
    assert action == AVOID_COMMIT


def test_losing_the_pillar_mid_track_freezes_the_last_solution():
    s = sup()
    run(s, [Obs("GREEN", 0.0, 600.0)], Tel(), n=3)
    frozen = s.heading_abs
    action, _, heading, _, _ = run(s, [], Tel(), n=3)   # camera loses it
    assert action == AVOID_COMMIT
    assert heading == frozen


def test_a_one_tick_dropout_does_not_freeze():
    s = sup()
    run(s, [Obs("GREEN", 0.0, 600.0)], Tel(), n=3)
    action, *_ = run(s, [], Tel(), n=1)
    assert action == AVOID_TRACK


def test_commit_runs_out_on_odometry_then_releases():
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(odo_mm=1000.0), n=3)
    leg = s.leg_mm
    action, _, _, remaining, _ = run(s, [], Tel(odo_mm=1000.0 + leg / 2), n=1)
    assert action == AVOID_COMMIT
    assert remaining < leg
    action, *_ = run(s, [], Tel(odo_mm=1000.0 + leg + 1), n=1)
    assert action == AVOID_NONE


def test_commit_survives_the_pillar_being_invisible():
    """The whole point of COMMIT: perception is not consulted at all."""
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(odo_mm=0.0), n=3)
    action, *_ = run(s, [], Tel(odo_mm=10.0), n=1)
    assert action == AVOID_COMMIT


# -------------------------------------------------------------- releases ---

def test_already_clear_releases_a_track():
    s = sup()
    run(s, [Obs("RED", 0.0, 600.0)], Tel(), n=3)
    # Now the same pillar is well off to our left: nothing to do.
    action, *_ = run(s, [Obs("RED", 40.0, 600.0)], Tel(), n=1)
    assert action == AVOID_NONE


def test_refractory_blocks_immediate_re_engagement():
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(odo_mm=0.0), n=3)
    leg = s.leg_mm
    run(s, [], Tel(odo_mm=leg + 1), n=1)                      # leg completes
    action, *_ = run(s, [Obs("RED", 0.0, 600.0)],
                     Tel(odo_mm=leg + 50), n=5)               # inside refractory
    assert action == AVOID_NONE
    action, *_ = run(s, [Obs("RED", 0.0, 600.0)],
                     Tel(odo_mm=leg + 400), n=5)              # past it
    assert action == AVOID_TRACK


def test_a_turn_cancels_everything():
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(), n=3)
    action, *_ = run(s, [Obs("RED", 0.0, 250.0)], Tel(state=ST_TURN90), n=1)
    assert action == AVOID_NONE
    assert s.action == AVOID_NONE


def test_stm32_already_avoiding_does_not_cancel():
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(), n=3)
    action, *_ = run(s, [], Tel(state=ST_AVOID, odo_mm=1.0), n=1)
    assert action == AVOID_COMMIT


def test_nearest_pillar_wins():
    s = sup()
    _, color, *_ = run(s, [Obs("RED", 10.0, 800.0), Obs("GREEN", -5.0, 400.0)],
                       Tel(), n=3)
    assert color == "GREEN"


def test_a_dashboard_stop_cancels_everything():
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(), n=3)
    action, _, _, _, note = run(s, [Obs("RED", 0.0, 250.0)],
                                Tel(state=ST_STOPPED), n=1)
    assert action == AVOID_NONE and note == "stm32 busy"
    assert s.action == AVOID_NONE


def test_snapshot_reports_the_lifecycle():
    s = sup()
    run(s, [Obs("GREEN", 0.0, 600.0)], Tel(), n=2)
    snap = s.snapshot()
    assert snap["confirm"] == 2 and snap["confirm_ticks"] == 3
    assert snap["action"] == AVOID_NONE
    run(s, [Obs("GREEN", 0.0, 600.0)], Tel(), n=1)
    assert s.snapshot()["action"] == AVOID_TRACK
