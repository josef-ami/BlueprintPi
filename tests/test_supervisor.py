"""
Lifecycle tests for control/supervisor.py — the NONE <-> TRACK machine, driven
by fake obstacles and fake telemetry. No hardware, no serial port.

There is no COMMIT any more (see the module docstring in supervisor.py): every
manoeuvre that ever leaves this file is NONE or TRACK, and release back to
NONE always happens the instant one of the release conditions is true — never
on a distance run out over odometry, never on a timer.

    cd BlueprintPi && python3 -m pytest tests/ -q
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.solver import AvoidCfg, AVOID_NONE, AVOID_TRACK
from control.supervisor import AvoidSupervisor, LOST_GRACE_TICKS
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
    assert leg > 0                # the backstop leg, not a real target


def test_heading_is_absolute_not_relative():
    s = sup()
    _, _, heading, _, _ = run(s, [Obs("RED", 0.0, 600.0)], Tel(heading_deg=90.0), n=3)
    # 90 deg lane heading plus a right-hand offset: still near 90, below it.
    assert 70.0 < heading < 90.0


# ------------------------------------------------------- close range ------

def test_close_range_still_sends_track_but_freezes_the_heading():
    """
    Inside freeze_mm the solver itself would hand back a near-degenerate
    large-angle answer (see solve()'s own docstring on why). The supervisor's
    fix is to simply stop adopting new headings once it is that close — it
    keeps sending TRACK (there is no third wire value for "close" any more),
    but heading_abs is whatever it already was.
    """
    s = sup()
    run(s, [Obs("RED", 10.0, 600.0)], Tel(), n=3)
    assert s.action == AVOID_TRACK
    far_heading = s.heading_abs

    # Same pillar, now close (r_axis <= freeze_mm): the keep-out half-angle
    # is bigger up close, so re-solving at the same bearing would swing the
    # heading a long way. If this were being re-adopted, it would jump.
    action, _, heading, _, _ = run(s, [Obs("RED", 10.0, 280.0)], Tel(), n=1)
    assert action == AVOID_TRACK
    assert heading == far_heading


# -------------------------------------------------------------- releases ---

def test_a_one_tick_dropout_does_not_release_yet():
    s = sup()
    run(s, [Obs("GREEN", 0.0, 600.0)], Tel(), n=3)
    action, *_ = run(s, [], Tel(), n=1)
    assert action == AVOID_TRACK


def test_losing_the_pillar_mid_track_releases_instantly():
    """
    Change 1: no more leg to run out. Once the pillar has been missing for
    LOST_GRACE_TICKS in a row, release happens on that exact tick — the
    firmware's own heading PID is what closes the gap back to lane heading
    from here, not a distance the Pi counted down.
    """
    s = sup()
    run(s, [Obs("GREEN", 0.0, 600.0)], Tel(), n=3)
    assert s.action == AVOID_TRACK
    action, *_ = run(s, [], Tel(), n=LOST_GRACE_TICKS - 1)
    assert action == AVOID_TRACK              # still within grace
    action, _, _, _, note = run(s, [], Tel(), n=1)
    assert action == AVOID_NONE
    assert "out of view" in note
    assert s.action == AVOID_NONE


def test_close_range_also_releases_after_losing_sight():
    """The old COMMIT ran out on odometry alone and never looked at the
    camera again. That is gone: even a close-range TRACK is released by the
    same grace-tick rule as everything else — there is nothing left that
    ignores perception."""
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(odo_mm=0.0), n=3)
    assert s.action == AVOID_TRACK
    action, *_ = run(s, [], Tel(odo_mm=10.0), n=LOST_GRACE_TICKS)
    assert action == AVOID_NONE


def test_already_clear_releases_a_track():
    s = sup()
    run(s, [Obs("RED", 0.0, 600.0)], Tel(), n=3)
    # Now the same pillar is well off to our left: nothing to do.
    action, *_ = run(s, [Obs("RED", 40.0, 600.0)], Tel(), n=1)
    assert action == AVOID_NONE


def test_refractory_blocks_immediate_re_engagement():
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(odo_mm=0.0), n=3)
    assert s.action == AVOID_TRACK
    # Pillar leaves view; the grace window runs out at odo 10 and that is
    # where the refractory zone starts (refractory_mm=150 from there).
    run(s, [], Tel(odo_mm=10.0), n=LOST_GRACE_TICKS - 1)
    action, *_ = run(s, [], Tel(odo_mm=10.0), n=1)
    assert action == AVOID_NONE and s.action == AVOID_NONE
    action, *_ = run(s, [Obs("RED", 0.0, 250.0)],
                     Tel(odo_mm=60.0), n=5)                # inside refractory
    assert action == AVOID_NONE
    action, *_ = run(s, [Obs("RED", 0.0, 250.0)],
                     Tel(odo_mm=200.0), n=5)                # past it
    assert action == AVOID_TRACK


def test_a_turn_cancels_everything():
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(), n=3)
    action, *_ = run(s, [Obs("RED", 0.0, 250.0)], Tel(state=ST_TURN90), n=1)
    assert action == AVOID_NONE
    assert s.action == AVOID_NONE


def test_stm32_already_avoiding_does_not_cancel():
    """ST_AVOID is not one of the busy states the supervisor defers to (it is
    the manoeuvre we ourselves are running) — a single missed detection here
    is just the ordinary one-tick grace, not an interruption."""
    s = sup()
    run(s, [Obs("RED", 0.0, 250.0)], Tel(), n=3)
    action, *_ = run(s, [], Tel(state=ST_AVOID, odo_mm=1.0), n=1)
    assert action == AVOID_TRACK


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
    assert "leg_mm" not in snap and "commit_odo" not in snap
    run(s, [Obs("GREEN", 0.0, 600.0)], Tel(), n=1)
    assert s.snapshot()["action"] == AVOID_TRACK
