"""The moving navigator: zero-input start, tracking, recovery, sections."""

import math

import numpy as np
import pytest

from perception import sim
from perception.nav import Navigator, section_of
from nav.pillarmap import RED


def test_section_of_splits_straights_and_corners():
    assert section_of(0, 1000) == "N"
    assert section_of(0, -1000) == "S"
    assert section_of(1000, 0) == "E"
    assert section_of(-1000, 0) == "W"
    assert section_of(1000, 1000) == "NE"
    assert section_of(-1200, -1100) == "SW"


def _lap(nav, sc, laps=1, seed=3, step=70.0):
    segs = sim.scenario_segments(sc)
    rng = np.random.default_rng(seed)
    wpts = sim.corridor_waypoints(step_mm=step)
    states = []
    for _ in range(laps):
        for truth in wpts:
            r, _q = sim.synth_scan(truth, segs, rng=rng)
            nav.step(r)
            states.append(nav.state)
    return states


def test_zero_input_start_locks_and_infers_direction():
    nav = Navigator()                       # no start, no direction
    sc = sim.random_scenario(seed=7, n_pillars=6)
    states = _lap(nav, sc, laps=1)
    assert states[0] in ("INIT", "LOCKED")
    assert nav.state == "LOCKED"
    assert nav.direction in ("CW", "CCW")
    # the demo trajectory runs counter-clockwise
    assert nav.direction == "CCW"


def test_holds_lock_around_a_full_lap():
    nav = Navigator()
    sc = sim.random_scenario(seed=7, n_pillars=6)
    states = _lap(nav, sc, laps=1)
    locked = sum(1 for s in states if s == "LOCKED")
    assert locked / len(states) > 0.95
    assert nav.corners >= 3          # went round the track


def test_map_only_grows_while_locked():
    """A sign entered at an untrusted pose persists and poisons the plan, so
    the belief must not be updated while LOST."""
    nav = Navigator()
    sc = sim.random_scenario(seed=7, n_pillars=6)
    _lap(nav, sc, laps=1)
    before = len(nav.wb.pillars.confirmed())
    nav.state = "LOST"
    for _ in range(10):                      # garbage scans while lost
        nav.step([float("inf")] * 360)
    assert len(nav.wb.pillars.confirmed()) == before


def test_recovery_stays_in_the_same_frame():
    """After a loss the re-fit must find the car inside the frame it has been
    building, not jump to a different symmetric corridor."""
    nav = Navigator()
    sc = sim.random_scenario(seed=7, n_pillars=6)
    segs = sim.scenario_segments(sc)
    rng = np.random.default_rng(3)
    wpts = sim.corridor_waypoints(step_mm=70.0)
    for truth in wpts[:40]:
        nav.step(sim.synth_scan(truth, segs, rng=rng)[0])
    assert nav.state == "LOCKED"
    straight_before = nav.straight
    nav.state = "LOST"                       # force a recovery
    truth = wpts[40]
    for _ in range(5):
        nav.step(sim.synth_scan(truth, segs, rng=rng)[0])
    assert nav.state == "LOCKED"
    assert nav.straight in (straight_before, *"NESW")
    assert nav.direction == "CCW"            # direction is not re-litigated
