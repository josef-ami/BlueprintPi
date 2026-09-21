"""Systematic start-up fit: walls + seats + parking, and its honest limits."""

import math

import numpy as np
import pytest

from perception import sim, fit as fitmod
from perception.state import WorldBelief
from nav.pillarmap import RED, GREEN
from nav import arena


def _scan(truth, sc, seed=0):
    return sim.synth_scan(truth, sim.scenario_segments(sc),
                          rng=np.random.default_rng(seed))[0]


def test_corridor_distances_match_the_corridor():
    sc = sim.Scenario(pillars={}, parking_side="N")
    r = _scan((0.0, -1000.0, 0.0), sc)          # mid south corridor, facing east
    d = fitmod.corridor_distances(r)
    # left wall (inner, y=-500) and right wall (outer, y=-1500) are 500 each
    assert d["left"] == pytest.approx(500, abs=60)
    assert d["right"] == pytest.approx(500, abs=60)
    assert d["left"] + d["right"] == pytest.approx(1000, abs=90)


def test_fit_recovers_pose_when_the_straight_is_known():
    sc = sim.Scenario(pillars={"S1": RED, "S4": GREEN},
                      parking_side="S", parking_along=0.0)
    truth = (-200.0, -1000.0, 0.0)
    wb = WorldBelief()
    res = wb.fit_start(_scan(truth, sc), sides=("S",))
    assert res is not None
    assert math.hypot(res.pose[0] - truth[0], res.pose[1] - truth[1]) < 60
    assert abs(math.degrees(res.pose[2] - truth[2])) < 3


def test_obstacles_on_seats_raise_the_seat_score():
    truth = (-200.0, -1000.0, 0.0)
    with_p = sim.Scenario(pillars={"S1": RED, "S4": GREEN}, parking_side="N")
    without = sim.Scenario(pillars={}, parking_side="N")
    a = WorldBelief().fit_start(_scan(truth, with_p), sides=("S",))
    b = WorldBelief().fit_start(_scan(truth, without), sides=("S",))
    assert a.seat > b.seat
    assert a.hits >= 1 and a.orphans == 0


def test_parking_pattern_is_two_close_lines_on_an_outer_wall():
    bay = arena.parking_bay("S", 0.0, 175.0)
    # two clusters hugging the south outer wall, 1.5*car_len apart
    clusters = [(-130.0, -1400.0, 4, 30.0), (130.0, -1400.0, 4, 30.0)]
    found, used = fitmod.find_parking(clusters, car_len_mm=175.0)
    assert found is not None and found.side == "S"
    assert used == {0, 1}
    # a single cluster is not a parking bay
    none, _ = fitmod.find_parking([clusters[0]], car_len_mm=175.0)
    assert none is None


def test_full_search_reports_rotational_ambiguity():
    """Honesty check: with all four straights allowed, a symmetric scene must
    say it is ambiguous rather than claim a corridor it cannot know."""
    sc = sim.Scenario(pillars={"S1": RED}, parking_side="S", parking_along=0.0)
    res = WorldBelief().fit_start(_scan((-200.0, -1000.0, 0.0), sc))
    assert res is not None
    assert res.ambiguous_with, "a 4-fold symmetric mat must report ambiguity"
