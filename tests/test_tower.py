"""LazyGo valley test: a sign is a deep, ~50 mm-wide valley in the raw scan."""

import math

import numpy as np
import pytest

from perception import sim
from perception.tower import find_towers, tower_points
from nav import arena
from nav.pillarmap import RED, GREEN

CC = arena.CORRIDOR_CENTER
POSE = (-300.0, -CC, 0.0)          # south corridor, facing east


def _scan(sc, pose=POSE, seed=0):
    return sim.synth_scan(pose, sim.scenario_segments(sc),
                          rng=np.random.default_rng(seed))[0]


def test_a_sign_is_a_valley_of_about_fifty_millimetres():
    sc = sim.Scenario(pillars={"S1": RED}, parking_side="N")
    towers = find_towers(_scan(sc))
    assert len(towers) == 1
    t = towers[0]
    assert 25 <= t.size_mm <= 95          # s = r*theta, a 50 mm sign
    assert t.depth_mm > 200               # stands clear of the background


def test_parking_blocks_are_not_towers():
    """The decisive one: a block is flush to the outer wall, so its valley is
    too shallow to qualify. This is what stops the bay being read as pillars,
    and it needs no map and no pose to do it."""
    sc = sim.Scenario(pillars={}, parking_side="S", parking_along=0.0)
    assert find_towers(_scan(sc)) == []


def test_sign_found_even_with_parking_in_view():
    sc = sim.Scenario(pillars={"S1": RED}, parking_side="S", parking_along=0.0)
    towers = find_towers(_scan(sc))
    assert len(towers) == 1
    seat, d = arena.nearest_seat(*tower_points(_scan(sc), POSE, towers)[0], tol=1e9)
    assert seat.id == "S1" and d < 90


def test_towers_land_on_their_seats():
    sc = sim.Scenario(pillars={"S1": RED, "S4": GREEN}, parking_side="N")
    r = _scan(sc)
    towers = find_towers(r)
    assert len(towers) == 2
    ids = set()
    for (x, y) in tower_points(r, POSE, towers):
        seat, d = arena.nearest_seat(x, y, tol=1e9)
        assert d < 90
        ids.add(seat.id)
    assert ids == {"S1", "S4"}


def test_dropouts_inside_an_object_do_not_lose_it():
    """A real scanner drops the odd ray off a dark facet. One inf in the middle
    of a sign used to abandon the whole valley."""
    sc = sim.Scenario(pillars={"S1": RED}, parking_side="N")
    r = np.asarray(_scan(sc), dtype=np.float64).copy()
    t = find_towers(r)[0]
    mid = (t.i0 + t.i1) // 2
    r[mid] = np.inf                       # punch a hole straight through it
    again = find_towers(r)
    assert len(again) == 1
    assert abs(again[0].bearing_deg - t.bearing_deg) < 6
