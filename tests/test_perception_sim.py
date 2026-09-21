"""Synthetic RPLidar: scans are well-formed and geometry is consistent."""

import math

import numpy as np
import pytest

from perception import sim
from nav.pillarmap import RED, GREEN


def test_scenario_segment_count():
    sc = sim.Scenario(pillars={"S0": RED, "S3": GREEN}, parking_side="N")
    segs = sim.scenario_segments(sc)
    # 8 walls + 4 per pillar (2) + 4 per parking block (2) = 8 + 8 + 8
    assert segs.shape == (24, 4)


def test_synth_scan_shape_and_returns():
    sc = sim.Scenario(pillars={}, parking_side="N")
    segs = sim.scenario_segments(sc)
    r, q = sim.synth_scan((-1000.0, -1000.0, 0.0), segs,
                          rng=np.random.default_rng(0))
    assert r.shape == (360,) and q.shape == (360,)
    finite = np.isfinite(r)
    assert finite.sum() > 300                     # a corner sees most rays
    assert r[finite].min() > 0


def test_scan_distance_matches_geometry_forward():
    """Sitting mid-corridor, both side rays must read half the corridor.

    Derived from geom, never hard-coded: the arena dimensions come from
    arena_cal.json once the mat has been measured, so a literal 500 here would
    assert the rulebook rather than the mat. This one measured
    2990/1090/950, making the half-corridor 475 mm.
    """
    from nav import arena, geom
    sc = sim.Scenario(pillars={}, parking_side="N")
    segs = sim.scenario_segments(sc)
    cc = arena.CORRIDOR_CENTER
    r, q = sim.synth_scan((0.0, -cc, 0.0), segs,
                          rng=np.random.default_rng(0),
                          range_sigma=0.0, dropout_p=0.0)
    to_inner = cc - geom.INNER / 2.0
    to_outer = geom.OUTER / 2.0 - cc
    assert to_inner == pytest.approx(geom.CORRIDOR / 2.0, abs=1.0)
    assert r[90] == pytest.approx(to_inner, abs=5.0)     # +90 deg = inner wall
    assert r[270] == pytest.approx(to_outer, abs=5.0)    # -90 deg = outer wall
