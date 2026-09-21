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
    # facing +x in the south corridor centre; nearest wall ahead is the outer
    # corner far away, but the ray straight up (index 90) hits the inner wall.
    sc = sim.Scenario(pillars={}, parking_side="N")
    segs = sim.scenario_segments(sc)
    r, q = sim.synth_scan((0.0, -1000.0, 0.0), segs,
                          rng=np.random.default_rng(0),
                          range_sigma=0.0, dropout_p=0.0)
    # ray at +90 deg (north) from y=-1000 hits inner wall at y=-500 -> 500 mm
    assert r[90] == pytest.approx(500.0, abs=5.0)
    # ray at -90 (index 270, south) hits outer wall at y=-1500 -> 500 mm
    assert r[270] == pytest.approx(500.0, abs=5.0)
