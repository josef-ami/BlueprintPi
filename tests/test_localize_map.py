"""Map-based localization recovers pose from synthetic scans."""

import math

import numpy as np
import pytest

from perception import sim
from perception.state import WorldBelief


def _scan(pose, sc=None, seed=0):
    sc = sc or sim.Scenario(pillars={}, parking_side="N")
    segs = sim.scenario_segments(sc)
    return sim.synth_scan(pose, segs, rng=np.random.default_rng(seed))[0]


def test_global_init_recovers_corner_pose():
    true = (-1000.0, -1000.0, math.radians(3))
    r = _scan(true)
    wb = WorldBelief()
    guess = (true[0] + 70, true[1] - 55, true[2] + math.radians(5))
    (x, y, th), score = wb.global_init(r, guess=guess)
    assert score > 0.6
    assert math.hypot(x - true[0], y - true[1]) < 30
    assert abs(math.degrees(th - true[2])) < 2.0


def test_straight_recovers_cross_track_and_heading():
    true = (0.0, -1000.0, 0.0)                 # mid south straight
    r = _scan(true)
    wb = WorldBelief()
    guess = (true[0] + 40, true[1] - 40, math.radians(4))
    (x, y, th), score = wb.global_init(r, guess=guess)
    assert abs(y - true[1]) < 20               # cross-track fixed
    assert abs(math.degrees(th)) < 2.0         # heading fixed


def test_lidar_only_tracking_pins_lateral_and_heading():
    # LiDAR-only (no odometry): drive down the south straight and check that
    # cross-track and heading stay tight every frame. Along-track is allowed to
    # be rough here - that is the documented limitation the encoder fixes later.
    wb = WorldBelief()
    start = (-250.0, -1000.0, 0.0)
    wb.global_init(_scan(start), guess=start)
    worst_cross = worst_head = 0.0
    for i in range(1, 9):
        truth = (start[0] + 60.0 * i, start[1], 0.0)
        wb.track(_scan(truth, seed=i))     # only the scan, no predict()
        x, y, th = wb.pose
        worst_cross = max(worst_cross, abs(y - truth[1]))
        worst_head = max(worst_head, abs(math.degrees(th)))
    assert worst_cross < 25.0
    assert worst_head < 2.5
