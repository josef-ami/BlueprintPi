"""
The LiDAR geometry: wall cone fits, pillar location, candidate clustering.

Every test builds a synthetic scan - a corridor, a pillar, a wall run - so the
maths is checked against a shape whose answer is known by construction rather
than against a recorded scan nobody can reason about.
"""

import math

import pytest

from sensors.lidar import (cones, fit_wall, lidar_candidates, locate_pillar,
                           pick_bearing, read_three, sector_min, u16,
                           unclassified, INVALID)
from worldstate import WallFit

INF = float("inf")


# -------------------------------------------------------- scan builders

def blank():
    return [INF] * 360


def add_wall(ranges, perp_mm, side, yaw_deg=0.0, span=60):
    """A straight wall at `perp_mm` perpendicular distance.

    side 'left' puts it at +90 deg, 'right' at -90. yaw_deg rotates the CAR,
    so the wall line tilts the other way in the car frame.
    """
    sign = 1.0 if side == "left" else -1.0
    for d in range(-span // 2, span // 2 + 1):
        a = 90.0 * sign + d
        # distance to a line at perpendicular distance perp, normal along
        # (90*sign - yaw)
        normal = math.radians(90.0 * sign - yaw_deg)
        ang = math.radians(a)
        cos_off = math.cos(ang - normal)
        if cos_off <= 0.05:
            continue
        r = perp_mm / cos_off
        if r > 4000:
            continue
        ranges[int(round(a)) % 360] = r
    return ranges


def add_point_cluster(ranges, x, y, n=4, spread_mm=20):
    """A small object at (x fwd, y left)."""
    for k in range(n):
        off = (k - (n - 1) / 2.0) * spread_mm
        px, py = x, y + off
        a = math.degrees(math.atan2(py, px)) % 360
        r = math.hypot(px, py)
        idx = int(round(a)) % 360
        if r < ranges[idx]:
            ranges[idx] = r
    return ranges


# ------------------------------------------------------------- the beams

def test_u16_sentinel():
    assert u16(None) == INVALID
    assert u16(INF) == INVALID
    assert u16(-1) == INVALID
    assert u16(70000) == INVALID
    assert u16(1234.6) == 1235


def test_sector_min():
    r = blank()
    r[0] = 1000.0
    r[5] = 400.0
    r[355] = 600.0
    assert sector_min(r, 0, 10) == 400.0
    assert sector_min(r, 180, 10) == INF


def test_pick_bearing_prefers_quality_then_closeness():
    r, q = blank(), [0] * 360
    r[88], q[88] = 500.0, 10
    r[90], q[90] = 900.0, 30
    r[92], q[92] = 400.0, 30
    p = pick_bearing(r, q, 90, 5)
    assert p[0] == 90, "a tie in quality breaks toward the centre"
    assert p[2] == 30


def test_pick_bearing_none_when_empty():
    assert pick_bearing(blank(), [0] * 360, 90, 5) is None


def test_read_three_order_is_front_left_right():
    r, q = blank(), [1] * 360
    r[0], r[90], r[270] = 1000.0, 500.0, 700.0
    assert read_three(r, q, 1) == (1000.0, 500.0, 700.0)


# --------------------------------------------------------- the cone fits

def test_fit_wall_finds_a_straight_wall():
    r = add_wall(blank(), 500.0, "left")
    got = fit_wall(r, 90)
    assert got is not None
    perp, yaw, n, m, c = got
    assert perp == pytest.approx(500.0, rel=0.02)
    assert yaw == pytest.approx(0.0, abs=1.0)


def test_fit_wall_reports_the_car_yaw():
    """+ yaw = the car points LEFT of the wall direction."""
    r = add_wall(blank(), 500.0, "left", yaw_deg=10.0)
    perp, yaw, n, m, c = fit_wall(r, 90)
    assert perp == pytest.approx(500.0, rel=0.05)
    assert yaw == pytest.approx(10.0, abs=2.0)


def test_fit_wall_none_when_nothing_is_there():
    assert fit_wall(blank(), 90) is None


def test_fit_wall_rejects_a_lone_pillar():
    """A 50 mm pillar face cannot span CONE_MIN_SPAN_MM, which is the whole
    point of the span test."""
    r = add_point_cluster(blank(), 100.0, 400.0, n=4, spread_mm=12)
    assert fit_wall(r, 90) is None


def test_fit_wall_ignores_a_pillar_in_front_of_the_wall(p):
    """Ties in the RANSAC go to the FARTHER line, so a pillar between the car
    and the wall is outliers, not the fit."""
    r = add_wall(blank(), 500.0, "left")
    add_point_cluster(r, 100.0, 300.0, n=5, spread_mm=15)
    perp, yaw, n, m, c = fit_wall(r, 90)
    assert perp == pytest.approx(500.0, rel=0.06), "the wall, not the pillar"


def test_cones_both_walls(p):
    r = add_wall(add_wall(blank(), 400.0, "left"), 600.0, "right")
    w = cones(r, p)
    assert w.left_mm == pytest.approx(400.0, rel=0.03)
    assert w.right_mm == pytest.approx(600.0, rel=0.03)
    assert w.yaw_deg == pytest.approx(0.0, abs=1.5)
    assert w.both()


def test_cones_disagreeing_walls_give_no_yaw(p):
    """At a corner, or where the inner wall ends, the two fits disagree. No
    yaw is better than a wrong one: a wrong yaw rotates every sighting."""
    r = add_wall(blank(), 400.0, "left", yaw_deg=0.0)
    add_wall(r, 600.0, "right", yaw_deg=25.0)
    w = cones(r, p)
    assert w.left_mm is not None and w.right_mm is not None
    assert w.yaw_deg is None


def test_cones_one_wall_uses_its_yaw(p):
    r = add_wall(blank(), 400.0, "left", yaw_deg=6.0)
    w = cones(r, p)
    assert w.right_mm is None
    assert w.yaw_deg == pytest.approx(6.0, abs=2.0)


def test_cones_empty_scan(p):
    w = cones(blank(), p)
    assert w.left_mm is None and w.right_mm is None and w.yaw_deg is None


# ----------------------------------------------------- pillar location

def test_locate_pillar_uses_the_lidar_range_on_the_camera_ray(p):
    r = add_point_cluster(blank(), 600.0, 100.0, n=5, spread_mm=15)
    bearing = math.degrees(math.atan2(100.0, 600.0))
    xy = locate_pillar(r, bearing, area=600, p=p)
    assert xy is not None
    assert xy[0] == pytest.approx(600.0, abs=60)
    assert xy[1] == pytest.approx(100.0, abs=60)


def test_locate_pillar_falls_back_to_area_when_nothing_agrees(p):
    """No LiDAR return on the ray at all: the size estimate is what is left."""
    area = 600
    xy = locate_pillar(blank(), 0.0, area=area, p=p)
    assert xy is not None
    expected = p["AREA_K"] / math.sqrt(area)
    assert xy[0] == pytest.approx(expected, rel=1e-6)


def test_locate_pillar_none_without_a_bearing(p):
    assert locate_pillar(blank(), None, area=600, p=p) is None


def test_locate_pillar_rejects_beyond_the_maximum(p):
    p.set("AREA_K", 60000)
    p.set("PILLAR_MAX_MM", 500)
    assert locate_pillar(blank(), 0.0, area=100, p=p) is None


def test_locate_pillar_adds_the_face_to_centre_offset(p):
    """The LiDAR sees the pillar's FACE; the planner wants its CENTRE."""
    r = blank()
    r[0] = 600.0
    xy = locate_pillar(r, 0.0, area=0, p=p)
    assert xy[0] == pytest.approx(600.0 + p["FACE_TO_CENTRE_MM"])


# --------------------------------------------------------- candidates

def test_candidates_find_a_pillar_in_the_corridor(p):
    r = add_wall(add_wall(blank(), 500.0, "left"), 500.0, "right")
    add_point_cluster(r, 700.0, 80.0, n=4, spread_mm=15)
    w = cones(r, p)
    out = lidar_candidates(r, w, p)
    assert out, "the pillar should be a candidate"
    x, y = out[0]
    assert x == pytest.approx(700.0, abs=80)
    assert y == pytest.approx(80.0, abs=80)


def test_candidates_reject_the_walls_themselves(p):
    """Clustering happens BEFORE filtering for exactly this reason: a wall is
    one long run and is thrown out whole by the width test. Filtering first
    would chop it into short fragments that look like pillars."""
    r = add_wall(add_wall(blank(), 500.0, "left"), 500.0, "right")
    w = cones(r, p)
    assert lidar_candidates(r, w, p) == []


def test_candidates_need_a_corridor(p):
    """No wall fitted either side means a corner, and there is no corridor to
    search in."""
    r = add_point_cluster(blank(), 700.0, 0.0, n=4)
    assert lidar_candidates(r, WallFit(), p) == []


def test_candidates_reject_something_too_wide(p):
    r = add_wall(add_wall(blank(), 500.0, "left"), 500.0, "right")
    add_point_cluster(r, 700.0, 0.0, n=12, spread_mm=40)   # 480 mm wide
    w = cones(r, p)
    assert lidar_candidates(r, w, p) == []


def test_candidates_are_sorted_nearest_first(p):
    r = add_wall(add_wall(blank(), 500.0, "left"), 500.0, "right")
    add_point_cluster(r, 1200.0, 150.0, n=3, spread_mm=12)
    add_point_cluster(r, 600.0, -150.0, n=3, spread_mm=12)
    w = cones(r, p)
    out = lidar_candidates(r, w, p)
    if len(out) >= 2:
        assert math.hypot(*out[0]) <= math.hypot(*out[1])


def test_unclassified_skips_the_named_pillar():
    cands = [(600.0, 100.0), (900.0, -50.0)]
    assert unclassified(cands, (610.0, 105.0)) == (900.0, -50.0)
    assert unclassified(cands, None) == (600.0, 100.0)
    assert unclassified([], (0.0, 0.0)) is None
