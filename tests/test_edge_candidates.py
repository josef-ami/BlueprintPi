"""
The edge-walk pillar detector in obstacleRound.py, and the camera fusion.

Every scan here is ray-cast from a described arena - a box corridor with
pillars in it - so what the detector should answer is known by construction.
That matters more for this detector than for the clustering it replaced: the
walk reads the BACKGROUND (the far wall) as much as the object, so a scan of
a pillar floating in empty space would test nothing that happens on the mat.
"""

import math

import pytest

import obstacleRound as obs

INF = float("inf")

LEFT_Y = 500.0            # corridor walls, car frame, y = left
RIGHT_Y = -500.0
AHEAD_X = 2500.0
BEHIND_X = -800.0

LLINE = (0.0, LEFT_Y)     # y = m*x + c, as cones_full() returns them
RLINE = (0.0, RIGHT_Y)

PILLAR_R = 25.0           # 50 mm square, modelled as its inscribed circle


# ------------------------------------------------------------ scan builder

def _box_hit(cx, sy):
    """Distance from the origin to the corridor box along (cx, sy)."""
    ts = []
    for num, den in ((AHEAD_X, cx), (BEHIND_X, cx), (LEFT_Y, sy), (RIGHT_Y, sy)):
        if abs(den) < 1e-12:
            continue
        t = num / den
        if t <= 0:
            continue
        x, y = cx * t, sy * t
        if BEHIND_X - 1e-6 <= x <= AHEAD_X + 1e-6 and \
                RIGHT_Y - 1e-6 <= y <= LEFT_Y + 1e-6:
            ts.append(t)
    return min(ts) if ts else INF


def _circle_hit(cx, sy, px, py, r=PILLAR_R):
    """Nearest intersection of the ray with a circle, or INF."""
    b = cx * px + sy * py                       # ray dir is a unit vector
    c = px * px + py * py - r * r
    disc = b * b - c
    if disc < 0:
        return INF
    t = b - math.sqrt(disc)
    return t if t > 0 else INF


def arena(pillars=()):
    """A full 360-bin scan of the corridor with `pillars` [(x, y)] standing in it."""
    out = []
    for deg in range(360):
        a = math.radians(deg)
        cx, sy = math.cos(a), math.sin(a)
        r = _box_hit(cx, sy)
        for px, py in pillars:
            r = min(r, _circle_hit(cx, sy, px, py))
        out.append(r)
    return out


def bearing_to(x, y):
    """Bearing of (x, y) as the CAMERA sees it - what vision reports."""
    return math.degrees(math.atan2(y, x - obs.CAMERA_FWD_MM))


def near(got, want, tol):
    return abs(got - want) <= tol


# ------------------------------------------------------------- the builder

def test_the_empty_corridor_is_a_plausible_scan():
    r = arena()
    assert near(r[0], AHEAD_X, 1.0)             # straight ahead: the wall ahead
    assert near(r[90], LEFT_Y, 1.0)
    assert near(r[270], -RIGHT_Y, 1.0)
    # past 11.3 deg the ray stops reaching the wall ahead and hits the one
    # beside instead - that corner is the steep gradient the walk has to not
    # mistake for an object
    assert near(r[45], LEFT_Y * math.sqrt(2.0), 1.0)


def test_a_pillar_shortens_the_bins_it_covers():
    r = arena([(800.0, 0.0)])
    assert near(r[0], 800.0 - PILLAR_R, 1.0)
    assert near(r[45], LEFT_Y * math.sqrt(2.0), 1.0)   # nothing there, still wall


# -------------------------------------------------------------- the walk

def test_bare_corridor_yields_no_candidates():
    """The whole point: a wall never opens and closes within a few bins."""
    assert obs.lidar_candidates(arena(), LLINE, RLINE) == []


def test_finds_a_pillar_in_the_middle_of_the_corridor():
    out = obs.lidar_candidates(arena([(800.0, 0.0)]), LLINE, RLINE)
    assert len(out) == 1
    x, y = out[0]
    assert near(x, 800.0, 60.0)                 # centre, not the face
    assert near(y, 0.0, 40.0)


def test_reports_the_centre_not_the_face():
    """The face is at 775 mm; FACE_TO_CENTRE_MM has to be added back on."""
    out = obs.lidar_candidates(arena([(800.0, 0.0)]), LLINE, RLINE)
    assert out[0][0] > 800.0 - PILLAR_R + 10.0


def test_two_pillars_come_back_nearest_first():
    out = obs.lidar_candidates(
        arena([(1400.0, 250.0), (700.0, -200.0)]), LLINE, RLINE)
    assert len(out) == 2
    assert math.hypot(*out[0]) < math.hypot(*out[1])
    assert near(out[0][0], 700.0, 70.0)
    assert near(out[0][1], -200.0, 60.0)


def test_finds_a_pillar_hugging_the_wall():
    """The case the clustering lost: within CAND_GAP_MM of the wall behind it,
    the pillar merged into the wall run and was discarded with it. The edge in
    RANGE is still there, so the walk keeps it."""
    out = obs.lidar_candidates(arena([(900.0, -380.0)]), LLINE, RLINE)
    assert len(out) == 1
    assert near(out[0][1], -380.0, 60.0)


def test_a_pillar_too_close_to_the_wall_is_rejected():
    """Inside CAND_WALL_MM of the fitted line: that is wall, not a pillar."""
    assert obs.lidar_candidates(arena([(900.0, -480.0)]), LLINE, RLINE) == []


def test_beyond_cand_max_mm_is_not_a_candidate():
    far = obs.CAND_MAX_MM + 300.0
    assert obs.lidar_candidates(arena([(far, 0.0)]), LLINE, RLINE) == []


def test_no_wall_fit_means_no_corridor_to_search():
    assert obs.lidar_candidates(arena([(800.0, 0.0)]), None, None) == []


def test_one_wall_is_enough():
    out = obs.lidar_candidates(arena([(800.0, 0.0)]), LLINE, None)
    assert len(out) == 1


def test_dropouts_shorter_than_the_fill_gap_do_not_lose_the_pillar():
    r = arena([(800.0, 0.0)])
    for b in (40, 41, 120, 121, 122):           # glass-reflection style holes
        r[b] = INF
    out = obs.lidar_candidates(r, LLINE, RLINE)
    assert len(out) == 1
    assert near(out[0][0], 800.0, 60.0)


# ------------------------------------------------------------- the fusion

def test_match_pillar_picks_the_candidate_on_the_camera_ray():
    cands = obs.lidar_candidates(
        arena([(1400.0, 300.0), (700.0, -250.0)]), LLINE, RLINE)
    assert len(cands) == 2
    hit = obs.match_pillar(cands, bearing_to(700.0, -250.0))
    assert hit is not None
    assert near(hit[1], -250.0, 70.0)


def test_match_pillar_gives_up_outside_the_tolerance():
    cands = obs.lidar_candidates(arena([(800.0, 0.0)]), LLINE, RLINE)
    assert cands
    assert obs.match_pillar(cands, 70.0) is None


def test_match_pillar_needs_a_bearing_and_a_candidate():
    assert obs.match_pillar([], 0.0) is None
    assert obs.match_pillar([(800.0, 0.0)], None) is None


def test_fuse_falls_back_to_the_ray_hunt_when_the_walk_missed_it():
    """No candidates at all, but the camera sees something: locate_pillar
    still has to answer, or the STM32 loses the pillar entirely."""
    r = arena([(800.0, 0.0)])
    assert obs._fuse([], r, obs.COL_RED, 0.0, 600) is not None


def test_fuse_is_silent_without_a_colour():
    r = arena([(800.0, 0.0)])
    assert obs._fuse([], r, obs.COL_NONE, 0.0, 600) is None
