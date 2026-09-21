"""
The lane planner, against the behaviour ObstacleRound.cpp's updatePlanner()
had. Every number here is either the firmware's default or arithmetic on it,
so a test failing means the port drifted, not that the robot changed.
"""

import math

import pytest

from control.planner import LanePlanner, PlanInput, Track
from worldstate import wrap180

RED, GREEN = "RED", "GREEN"


def base(p, **kw):
    """A PlanInput with the car straight, centred and driving."""
    kw.setdefault("new_frame", True)
    kw.setdefault("enabled", True)
    kw.setdefault("lidar_stale", False)
    kw.setdefault("cone_left", 500.0)
    kw.setdefault("cone_right", 500.0)
    return PlanInput(p=p, **kw)


# ---------------------------------------------------------------- derived

def test_derived_match_the_firmware(p):
    # recomputeDerived(): PASS_CLEAR = PILLAR_HALF + CAR_HALF_W + PASS_MARGIN
    #                     LANE_LIMIT = CORRIDOR/2 - CAR_HALF_W - WALL_MARGIN
    assert p.derived["PASS_CLEAR_MM"] == pytest.approx(25 + 57 + 80)     # 162
    assert p.derived["LANE_LIMIT_MM"] == pytest.approx(500 - 57 - 45)    # 398
    assert p.derived["AVOID_CLEAR_MM"] == pytest.approx(25 + 57 + 15)    # 97


def test_derived_track_a_change(p):
    p.set("CORRIDOR_MM", 1200)
    assert p.derived["LANE_LIMIT_MM"] == pytest.approx(600 - 57 - 45)
    p.set("PASS_MARGIN_MM", 0)
    assert p.derived["PASS_CLEAR_MM"] == pytest.approx(82)


def test_lane_limit_never_negative(p):
    # CORRIDOR_MM's own lower bound is 300, which still leaves 48 mm, so the
    # clamp has to be provoked from the other side.
    p.set("CORRIDOR_MM", 300)
    p.set("CAR_HALF_W_MM", 200)
    p.set("WALL_MARGIN_MM", 300)
    assert p.derived["LANE_LIMIT_MM"] == 0


# ------------------------------------------------------------ lane offset

def test_lane_offset_centred(p):
    pl = LanePlanner()
    pl.update(base(p, cone_left=500.0, cone_right=500.0))
    assert pl.lane_off_ok
    assert pl.lane_off == pytest.approx(0.0)


def test_lane_offset_sign_is_left_positive(p):
    pl = LanePlanner()
    # closer to the LEFT wall -> the car is left of centre -> positive
    pl.update(base(p, cone_left=300.0, cone_right=700.0))
    assert pl.lane_off == pytest.approx(200.0)


def test_lane_offset_one_wall_only(p):
    pl = LanePlanner()
    pl.update(base(p, cone_left=400.0, cone_right=None))
    assert pl.lane_off == pytest.approx(500 - 400)
    pl2 = LanePlanner()
    pl2.update(base(p, cone_left=None, cone_right=400.0))
    assert pl2.lane_off == pytest.approx(400 - 500)


def test_lane_offset_no_walls_is_not_ok(p):
    pl = LanePlanner()
    pl.update(base(p, cone_left=None, cone_right=None))
    assert not pl.lane_off_ok


def test_lane_offset_rejects_a_jump_until_it_persists(p):
    """The car moves under 40 mm sideways per revolution, so a bigger step is
    a bad fit - unless it persists for three revolutions."""
    pl = LanePlanner()
    pl.update(base(p, cone_left=500.0, cone_right=500.0))
    assert pl.lane_off == pytest.approx(0.0)
    for _ in range(3):                       # jump of 300 mm, with new revs
        pl.update(base(p, cone_left=200.0, cone_right=800.0, new_rev=True))
        assert pl.lane_off == pytest.approx(0.0), "should hold the old value"
    pl.update(base(p, cone_left=200.0, cone_right=800.0, new_rev=True))
    assert pl.lane_off == pytest.approx(300.0), "persisted, so it is real"


def test_lane_offset_disagreeing_walls_keep_the_closer_side(p):
    """Walls that do not add up to the corridor mean one cone is fitted to
    something else - a pillar beside the car. Keep the side that agrees with
    what we believed a moment ago."""
    pl = LanePlanner()
    pl.update(base(p, cone_left=500.0, cone_right=500.0))
    # left reads 200 (a pillar), right still 500: sum 700, not ~1000
    pl.update(base(p, cone_left=200.0, cone_right=500.0))
    # fromL = 500-200 = 300 (far from 0), fromR = 500-500 = 0 (agrees)
    assert pl.lane_off == pytest.approx(0.0)


# --------------------------------------------------------------- reach

def test_lat_reach_zero_distance(p):
    assert LanePlanner.lat_reach(0, 0, 270, 75) == 0
    assert LanePlanner.lat_reach(-5, 0, 270, 75) == 0


def test_lat_reach_grows_with_distance(p):
    a = LanePlanner.lat_reach(100, 0, 270, 75)
    b = LanePlanner.lat_reach(300, 0, 270, 75)
    c = LanePlanner.lat_reach(900, 0, 270, 75)
    assert 0 < a < b < c


def test_lat_reach_helped_by_starting_yaw(p):
    """psi0 is + when the car is already angled TOWARD the target."""
    straight = LanePlanner.lat_reach(400, 0.0, 270, 75)
    angled = LanePlanner.lat_reach(400, math.radians(30), 270, 75)
    assert angled > straight


def test_lat_reach_straight_line_beyond_the_arc(p):
    """Past the arc the car runs straight at yaw_max, so reach grows
    linearly with tan(phi)."""
    R, phi = 270.0, math.radians(75)
    a_arc = R * math.sin(phi)
    near = LanePlanner.lat_reach(a_arc + 100, 0, 270, 75)
    far = LanePlanner.lat_reach(a_arc + 200, 0, 270, 75)
    assert far - near == pytest.approx(100 * math.tan(phi), rel=1e-6)


# -------------------------------------------------------------- sighting

def test_sighting_straight_ahead_lands_where_expected(p):
    pl = LanePlanner()
    pl.update(base(p))                         # establish the lane offset
    pl.update(base(p, pillar_colour=RED, pillar_xy=(600.0, 100.0)))
    assert len(pl.tracks) == 1
    t = pl.tracks[0]
    assert t.colour == RED
    assert t.along == pytest.approx(600.0)
    assert t.lat == pytest.approx(100.0)       # car centred, so lat = y


def test_sighting_is_rotated_by_the_car_yaw(p):
    """A pillar dead ahead of a yawed car is off to the side in the LANE
    frame, by the yaw. 30 deg, not 90: at 90 the pillar would land 500 mm off
    the lane centre, past PILLAR_MAX_LAT_MM, where the firmware stops treating
    it as a pillar of this straight at all."""
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, heading=30.0, lane_heading=0.0,
                   pillar_colour=GREEN, pillar_xy=(500.0, 0.0)))
    t = pl.tracks[0]
    assert t.along == pytest.approx(500.0 * math.cos(math.radians(30)))
    assert t.lat == pytest.approx(500.0 * math.sin(math.radians(30)))


def test_sighting_rotation_is_signed_the_right_way(p):
    """Yawed LEFT, a pillar ahead appears to the LEFT of the lane, so its
    lat is positive. Getting this sign wrong swaps every passing side."""
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, heading=20.0, pillar_colour=RED, pillar_xy=(600.0, 0.0)))
    assert pl.tracks[0].lat > 0
    pl2 = LanePlanner()
    pl2.update(base(p))
    pl2.update(base(p, heading=-20.0, pillar_colour=RED, pillar_xy=(600.0, 0.0)))
    assert pl2.tracks[0].lat < 0


def test_sighting_too_far_ahead_is_ignored(p):
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, pillar_colour=RED,
                   pillar_xy=(p["PLAN_MAX_AHEAD_MM"] + 100, 0.0)))
    assert pl.tracks == []


def test_sighting_well_behind_is_ignored(p):
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, pillar_colour=RED, pillar_xy=(-200.0, 0.0)))
    assert pl.tracks == []


def test_repeated_sightings_refine_and_confirm(p):
    pl = LanePlanner()
    pl.update(base(p))
    for _ in range(3):
        pl.update(base(p, pillar_colour=RED, pillar_xy=(600.0, 100.0)))
    assert pl.tracks[0].hits == 3
    assert pl.tracks[0].hits >= p["TRACK_CONFIRM"]


def test_track_table_never_exceeds_its_size(p):
    from control.planner import MAX_TRACKS
    pl = LanePlanner()
    pl.update(base(p))
    for i in range(MAX_TRACKS + 3):
        pl.update(base(p, pillar_colour=RED,
                       pillar_xy=(200.0 + 300 * i, 0.0)))
    assert len(pl.tracks) <= MAX_TRACKS


def test_unconfirmed_track_is_forgotten_after_TRACK_FORGET(p):
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, pillar_colour=RED, pillar_xy=(600.0, 100.0)))
    assert len(pl.tracks) == 1 and pl.tracks[0].hits == 1
    pl.lane_along = p["TRACK_FORGET_MM"] + 50     # drive on without seeing it
    pl.update(base(p))
    assert pl.tracks == []


# ------------------------------------------------------- cross-corner bit

def test_cross_corner_sighting_records_only_the_colour(p):
    """A pillar of another straight, seen across the corner. Its coordinates
    are meaningless here, so it must not become a track - but its colour says
    which side to leave the corner on."""
    pl = LanePlanner()
    pl.update(base(p))
    # clockwise: the turn is to the right, so a lat < 0 sighting is toward it
    pl.update(base(p, pillar_colour=GREEN, pillar_xy=(800.0, -600.0),
                   clockwise=True, turn_clockwise=True))
    assert pl.tracks == [], "must not steer the car in this lane frame"
    assert pl.next_straight_colour == GREEN


def test_cross_corner_ignores_the_side_away_from_the_turn(p):
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, pillar_colour=GREEN, pillar_xy=(800.0, 600.0),
                   clockwise=True, turn_clockwise=True, direction_known=True))
    assert pl.next_straight_colour is None


def test_cross_corner_side_test_waits_for_the_direction(p):
    """Before a floor colour has been read, turn_clockwise is a guess
    (anticlockwise). Rejecting on it would discard every clockwise sighting
    for the whole first straight, so the side test is suspended until the
    direction is actually known."""
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, pillar_colour=GREEN, pillar_xy=(800.0, -600.0),
                   turn_clockwise=False, direction_known=False))
    assert pl.next_straight_colour == GREEN


def test_cross_corner_respects_CARRY_TRACKS(p):
    p.set("CARRY_TRACKS", False)
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, pillar_colour=GREEN, pillar_xy=(800.0, -600.0),
                   turn_clockwise=True))
    assert pl.next_straight_colour is None


def test_cross_corner_beyond_CROSS_MAX_LAT_is_not_the_next_straight(p):
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, pillar_colour=RED,
                   pillar_xy=(800.0, -(p["CROSS_MAX_LAT_MM"] + 100)),
                   turn_clockwise=True))
    assert pl.next_straight_colour is None


# ------------------------------------------------------------- the passes

def confirmed(pl, colour, lat, along, flipped=False):
    t = Track(colour=colour, lat=lat, along=along, hits=9, flipped=flipped)
    pl.tracks.append(t)
    return t


def test_red_is_passed_on_its_right(p):
    """Red -> the car must end up to the pillar's RIGHT, i.e. at a LOWER lane
    offset: car lat <= pillar lat - PASS_CLEAR."""
    pl = LanePlanner()
    pl.update(base(p))
    confirmed(pl, RED, lat=0.0, along=600.0)
    pl.update(base(p))
    assert pl.pass_active
    assert pl.lat_target <= 0.0 - p.derived["PASS_CLEAR_MM"] + 1e-6


def test_green_is_passed_on_its_left(p):
    pl = LanePlanner()
    pl.update(base(p))
    confirmed(pl, GREEN, lat=0.0, along=600.0)
    pl.update(base(p))
    assert pl.lat_target >= 0.0 + p.derived["PASS_CLEAR_MM"] - 1e-6


def test_target_stays_inside_the_lane_limit(p):
    pl = LanePlanner()
    pl.update(base(p))
    confirmed(pl, GREEN, lat=380.0, along=600.0)   # hard against the wall
    pl.update(base(p))
    assert abs(pl.lat_target) <= p.derived["LANE_LIMIT_MM"] + 1e-6


def test_no_pillar_means_aim_at_the_centre(p):
    pl = LanePlanner()
    pl.update(base(p, cone_left=300.0, cone_right=700.0))
    pl.update(base(p, cone_left=300.0, cone_right=700.0))
    assert not pl.pass_active
    assert pl.lat_target == pytest.approx(0.0)
    # off-centre to the left, so the yaw command must point right (negative)
    assert pl.lat_yaw_cmd < 0


def test_yaw_command_is_capped_by_the_mode(p):
    """Plain centring is deliberately gentle; a pass gets far more authority."""
    pl = LanePlanner()
    pl.lane_along = p["POST_CORNER_BOOST_MM"] + 100      # past the boost zone
    pl.update(base(p, cone_left=100.0, cone_right=900.0))
    pl.update(base(p, cone_left=100.0, cone_right=900.0))
    assert abs(pl.lat_yaw_cmd) <= p["CENTRE_YAW_MAX"] + 1e-6

    pl2 = LanePlanner()
    pl2.lane_along = p["POST_CORNER_BOOST_MM"] + 100
    pl2.update(base(p, cone_left=100.0, cone_right=900.0))
    confirmed(pl2, GREEN, lat=380.0, along=300.0)
    pl2.update(base(p, cone_left=100.0, cone_right=900.0))
    assert abs(pl2.lat_yaw_cmd) <= p["PASS_YAW_MAX"] + 1e-6


def test_post_corner_boost_gives_more_authority_than_centring(p):
    pl = LanePlanner()
    pl.update(base(p, cone_left=100.0, cone_right=900.0, corner_count=1))
    pl.update(base(p, cone_left=100.0, cone_right=900.0, corner_count=1))
    assert pl.lane_along < p["POST_CORNER_BOOST_MM"]
    assert abs(pl.lat_yaw_cmd) <= p["POST_CORNER_YAW_MAX"] + 1e-6


def test_a_passed_pillar_is_retired_and_records_an_inner_pass(p):
    """Clockwise, red is passed on its right, which is the inner side."""
    pl = LanePlanner()
    pl.update(base(p))
    confirmed(pl, RED, lat=0.0, along=-p["PASS_HOLD_MM"] - 50)
    pl.update(base(p, clockwise=True))
    assert pl.tracks == []
    assert pl.have_inner_pass


def test_green_passed_clockwise_is_not_an_inner_pass(p):
    pl = LanePlanner()
    pl.update(base(p))
    confirmed(pl, GREEN, lat=0.0, along=-p["PASS_HOLD_MM"] - 50)
    pl.update(base(p, clockwise=True))
    assert not pl.have_inner_pass


def test_conflicting_pillars_let_the_most_urgent_win(p):
    """A red and a green at the same distance demand opposite sides. lo > hi,
    so the nearest one's bound is taken rather than something between them
    that satisfies neither."""
    pl = LanePlanner()
    pl.update(base(p))
    confirmed(pl, RED, lat=0.0, along=400.0)
    confirmed(pl, GREEN, lat=0.0, along=401.0)
    pl.update(base(p))
    clear = p.derived["PASS_CLEAR_MM"]
    assert pl.lat_target in (pytest.approx(-clear), pytest.approx(clear))


# ------------------------------------------------------------- giving up

def test_give_up_flips_to_the_reachable_side(p):
    """The correct side is unreachable in the distance left and the other
    side is not, so the track is flipped rather than driven into.

    The window for this is narrow and the numbers are load-bearing: car at
    -380 (hard against the right wall), a green at lat +100 so its correct
    side is +262 (need 642 mm of lateral travel) and its wrong side is -62
    (need 318). At 361 mm ahead the reach is ~350 mm: too little for 642,
    enough for 318.
    """
    pl = LanePlanner()
    pl.update(base(p, cone_left=880.0, cone_right=120.0))   # hard right
    assert pl.lane_off == pytest.approx(-380.0)
    confirmed(pl, GREEN, lat=100.0, along=pl.lane_along + 361.0)
    pl.update(base(p, cone_left=880.0, cone_right=120.0))
    assert pl.tracks[0].flipped
    assert pl.lat_target == pytest.approx(100.0 - p.derived["PASS_CLEAR_MM"])


def test_no_give_up_when_the_correct_side_is_still_reachable(p):
    pl = LanePlanner()
    pl.update(base(p, cone_left=880.0, cone_right=120.0))
    confirmed(pl, GREEN, lat=100.0, along=pl.lane_along + 900.0)
    pl.update(base(p, cone_left=880.0, cone_right=120.0))
    assert not pl.tracks[0].flipped
    assert pl.lat_target == pytest.approx(100.0 + p.derived["PASS_CLEAR_MM"])


def test_avoid_only_when_neither_side_is_reachable(p):
    """Holding an unreachable target drives the car into the pillar. A
    wrong-side pass costs points; a collision costs the run. So fall back to
    the bare geometric miss - AVOID_CLEAR, no PASS_MARGIN - and take whichever
    side is closer."""
    pl = LanePlanner()
    pl.update(base(p, cone_left=880.0, cone_right=120.0))
    confirmed(pl, GREEN, lat=300.0, along=pl.lane_along + 220.0)
    pl.update(base(p, cone_left=880.0, cone_right=120.0))
    assert not pl.tracks[0].flipped, "neither side was reachable"
    assert pl.lat_target == pytest.approx(300.0 - p.derived["AVOID_CLEAR_MM"])
    assert "AVOID only" in pl.note


def test_give_up_is_off_when_ALLOW_GIVE_UP_is_false(p):
    p.set("ALLOW_GIVE_UP", False)
    pl = LanePlanner()
    pl.update(base(p, cone_left=880.0, cone_right=120.0))
    confirmed(pl, GREEN, lat=100.0, along=pl.lane_along + 361.0)
    pl.update(base(p, cone_left=880.0, cone_right=120.0))
    assert not pl.tracks[0].flipped
    assert pl.lat_target == pytest.approx(100.0 + p.derived["PASS_CLEAR_MM"])


def test_flipped_track_reverses_the_passing_side(p):
    t = Track(colour=RED, lat=0.0, along=0.0, hits=9)
    assert t.pass_right() is True
    t.flipped = True
    assert t.pass_right() is False


# ------------------------------------------------------- unknown objects

def test_far_unknown_object_is_lined_up_with(p):
    """Beyond UNK_COMMIT the car lines up with it, so it comes into view and
    both passing sides stay open."""
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, unknown_xy=(p["UNK_COMMIT_MM"] + 300, 150.0)))
    assert pl.pass_active
    assert pl.lat_target == pytest.approx(150.0)


def test_near_unknown_object_is_dodged(p):
    """Inside UNK_COMMIT it is still unnamed, so dodge rather than drive into
    it - to the side with more room."""
    pl = LanePlanner()
    pl.update(base(p))
    pl.update(base(p, unknown_xy=(p["UNK_COMMIT_MM"] - 100, -150.0)))
    assert pl.pass_active
    # it is to the right, so we go further right of it... no: lat < 0 means
    # the firmware adds PASS_CLEAR, moving the car to its LEFT
    assert pl.lat_target == pytest.approx(-150.0 + p.derived["PASS_CLEAR_MM"])


def test_unknown_at_a_named_pillar_is_not_treated_twice(p):
    pl = LanePlanner()
    pl.update(base(p))
    confirmed(pl, RED, lat=100.0, along=700.0)
    pl.update(base(p, unknown_xy=(700.0, 100.0)))
    # the target is the red's bound, not the unknown's line-up
    assert pl.lat_target == pytest.approx(100.0 - p.derived["PASS_CLEAR_MM"])


# ------------------------------------------------------------- levelling

def test_level_pulls_the_lane_heading_toward_the_walls(p):
    pl = LanePlanner()
    i = base(p, new_rev=True, heading=2.0, lane_heading=0.0, wall_ang=0.0)
    step = pl.level_step(i)
    # est = heading - wall_ang = 2, diff = 2, step = 0.05 * 2 = 0.1
    assert step == pytest.approx(0.1)


def test_level_is_capped(p):
    pl = LanePlanner()
    i = base(p, new_rev=True, heading=7.0, lane_heading=0.0, wall_ang=0.0)
    assert pl.level_step(i) == pytest.approx(p["LEVEL_MAX_STEP"])


def test_level_refuses_a_big_disagreement(p):
    """A large difference is a bad fit, not drift."""
    pl = LanePlanner()
    i = base(p, new_rev=True, heading=p["LEVEL_MAX_DIFF"] + 5,
             lane_heading=0.0, wall_ang=0.0)
    assert pl.level_step(i) == 0.0


def test_level_refuses_while_swerving(p):
    pl = LanePlanner()
    pl.pass_active = True
    i = base(p, new_rev=True, heading=2.0, lane_heading=0.0, wall_ang=0.0)
    assert pl.level_step(i) == 0.0


def test_level_refuses_a_yawed_car(p):
    pl = LanePlanner()
    i = base(p, new_rev=True, heading=2.0, lane_heading=0.0,
             wall_ang=p["LEVEL_MAX_WALLANG"] + 1)
    assert pl.level_step(i) == 0.0


def test_level_needs_both_walls(p):
    pl = LanePlanner()
    i = base(p, new_rev=True, heading=2.0, wall_ang=0.0, cone_right=None)
    assert pl.level_step(i) == 0.0


def test_level_only_runs_on_a_new_revolution(p):
    pl = LanePlanner()
    i = base(p, new_rev=False, heading=2.0, wall_ang=0.0)
    assert pl.level_step(i) == 0.0


# ----------------------------------------------------------- 3-point cue

def test_inner_pillar_cue_can_be_switched_off(p):
    p.set("USE_CORNER_MANEUVER", False)
    pl = LanePlanner()
    pl.have_inner_pass = True
    pl.last_inner_pass_at = 0.0
    assert not pl.inner_pillar_near_corner(p, True)


def test_inner_pillar_cue_is_on_by_default(p):
    """It is gated now - see the next test - which is what made it safe to
    turn on. Fired on every inner pass it cost the outer-side cases 60-300 mm
    of clearance for nothing."""
    assert p["USE_CORNER_MANEUVER"]
    pl = LanePlanner()
    pl.have_inner_pass = True
    pl.last_inner_pass_at = 0.0
    pl.lane_along = 100.0
    assert pl.inner_pillar_near_corner(p, True)


def test_inner_pillar_cue_needs_the_corner_to_exit_inner(p):
    """The 3-point is for the case where the next straight ALSO needs the
    inner side. If the corner is already planned to exit outer, a plain arc
    is doing the right thing and stopping to shuffle only costs clearance."""
    pl = LanePlanner()
    pl.have_inner_pass = True
    pl.last_inner_pass_at = 0.0
    pl.lane_along = 100.0

    pl.corner_exit_cmd = -200.0            # CW: negative = inner
    assert pl.inner_pillar_near_corner(p, True)

    pl.corner_exit_cmd = +200.0            # CW: positive = outer
    assert not pl.inner_pillar_near_corner(p, True)


def test_inner_pillar_cue_is_clockwise_only(p):
    p.set("USE_CORNER_MANEUVER", True)
    pl = LanePlanner()
    pl.have_inner_pass = True
    pl.last_inner_pass_at = 0.0
    pl.lane_along = 100.0
    assert pl.inner_pillar_near_corner(p, True)
    assert not pl.inner_pillar_near_corner(p, False)


def test_inner_pillar_cue_expires_outside_the_zone(p):
    p.set("USE_CORNER_MANEUVER", True)
    pl = LanePlanner()
    pl.have_inner_pass = True
    pl.last_inner_pass_at = 0.0
    pl.lane_along = p["MNV_ZONE_MM"] + 50
    assert not pl.inner_pillar_near_corner(p, True)


# ------------------------------------------------------------- odometry

def test_lane_distance_projects_onto_the_lane(p):
    """Driving at 60 deg to the lane advances `along` by cos(60) of the
    distance travelled."""
    pl = LanePlanner()
    tpm = 1.4853
    pl.update(base(p, odo_ticks=0, ticks_per_mm=tpm))
    pl.update(base(p, odo_ticks=int(1000 * tpm), ticks_per_mm=tpm,
                   heading=60.0, lane_heading=0.0))
    assert pl.lane_along == pytest.approx(1000 * math.cos(math.radians(60)),
                                          rel=1e-3)


def test_reset_lane_along_clears_the_cross_corner_colour(p):
    pl = LanePlanner()
    pl.next_straight_colour = RED
    pl.have_inner_pass = True
    pl.reset_lane_along(1234)
    assert pl.lane_along == 0.0
    assert pl.next_straight_colour is None
    assert not pl.have_inner_pass


def test_planner_disabled_produces_no_yaw(p):
    pl = LanePlanner()
    pl.update(base(p, cone_left=100.0, cone_right=900.0, enabled=False))
    assert pl.lat_yaw_cmd == 0.0
    assert not pl.pass_active


# ------------------------------------------------------ the second pillar
#
# The corner has to decide which side of the lane to come out on, and the
# evidence for that is the FIRST pillar of the NEXT straight - which, while
# the car is still on this one, is usually the second-largest blob in frame.
# "Further away" is not on its own a reason to believe it: a straight with two
# pillars on it also has a nearer and a further one, and shaping the corner
# from the second pillar of the CURRENT straight is worse than not shaping it.
# So every one of these gates matters.

def sec(p, **kw):
    """A PlanInput carrying a second pillar, with the lane fix established."""
    kw.setdefault("sec_colour", "GREEN")
    kw.setdefault("direction_known", True)
    kw.setdefault("turn_clockwise", True)
    return base(p, **kw)


def primed(p, **kw):
    """A planner with a valid lane offset, ready to judge a sighting."""
    pl = LanePlanner()
    pl.update(base(p, **kw))
    assert pl.lane_off_ok
    return pl


def test_a_second_pillar_outside_the_corridor_is_the_next_straight(p):
    pl = primed(p)
    # 900 mm ahead and 600 mm to the right: outside PILLAR_MAX_LAT_MM (420),
    # inside CROSS_MAX_LAT_MM, and to the right is where a CW corner turns.
    pl.update(sec(p, sec_xy=(900.0, -600.0)))
    assert pl.sec_seen_colour == "GREEN"
    assert pl.planned_next_colour(p) == "GREEN"
    assert pl.secondary_is_live(p)


def test_a_second_pillar_inside_the_corridor_is_just_another_pillar(p):
    """Inside PILLAR_MAX_LAT_MM it belongs to THIS straight, and the ordinary
    planner deals with it."""
    pl = primed(p)
    pl.update(sec(p, sec_xy=(900.0, -200.0)))
    assert pl.sec_seen_colour is None
    assert pl.planned_next_colour(p) is None


def test_a_second_pillar_on_the_wrong_side_is_the_straight_behind(p):
    """A CW corner turns right, so a pillar out to the LEFT is not in the
    straight the car is about to enter."""
    pl = primed(p)
    pl.update(sec(p, sec_xy=(900.0, +600.0), turn_clockwise=True))
    assert pl.sec_seen_colour is None


def test_the_side_test_is_suspended_until_the_direction_is_known(p):
    """Before any floor colour has been read, turn_clockwise is a guess.
    Rejecting on it then would throw away everything on one side for the
    whole first straight - which is the bug directionKnown() exists for."""
    pl = primed(p)
    pl.update(sec(p, sec_xy=(900.0, +600.0), direction_known=False))
    assert pl.sec_seen_colour == "GREEN"


def test_a_second_pillar_nearer_than_the_first_is_ignored(p):
    """If it is closer than the pillar being passed it is not the next
    straight's, whatever the camera's area ranking said."""
    pl = primed(p)
    pl.update(sec(p, sec_xy=(500.0, -600.0), pillar_xy=(900.0, 0.0),
                  pillar_colour="RED"))
    assert pl.sec_seen_colour is None


def test_a_second_pillar_behind_the_car_is_ignored(p):
    pl = primed(p)
    pl.update(sec(p, sec_xy=(-400.0, -600.0)))
    assert pl.sec_seen_colour is None


def test_the_second_pillar_can_be_switched_off(p):
    p.set("USE_SECONDARY_CORNER", False)
    pl = primed(p)
    pl.update(sec(p, sec_xy=(900.0, -600.0)))
    assert pl.sec_seen_colour is None


def test_the_second_pillar_goes_stale_with_distance(p):
    pl = primed(p)
    pl.update(sec(p, sec_xy=(900.0, -600.0)))
    assert pl.planned_next_colour(p) == "GREEN"
    pl.lane_along = pl.sec_seen_at + p["SECONDARY_ZONE_MM"] + 1.0
    assert pl.planned_next_colour(p) is None
    assert not pl.secondary_is_live(p)


def test_the_second_pillar_beats_the_cross_corner_sighting(p):
    """Both are evidence about the next straight; the second pillar is the
    direct kind, so it wins."""
    pl = primed(p)
    pl.next_straight_colour = "RED"
    assert pl.planned_next_colour(p) == "RED"
    pl.update(sec(p, sec_xy=(900.0, -600.0), sec_colour="GREEN"))
    assert pl.planned_next_colour(p) == "GREEN"


def test_the_cross_corner_sighting_is_the_fallback(p):
    """It is what a frame with only ONE pillar in it can still report."""
    pl = primed(p)
    pl.next_straight_colour = "RED"
    pl.update(base(p))                     # no second pillar this frame
    assert pl.planned_next_colour(p) == "RED"
    assert not pl.secondary_is_live(p)


def test_a_corner_clears_the_second_pillar(p):
    """Its colour describes the straight the car is now ON, so carrying it
    into the next corner would shape that one from stale evidence."""
    pl = primed(p)
    pl.update(sec(p, sec_xy=(900.0, -600.0)))
    assert pl.sec_seen_colour == "GREEN"
    pl.reset_lane_along()
    assert pl.sec_seen_colour is None
    assert pl.planned_next_colour(p) is None
