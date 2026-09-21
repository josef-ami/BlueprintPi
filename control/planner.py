"""
planner.py - lane position and the pillar pass planner.

Ported from ObstacleRound.cpp's updatePlanner() / updateLevel(), which this
replaces. It ran there at the Pi's frame rate already (`if (!lidarNewFrame)
return;`), so running it here costs no control bandwidth - it just runs where
the camera and the LiDAR already are, in a language where the give-up
arithmetic is legible.

THE MODEL

Everything is a LATERAL POSITION in the lane (mm, + = left of the lane
centre):

    car      lane_off    from the two cone wall fits (perpendicular mm)
    pillar   track.lat   car lane position + the pillar's position in the car
                         frame, rotated by the car's yaw into the lane

    Red must be passed on its right  ->  car lat <= pillar lat - PASS_CLEAR
    Green must be passed on its left ->  car lat >= pillar lat + PASS_CLEAR

The car aims for the lane position closest to the centre that satisfies every
pillar it is approaching or still alongside, and steers there with a yaw
command that the STM32's heading PID tracks. The yaw points at a PASS POINT -
the target lane position PASS_LEAD_MM before the pillar - so it sharpens as
the pillar nears and arrives in time. A plain proportional law eases off near
the target and arrived too late.

With no pillar in play the target is the lane centre. That IS the wall
centring; there is no separate centring law.

WHY NOT A PIXEL LAW
The original held the pillar at a fixed image offset (+/-150 px). At close
range that makes the car ORBIT the pillar and hit it, and with the 160 deg
lens a pillar is only big enough to react to in the last ~50 cm. Lane
positions do not have either problem.

Nothing here touches hardware, reads config.json or imports a sensor, so the
whole thing is testable on a bench with a list of numbers.
"""

import math
from dataclasses import dataclass, field

from worldstate import clamp, wrap180

DEG = math.pi / 180.0

MAX_TRACKS = 4
OFF_JUMP_REVS = 3        # a lane-offset jump must persist this long to be real

RED, GREEN = "RED", "GREEN"


@dataclass
class Track:
    """One pillar the car is dealing with, in lane coordinates."""
    colour: str
    lat: float               # + = left of the lane centre
    along: float             # lane distance from the last corner
    hits: int = 1
    last_seen: float = 0.0   # lane_along when it was last sighted
    flipped: bool = False    # the correct side was unreachable; going the
                             # other way on purpose

    def pass_right(self):
        """True when the car should go by on this pillar's RIGHT."""
        return (self.colour == RED) != self.flipped


@dataclass
class PlanInput:
    """One tick of everything the planner can see."""
    p: object                      # PiParams
    odo_ticks: int = 0
    ticks_per_mm: float = 1.4853
    heading: float = 0.0           # IMU, absolute
    lane_heading: float = 0.0      # the straight's direction
    new_frame: bool = False        # a DRIVE-rate perception frame arrived
    new_rev: bool = False          # the LiDAR completed a revolution
    lidar_stale: bool = True
    front_mm: float = float("inf")
    cone_left: float = None        # perpendicular mm, None = no fit
    cone_right: float = None
    wall_ang: float = None         # car yaw to the walls, + = pointing left
    pillar_xy: tuple = None        # (x, y) car frame of the named pillar
    pillar_colour: str = None      # its colour, or None if none is seen
    sec_xy: tuple = None           # (x, y) car frame of the SECOND pillar
    sec_colour: str = None         # its colour, or None
    unknown_xy: tuple = None       # nearest unnamed object, car frame
    direction_known: bool = False  # has any floor colour been read yet? Before
                                   # that, turn_clockwise is only a guess, and
                                   # a test that uses it to REJECT evidence
                                   # would throw away one whole side
    clockwise: bool = True         # the LOCKED direction (defaults to CW)
    turn_clockwise: bool = True    # which way the NEXT corner goes - before the
                                   # direction locks this is the floor colour's
                                   # guess, and it differs from `clockwise`.
                                   # Only the cross-corner and second-pillar
                                   # side tests use it, which is what the
                                   # firmware does too
    corner_count: int = 0
    enabled: bool = False          # DRIVE / FINAL only - a sighting taken
                                   # mid-turn is in the wrong lane frame


class LanePlanner:
    """The lane planner's state: where the car is in the lane, which pillars
    it knows about, and what yaw it wants."""

    def __init__(self):
        self.tracks = []
        self.lane_off = 0.0
        self.lane_off_ok = False
        self.lane_along = 0.0
        self._odo_base = None
        self._off_jumps = 0

        self.lat_yaw_cmd = 0.0     # + = yaw left of the lane heading
        self.lat_target = 0.0
        self.aim_mm = 0.0
        self.pass_active = False
        self.note = ""

        # cross-corner sighting: the colour waiting after the next corner
        self.next_straight_colour = None
        self.corner_exit_cmd = 0.0

        # the SECOND pillar's colour, and the lane distance it was seen at.
        # Direct evidence of the next straight, where next_straight_colour is
        # the fallback for a frame that only had one pillar in it at all.
        self.sec_seen_colour = None
        self.sec_seen_at = -1e9

        # the last pillar this straight passed on the INNER side, which is
        # what decides whether the corner becomes a 3-point
        self.have_inner_pass = False
        self.last_inner_pass_at = 0.0

        self.level_total = 0.0     # running total, logged per corner

    # ---------------------------------------------------------------- reset

    def clear_tracks(self):
        self.tracks = []
        self.pass_active = False

    def clear_cross_corner(self):
        self.next_straight_colour = None

    def clear_secondary(self):
        self.sec_seen_colour = None
        self.sec_seen_at = -1e9

    def reset_lane_along(self, odo_ticks=None):
        self.lane_along = 0.0
        if odo_ticks is not None:
            self._odo_base = odo_ticks
        self.have_inner_pass = False
        self.clear_cross_corner()
        self.clear_secondary()

    # ------------------------------------------------------- lane position

    def _lane_offset_raw(self, p, cone_l, cone_r):
        """(ok, offset) from the two cone fits. + = car left of centre."""
        limit = p["LANE_VALID_MAX_MM"]
        corridor = p["CORRIDOR_MM"]
        l = cone_l is not None and cone_l <= limit
        r = cone_r is not None and cone_r <= limit
        if l and r:
            both = 0.5 * (cone_r - cone_l)
            total = cone_l + cone_r
            if abs(total - corridor) < 150.0 or not self.lane_off_ok:
                return True, both
            # The walls do not add up to the corridor: one cone is fitted to
            # something else - a pillar beside the car, usually. Keep the side
            # that agrees with what we believed a moment ago.
            from_l = corridor / 2.0 - cone_l
            from_r = cone_r - corridor / 2.0
            return True, (from_l if abs(from_l - self.lane_off)
                          < abs(from_r - self.lane_off) else from_r)
        if l:
            return True, corridor / 2.0 - cone_l
        if r:
            return True, cone_r - corridor / 2.0
        return False, self.lane_off

    def _lane_offset(self, p, cone_l, cone_r, new_rev):
        """The jump filter. The car moves less than 40 mm sideways per LiDAR
        revolution, so a bigger step is a bad fit - unless it persists for
        OFF_JUMP_REVS, at which point it is real (just after a corner, say)."""
        ok, o = self._lane_offset_raw(p, cone_l, cone_r)
        if not ok:
            self._off_jumps = 0
            return False
        if self.lane_off_ok and abs(o - self.lane_off) > p["OFF_JUMP_MM"] \
                and self._off_jumps < OFF_JUMP_REVS:
            if new_rev:
                self._off_jumps += 1
            return True                     # keep the previous value
        self._off_jumps = 0
        self.lane_off = o
        return True

    # ------------------------------------------------------------ sightings

    def planned_next_colour(self, p):
        """Which colour is waiting on the next straight, or None.

        The SECOND pillar is direct evidence - a pillar the camera can still
        see that is further off than the one being passed - so it wins. The
        cross-corner sighting is the fallback for a frame in which only one
        pillar was visible at all.
        """
        if p["USE_SECONDARY_CORNER"] and self.sec_seen_colour is not None \
                and self.lane_along - self.sec_seen_at <= p["SECONDARY_ZONE_MM"]:
            return self.sec_seen_colour
        return self.next_straight_colour

    def secondary_is_live(self, p):
        """True when planned_next_colour() is answering from the second
        pillar rather than the cross-corner sighting. For the log line and
        the dashboard - the two want telling apart when a corner goes wrong."""
        return bool(p["USE_SECONDARY_CORNER"]
                    and self.sec_seen_colour is not None
                    and self.lane_along - self.sec_seen_at
                    <= p["SECONDARY_ZONE_MM"])

    def track_secondary(self, i: PlanInput):
        """Decide whether the second pillar is the next straight's.

        "Further away" is NOT on its own enough to mean "in the next
        straight". A straight with two pillars on it - which the rulebook
        allows - also presents a nearer and a further one, and shaping the
        corner from the second pillar of the CURRENT straight would be worse
        than not shaping it at all.

        So the colour is only believed when the geometry agrees: the second
        pillar is put into this lane's frame exactly as add_sighting does,
        and it counts only if it lands OUTSIDE this corridor, on the side the
        car is about to turn towards. Anything inside the corridor is just
        another pillar on this straight, and the ordinary planner deals with
        it.
        """
        p = i.p
        if not p["USE_SECONDARY_CORNER"] or i.sec_colour is None:
            return
        if i.lidar_stale or i.sec_xy is None or not self.lane_off_ok:
            return
        sx, sy = i.sec_xy
        if sx <= 0.0:                                   # behind the car
            return
        if i.pillar_xy is not None:                     # must be the further one
            px, py = i.pillar_xy
            if sx * sx + sy * sy <= px * px + py * py:
                return

        yaw = wrap180(i.heading - i.lane_heading) * DEG
        along = sx * math.cos(yaw) - sy * math.sin(yaw)
        lat = self.lane_off + sx * math.sin(yaw) + sy * math.cos(yaw)
        if along <= 0.0 or abs(lat) > p["CROSS_MAX_LAT_MM"]:
            return
        if abs(lat) <= p["PILLAR_MAX_LAT_MM"]:          # still in this corridor
            return
        if i.direction_known:
            # Before any floor colour has been read, turn_clockwise is a
            # guess. Rejecting on it then would throw away everything on one
            # side for the whole first straight, so the side test only
            # applies once the direction is actually known.
            toward_turn = (lat < 0.0) if i.turn_clockwise else (lat > 0.0)
            if not toward_turn:
                return                                  # the straight behind

        self.sec_seen_colour = i.sec_colour
        self.sec_seen_at = self.lane_along

    def add_sighting(self, i: PlanInput):
        """Fold the camera's pillar into the track table, in lane coordinates."""
        p = i.p
        if i.pillar_colour is None or i.pillar_xy is None or not self.lane_off_ok:
            return
        yaw = wrap180(i.heading - i.lane_heading) * DEG
        px, py = i.pillar_xy
        along = px * math.cos(yaw) - py * math.sin(yaw)
        lat_rel = px * math.sin(yaw) + py * math.cos(yaw)
        lat = self.lane_off + lat_rel
        if along < -50.0 or along > p["PLAN_MAX_AHEAD_MM"]:
            return

        if abs(lat) > p["PILLAR_MAX_LAT_MM"]:
            # A pillar of ANOTHER straight, seen across the corner. Its
            # position in this lane frame is meaningless, so it must not steer
            # the car here - but if it lies on the side the car is about to
            # turn towards, its COLOUR says which side the corner has to be
            # exited on. One bit, and it survives the corner perfectly, which
            # the coordinates do not: the car travels most of a corridor width
            # during the arc, so rotating an estimate 90 deg about the car is
            # wrong by hundreds of mm, and a confident wrong pillar is worse
            # than none.
            if p["CARRY_TRACKS"]:
                # Before any floor colour has been read the turn direction is
                # a guess, and rejecting on a guess throws away one whole side
                # for the first straight - so the side test waits for it.
                toward_turn = (not i.direction_known) or (
                    (lat < 0.0) if i.turn_clockwise else (lat > 0.0))
                if toward_turn and along > 0.0 and abs(lat) < p["CROSS_MAX_LAT_MM"]:
                    self.next_straight_colour = i.pillar_colour
            return

        at = self.lane_along + along
        match = p["TRACK_MATCH_MM"]
        for t in self.tracks:
            if t.colour == i.pillar_colour and abs(t.along - at) < match \
                    and abs(t.lat - lat) < match:
                t.lat += 0.5 * (lat - t.lat)          # refine
                t.along += 0.5 * (at - t.along)
                t.last_seen = self.lane_along
                t.hits = min(255, t.hits + 1)
                if t.hits == p["TRACK_CONFIRM"]:
                    self.note = (f"pillar {t.colour} lat={t.lat:.0f} "
                                 f"at={t.along:.0f}")
                return

        track = Track(colour=i.pillar_colour, lat=lat, along=at,
                      last_seen=self.lane_along)
        if len(self.tracks) < MAX_TRACKS:
            self.tracks.append(track)
        else:
            # evict the one furthest behind
            self.tracks[min(range(len(self.tracks)),
                            key=lambda k: self.tracks[k].along)] = track

    # ----------------------------------------------------------- reachability

    @staticmethod
    def lat_reach(s, psi0, radius_mm, yaw_max_deg):
        """Most sideways travel the car can make in `s` mm of lane, starting
        at yaw psi0 (rad, + = already angled TOWARD the target): arc at full
        lock up to yaw_max, then straight at that angle."""
        if s <= 0:
            return 0.0
        R, phi = radius_mm, yaw_max_deg * DEG
        psi0 = clamp(psi0, -phi, phi)
        a_arc = R * (math.sin(phi) - math.sin(psi0))    # lane used by the arc
        if s <= a_arc:                        # still arcing when we get there
            sp = math.asin(clamp(math.sin(psi0) + s / R, -1.0, 1.0))
            return R * (math.cos(psi0) - math.cos(sp))
        return R * (math.cos(psi0) - math.cos(phi)) + (s - a_arc) * math.tan(phi)

    # ----------------------------------------------------------------- tick

    def update(self, i: PlanInput):
        """One planner pass. Sets lat_yaw_cmd, lat_target, pass_active."""
        p = i.p
        self.note = ""

        # ---- lane distance: encoder projected onto the lane direction ----
        if self._odo_base is None:
            self._odo_base = i.odo_ticks
        dmm = (i.odo_ticks - self._odo_base) / max(1e-6, i.ticks_per_mm)
        self._odo_base = i.odo_ticks
        self.lane_along += dmm * math.cos(wrap180(i.heading - i.lane_heading) * DEG)

        if not i.new_frame:
            return
        self.lane_off_ok = self._lane_offset(p, i.cone_left, i.cone_right,
                                             i.new_rev)
        if not i.enabled:
            self.lat_yaw_cmd = 0.0
            self.pass_active = False
            return
        self.add_sighting(i)
        self.track_secondary(i)

        confirm = p["TRACK_CONFIRM"]
        pass_clear = p.derived["PASS_CLEAR_MM"]
        lane_limit = p.derived["LANE_LIMIT_MM"]

        # ---- which tracks are in play ----
        # Everything still within PASS_HOLD behind, up to and including the
        # NEAREST pillar ahead. Farther ones wait their turn.
        nearest_ahead = float("inf")
        keep = []
        for t in self.tracks:
            rel = t.along - self.lane_along
            if rel < -p["PASS_HOLD_MM"]:                          # passed
                if t.hits >= confirm and t.pass_right() == i.clockwise:
                    # the car went by on the INNER side
                    self.have_inner_pass = True
                    self.last_inner_pass_at = t.along
                continue
            if t.hits < confirm:                              # not trusted yet
                if self.lane_along - t.last_seen > p["TRACK_FORGET_MM"]:
                    continue
                keep.append(t)
                continue
            if 0 < rel < nearest_ahead:
                nearest_ahead = rel
            keep.append(t)
        self.tracks = keep

        # ---- the bounds every pillar in play imposes ----
        lo, hi = -lane_limit, lane_limit
        urgent_rel, urgent_bound, any_pillar = float("inf"), 0.0, False
        aim_mm = p["CENTRE_AIM_MM"]
        for t in self.tracks:
            if t.hits < confirm:
                continue
            rel = t.along - self.lane_along
            if rel > nearest_ahead + 1.0:
                continue
            any_pillar = True
            if t.pass_right():
                bound = t.lat - pass_clear
                hi = min(hi, bound)
            else:
                bound = t.lat + pass_clear
                lo = max(lo, bound)
            if rel < urgent_rel:
                urgent_rel, urgent_bound = rel, bound
        self.pass_active = any_pillar

        # ---- the default lane target, with no pillar in play ----
        # Normally the lane centre; just after a corner it is the commanded
        # exit offset, and just before one the pre-corner swing. Both are
        # signed "+ = OUTER side of the lap", so one number covers clockwise
        # and anticlockwise.
        target = 0.0
        if not any_pillar:
            if i.corner_count > 0 and self.lane_along < p["POST_CORNER_BOOST_MM"]:
                target = self.corner_exit_cmd
            elif p["PRE_CORNER_ZONE_MM"] > 0.0 and not i.lidar_stale \
                    and i.front_mm < p["PRE_CORNER_ZONE_MM"]:
                # Enter wide to leave tight. A 90 deg arc throws the car AWAY
                # from the side it started on, so the way to come out of a
                # corner near the inner wall is to go into it near the outer
                # one - the ordinary racing line. The exit side is already
                # known here (the second pillar told us before the corner),
                # so the entry is simply its OPPOSITE.
                nxt = self.planned_next_colour(p)
                if nxt is None:
                    exit_side = 1.0 if i.clockwise else -1.0
                else:
                    exit_side = -1.0 if nxt == RED else 1.0
                target = -exit_side * p["PRE_CORNER_SWING_MM"]
        if lo > hi:
            target = urgent_bound          # conflict: most urgent pillar wins
        else:
            target = clamp(target, lo, hi)

        if any_pillar:
            # Aim at the pass point: the target lane position PASS_LEAD before
            # the nearest pillar ahead, so the swerve sharpens as it closes and
            # arrives in time. Alongside or past it, a gentle hold.
            if nearest_ahead < 1e8 and nearest_ahead > p["PASS_LEAD_MM"]:
                aim_mm = max(nearest_ahead - p["PASS_LEAD_MM"], p["PASS_AIM_MIN_MM"])
            else:
                aim_mm = p["HOLD_AIM_MM"]
            target = self._reach_check(i, target, nearest_ahead, lane_limit,
                                       pass_clear, confirm)

        # ---- an object whose colour is still unknown ----
        t2, a2, active2 = self._unknown(i, target, aim_mm, nearest_ahead,
                                        lane_limit, pass_clear, confirm)
        if active2:
            target, aim_mm = t2, a2
            self.pass_active = True

        self.lat_target = clamp(target, -lane_limit, lane_limit)
        self.aim_mm = aim_mm

        if not self.lane_off_ok:
            self.lat_yaw_cmd = 0.0          # no walls: just hold the heading
            return

        # Centring authority is deliberately gentle so the car does not weave
        # down a straight. Right after a corner that is the wrong trade: the
        # car may have a whole lane width to recover and only a few hundred mm
        # to do it in, so it gets PASS-level authority for the first stretch.
        if self.pass_active:
            ymax = p["PASS_YAW_MAX"]
        elif self.lane_along < p["POST_CORNER_BOOST_MM"]:
            ymax = p["POST_CORNER_YAW_MAX"]
        else:
            ymax = p["CENTRE_YAW_MAX"]
        # + lateral error = the target is to the LEFT = yaw left
        self.lat_yaw_cmd = clamp(
            math.atan2(self.lat_target - self.lane_off, aim_mm) / DEG,
            -ymax, ymax)

    # ------------------------------------------------- give up, or just miss

    def _reach_check(self, i: PlanInput, target, nearest_ahead, lane_limit,
                     pass_clear, confirm):
        """Can the car still get to the pass position before the nearest
        pillar? If not, commit to its other side rather than hit it."""
        p = i.p
        if not (p["ALLOW_GIVE_UP"] and self.lane_off_ok
                and 0 < nearest_ahead < 1e8):
            return target
        need = abs(target - self.lane_off)
        s_avail = nearest_ahead - p["GIVEUP_LEAD_MM"]
        yaw_now = wrap180(i.heading - i.lane_heading) * DEG        # + = left
        psi0 = yaw_now if target > self.lane_off else -yaw_now     # + = toward
        radius, ymax = p["TURN_RADIUS_MM"], p["PASS_YAW_MAX"]
        if not (need > 60.0
                and need > self.lat_reach(s_avail, psi0, radius, ymax) + 30.0):
            return target

        flipped = False
        for t in self.tracks:
            if t.hits < confirm or t.flipped:
                continue
            if abs((t.along - self.lane_along) - nearest_ahead) > 1.0:
                continue
            other = (t.lat + pass_clear if t.colour == RED
                     else t.lat - pass_clear)
            need_o = abs(other - self.lane_off)
            psi_o = yaw_now if other > self.lane_off else -yaw_now
            # Switch only if the other side can really be reached from here.
            # Late in a swerve it cannot, and swinging back is worse.
            if abs(other) <= lane_limit and need_o < need \
                    and need_o <= self.lat_reach(s_avail, psi_o, radius, ymax):
                t.flipped = True
                flipped = True
                target = other
                self.note = (f"give up: {t.colour} unreachable, need "
                             f"{need:.0f} in {s_avail:.0f}")
        if flipped:
            return target

        # NEITHER side reachable with full clearance. Holding the unreachable
        # target here drives the car straight into the pillar. A wrong-side
        # pass costs points; a collision costs the run. So fall back to the
        # bare geometric miss distance (no PASS_MARGIN) and take whichever
        # side is closer to reach, correct or not.
        avoid_clear = i.p.derived["AVOID_CLEAR_MM"]
        for t in self.tracks:
            if t.hits < confirm:
                continue
            if abs((t.along - self.lane_along) - nearest_ahead) > 1.0:
                continue
            a, b = t.lat - avoid_clear, t.lat + avoid_clear
            na, nb = abs(a - self.lane_off), abs(b - self.lane_off)
            pick = a if na <= nb else b
            if abs(pick) > lane_limit:              # that side is wall
                pick = b if na <= nb else a
            if abs(pick - self.lane_off) < need:
                target = pick
                self.note = (f"AVOID only: pillar at {nearest_ahead:.0f} mm, "
                             f"aiming {pick:.0f}")
        return target

    # --------------------------------------------------- unknown-colour object

    def _unknown(self, i: PlanInput, target, aim_mm, nearest_ahead, lane_limit,
                 pass_clear, confirm):
        """The camera only covers about +/-48 deg, so a pillar near the far
        wall can be out of view until too late. Line up with it - it comes
        into view and both passing sides stay open - and if it is still
        unnamed at UNK_COMMIT_MM, dodge to the side with more room so the car
        never simply drives into it."""
        p = i.p
        if i.unknown_xy is None or not self.lane_off_ok:
            return target, aim_mm, False
        yaw = wrap180(i.heading - i.lane_heading) * DEG
        ux, uy = i.unknown_xy
        u_along = ux * math.cos(yaw) - uy * math.sin(yaw)
        u_lat = self.lane_off + ux * math.sin(yaw) + uy * math.cos(yaw)

        match = p["TRACK_MATCH_MM"]
        for t in self.tracks:                  # same place as a named pillar?
            if t.hits >= confirm \
                    and abs(t.along - (self.lane_along + u_along)) < match \
                    and abs(t.lat - u_lat) < match:
                return target, aim_mm, False

        if not (100.0 < u_along < p["PLAN_MAX_AHEAD_MM"]
                and abs(u_lat) < p["PILLAR_MAX_LAT_MM"]
                and u_along < nearest_ahead - 150.0):
            return target, aim_mm, False

        # only pillars already alongside may constrain us now
        lo2, hi2 = -lane_limit, lane_limit
        for t in self.tracks:
            if t.hits < confirm or t.along - self.lane_along > 50.0:
                continue
            if t.pass_right():
                hi2 = min(hi2, t.lat - pass_clear)
            else:
                lo2 = max(lo2, t.lat + pass_clear)

        commit = p["UNK_COMMIT_MM"]
        if u_along > commit:
            t_new = u_lat                                   # line up with it
            a_new = max(u_along - commit, 250.0)
        else:
            t_new = u_lat + pass_clear if u_lat < 0 else u_lat - pass_clear
            a_new = max(u_along - p["PASS_LEAD_MM"], p["PASS_AIM_MIN_MM"])
        t_new = clamp(t_new, lo2, hi2) if lo2 <= hi2 else t_new
        return t_new, a_new, True

    # ------------------------------------------------------------ levelling

    def level_step(self, i: PlanInput):
        """How far to nudge the lane heading toward the fitted wall direction.

        The wall fit is noisy per revolution but has no drift; the IMU is
        smooth but drifts. So the lane heading is pulled a small step toward
        the fit once per revolution - and only while the fit is trustworthy:
        both cone walls inside a corridor width, the car not badly yawed, no
        pillar steering it, and the estimate close enough to what the IMU
        already believes that a disagreement means a bad fit, not drift.

        Returns the step in degrees (0 when it should not run). The caller
        owns the lane heading and applies it.
        """
        p = i.p
        if not (i.new_rev and not i.lidar_stale and i.wall_ang is not None):
            return 0.0
        limit = p["LANE_VALID_MAX_MM"]
        if i.cone_left is None or i.cone_right is None \
                or i.cone_left > limit or i.cone_right > limit:
            return 0.0
        if abs(i.wall_ang) > p["LEVEL_MAX_WALLANG"]:
            return 0.0
        if self.pass_active:                   # swerving: the car is not level
            return 0.0
        est = wrap180(i.heading - i.wall_ang)
        diff = wrap180(est - i.lane_heading)
        if abs(diff) > p["LEVEL_MAX_DIFF"]:
            return 0.0
        step = clamp(p["LEVEL_GAIN"] * diff,
                     -p["LEVEL_MAX_STEP"], p["LEVEL_MAX_STEP"])
        self.level_total += step
        return step

    # ----------------------------------------------------------- 3-point cue

    def inner_pillar_near_corner(self, p, clockwise):
        """True when a pillar was (or is being) passed on the INNER side just
        before the corner - the cue for the 3-point corner. Clockwise only.

        Requires plan_corner_exit() to have run already, because the second
        half of the rule is what corner_exit_cmd says.
        """
        if not p["USE_CORNER_MANEUVER"] or not clockwise:
            return False

        # Only when the corner has to come out on the INNER side.
        #
        # The 3-point exists because an inner-side pillar before the corner
        # pins the car against the inner wall, and a plain arc from there
        # lands it on the OUTER side of the next straight - fatal if the next
        # pillar needs the inner side too. If the next one needs the outer
        # side, that plain arc is already doing the right thing and stopping
        # to shuffle only costs clearance: in sim the outer-side greens went
        # from 150-393 mm of clearance down to 79-92 mm when the maneuver
        # fired on them for nothing.
        if self.corner_exit_cmd > 0.0:          # CW: positive = outer
            return False

        zone = p["MNV_ZONE_MM"]
        if self.have_inner_pass and self.lane_along - self.last_inner_pass_at < zone:
            return True
        for t in self.tracks:                  # still alongside / just ahead
            if t.hits < p["TRACK_CONFIRM"]:
                continue
            rel = t.along - self.lane_along
            if rel < -zone or rel > 250.0:
                continue
            if t.pass_right() == clockwise:
                return True
        return False

    # ------------------------------------------------------------- snapshot

    def snapshot(self):
        return {
            "lane_off": round(self.lane_off, 1),
            "lane_off_ok": self.lane_off_ok,
            "lane_along": round(self.lane_along, 1),
            "lat_target": round(self.lat_target, 1),
            "lat_yaw_cmd": round(self.lat_yaw_cmd, 2),
            "aim_mm": round(self.aim_mm, 0),
            "pass_active": self.pass_active,
            "next_straight": self.next_straight_colour,
            "sec_colour": self.sec_seen_colour,
            "sec_age_mm": (None if self.sec_seen_colour is None
                           else round(self.lane_along - self.sec_seen_at, 0)),
            "corner_exit_cmd": round(self.corner_exit_cmd, 1),
            "have_inner_pass": self.have_inner_pass,
            "level_total": round(self.level_total, 2),
            "tracks": [{"colour": t.colour, "lat": round(t.lat, 0),
                        "along": round(t.along, 0), "hits": t.hits,
                        "flipped": t.flipped,
                        "rel": round(t.along - self.lane_along, 0)}
                       for t in self.tracks],
        }
