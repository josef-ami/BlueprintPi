"""
planner.py - a lap-length path plan over the known arena, and a tracker.

THE SHIFT
    The current firmware decides where to be from what it can see right now:
    a lane offset from two wall fits, a target from the nearest confirmed
    pillar, a corner shape guessed from whichever sign happened to be the
    second-largest blob. Every one of those decisions has a horizon of a few
    hundred millimetres, and the corner throws the state away.

    With a global pose and a global sign map, the whole lap is a single
    geometry problem that can be solved once and re-solved whenever the map
    changes. The car is no longer reacting to a pillar 300 mm away; it is
    following a line that was drawn to clear that pillar a metre and a half
    before reaching it. That is the difference between needing a 75-degree
    swerve and needing a 12-degree lane change.

THE SOLVER
    The reference line is the mid-corridor loop: four 1000 mm straights and
    four 500 mm-radius quarter arcs (lap = 7142 mm). 500 mm is comfortably
    inside the measured 250-270 mm minimum radius, so the reference itself is
    always drivable.

    Everything is then expressed as a lateral offset lat(s) from that line:

      * corridor walls give a hard interval [lo(s), hi(s)] at every station,
        computed from the real arena geometry rather than assumed symmetric -
        which matters at corners, where the outside of the bend is 646 mm
        away and the inside only 500 mm.
      * each sign narrows one side over the stations where the car body would
        overlap it: red forbids everything right of lat_p - clear, green
        everything left of lat_p + clear.
      * a kinematic rate limit |dlat/ds| <= tan(35 deg) is propagated through
        the interval in a forward and a backward pass, which is what turns a
        local constraint into an early lane change. A sign at station s only
        reachable by starting to move 800 mm earlier tightens the interval
        800 mm earlier, automatically.

    If the interval closes, the plan is infeasible and the solver relaxes in
    a fixed order - tighter clearance first, then, only if that still fails,
    a deliberate wrong-side pass. That ordering is the policy decision the
    firmware currently makes implicitly and inconsistently: a wrong-side pass
    costs points, a collision costs the run.
"""

from __future__ import annotations

import math

import numpy as np

from .geom import raycast, WALL_SEGMENTS, MID, INNER, OUTER

# 35 mm stations. The dilation is O(REACH_W * n) and both scale with
# 1/ds, so halving the resolution is ~4x cheaper. 35 mm is still well
# under the shortest feature the plan has to represent (a 326 mm sign
# constraint window), and the tracker interpolates between stations.
STEP_MM = 22.0

# Reference corner radius. The car's measured full-lock radius is 270 mm, so
# a 500 mm reference arc spends 54% of the entire steering budget just
# following the centreline, leaving almost nothing for avoidance ON the
# corner - and the corner is where the signs the team loses runs to are.
# 700 mm cuts that to 39% and still clears the inner block corner by 417 mm.
# The tangent points move to +/-(MID - ARC_R) along each straight, so this
# must stay below MID.
ARC_R = 700.0
TANGENT = MID - ARC_R                       # half-length of each straight
LAP_MM = 4 * (2 * TANGENT) + 4 * (0.5 * math.pi * ARC_R)

# Curvature the plan is allowed to ask for. Full lock is 1/270 = 0.0037/mm;
# planning to 70% of that leaves the tracker authority to correct with.
K_CAR = 1.0 / 270.0
K_PLAN = 0.70 * K_CAR


def _build_ccw():
    """Dense anticlockwise centreline: (x, y, heading) every STEP_MM."""
    pts = []

    def straight(x1, y1, x2, y2):
        L = math.hypot(x2 - x1, y2 - y1)
        n = max(1, int(round(L / STEP_MM)))
        h = math.atan2(y2 - y1, x2 - x1)
        for i in range(n):
            f = i / n
            pts.append((x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, h))

    def arc(cx, cy, a0, a1):
        L = ARC_R * abs(a1 - a0)
        n = max(1, int(round(L / STEP_MM)))
        for i in range(n):
            a = a0 + (a1 - a0) * i / n
            pts.append((cx + ARC_R * math.cos(a), cy + ARC_R * math.sin(a),
                        a + math.pi / 2))          # CCW: tangent leads by 90 deg

    h = TANGENT
    c = MID - ARC_R                                 # arc-centre offset
    straight(-h, -MID, h, -MID)                     # S, +x
    arc(c, -c, -math.pi / 2, 0.0)                   # SE
    straight(MID, -h, MID, h)                       # E, +y
    arc(c, c, 0.0, math.pi / 2)                     # NE
    straight(h, MID, -h, MID)                       # N, -x
    arc(-c, c, math.pi / 2, math.pi)                # NW
    straight(-MID, h, -MID, -h)                     # W, -y
    arc(-c, -c, math.pi, 1.5 * math.pi)             # SW
    return pts


class Path:
    """Closed reference loop with per-station corridor bounds.

    Built once per direction and cached - the arena never changes, so this is
    a compile-time constant in everything but name.
    """
    _cache = {}

    def __new__(cls, clockwise: bool, half_w: float = 57.1, margin: float = 45.0):
        key = (clockwise, round(half_w, 2), round(margin, 2))
        if key in cls._cache:
            return cls._cache[key]
        self = super().__new__(cls)
        self._init(clockwise, half_w, margin)
        cls._cache[key] = self
        return self

    def _init(self, clockwise, half_w, margin):
        pts = _build_ccw()
        if clockwise:
            # same loop, other way round: reverse the order and flip tangents
            pts = [(x, y, h + math.pi) for (x, y, h) in reversed(pts)]
        self.clockwise = clockwise
        self.n = len(pts)
        self.xy = np.array([(p[0], p[1]) for p in pts])
        self.th = np.array([p[2] for p in pts])
        self.ds = LAP_MM / self.n
        self.s = np.arange(self.n) * self.ds
        # left normal
        self.nx = -np.sin(self.th)
        self.ny = np.cos(self.th)

        # corridor bounds from the real geometry: cast along +/- the normal
        hi = np.zeros(self.n)
        lo = np.zeros(self.n)
        for i in range(self.n):
            x, y = self.xy[i]
            aL = math.atan2(self.ny[i], self.nx[i])
            dL = raycast(x, y, np.array([aL]), WALL_SEGMENTS, 3000.0)[0]
            dR = raycast(x, y, np.array([aL + math.pi]), WALL_SEGMENTS, 3000.0)[0]
            hi[i] = (dL if np.isfinite(dL) else 500.0) - half_w - margin
            lo[i] = -((dR if np.isfinite(dR) else 500.0) - half_w - margin)
        self.hi0 = hi
        self.lo0 = lo

        # reference curvature per station, and the curvature budget left over
        # for the lateral profile after the reference corner has taken its cut
        dth = np.diff(np.unwrap(self.th), append=self.th[0])
        dth = (dth + np.pi) % (2 * np.pi) - np.pi
        self.k_ref = dth / self.ds
        self.k_budget = np.maximum(K_PLAN - np.abs(self.k_ref), 0.00025)

    # ---- projection ----
    def project(self, x, y):
        """(station index, lateral offset, heading error basis)."""
        d = self.xy - np.array([x, y])
        i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        lat = -(d[i, 0] * self.nx[i] + d[i, 1] * self.ny[i])
        return i, lat

    def point(self, i):
        i %= self.n
        return self.xy[i, 0], self.xy[i, 1], self.th[i]

    def offset_point(self, i, lat):
        i %= self.n
        return (self.xy[i, 0] + self.nx[i] * lat,
                self.xy[i, 1] + self.ny[i] * lat)


# --------------------------------------------------------------- the solver
PASS_MARGIN_MM = 80.0
BARE_MARGIN_MM = 20.0
PILLAR_HALF = 25.0
SLOPE = math.tan(math.radians(35.0))

# Longitudinal window over which a sign constrains the line. This is the
# body-overlap span, not a guess: the reference point is the rear axle, the
# nose is 197 mm ahead of it and the tail 29 mm behind, and the sign is 25 mm
# half-wide. So the constraint starts when the nose reaches the sign and ends
# when the tail clears it.
#
# Getting this too wide is expensive. At 260/170 the hard window was 430 mm
# long, so two signs 500 mm apart left only 70 mm of free track between their
# windows to swap sides in - which made a flip mandatory on layouts that did
# not need one.
LEAD_MM = 197.0 + PILLAR_HALF + 30.0        # nose + sign + slack
TRAIL_MM = 29.0 + PILLAR_HALF + 20.0        # tail + sign + slack

# Body width grows with yaw: a car at 20 degrees of yaw presents
# half_w*cos + nose*sin = 120 mm of half-width, not 57. The plan is drawn for
# the rear axle, so it has to reserve for that.
YAW_WIDTH_MM = 30.0


# Curvature actually available to the lateral profile at the worst station
# (on a reference arc, most of the budget is already spent turning the
# corner). Used as one conservative constant for the reachability kernel.
K_MOVE = max(K_PLAN - 1.0 / ARC_R, 0.0004)
REACH_W = 73                       # stations each side = ~1600 mm of lookahead

def _propagate(lo, hi, ds, path):
    """Grow each bound by what the car can actually reach from it.

    A min-plus dilation: hi[i] = min over nearby j of (hi[j] + move(|i-j|)),
    and the mirror for lo. That is the exact statement of "the line at i must
    be somewhere the car could still be at j", and it wraps round the loop.

    move() uses the curvature budget actually available BETWEEN i and j - the
    running minimum along that stretch - rather than one global worst case.
    The difference is large and in the direction that matters: on a straight
    the budget is 0.00259/mm and a 300 mm lane change needs 681 mm, while the
    worst-case arc budget of 0.00116 would demand 1017 mm. Most signs sit on
    straights, so using the arc figure everywhere threw away a third of the
    available lookahead and turned solvable layouts into "infeasible".

    This is also what gives the plan its lookahead. A sign needing 700 mm of
    lane change tightens the interval 700 mm earlier by itself, so the car
    starts moving over long before it - the difference between a 12 degree
    lane change and a 75 degree swerve.
    """
    kb = path.k_budget
    out_lo, out_hi = lo.copy(), hi.copy()
    for sgn in (-1, 1):
        run = kb.copy()
        for o in range(1, REACH_W + 1):
            run = np.minimum(run, np.roll(kb, sgn * o))
            d = o * ds
            mv = np.minimum(SLOPE * d, run * d * d / 4.0)
            np.minimum(out_hi, np.roll(hi, sgn * o) + mv, out=out_hi)
            np.maximum(out_lo, np.roll(lo, sgn * o) - mv, out=out_lo)
    return out_lo, out_hi


def _prep(path, pillars, half_w, margin):
    """Per-sign: station window, lateral position, and the natural side.

    Cached geometry so the relaxation search can re-assign sides without
    re-projecting anything.
    """
    n = path.n
    k0 = int(round(LEAD_MM / path.ds))
    k1 = int(round(TRAIL_MM / path.ds))
    clear = PILLAR_HALF + half_w + margin
    out = []
    for p in pillars:
        i, lat = path.project(p.x, p.y)
        idx = np.array([(i + k) % n for k in range(-k0, k1 + 1)])
        # RED is passed on its right -> the car must be right of it
        want_right = (p.color == 1)
        known = p.color in (0, 1)
        out.append(dict(idx=idx, lat=lat, clear=clear + YAW_WIDTH_MM,
                        want_right=want_right, known=known))
    return out


def _build(path, specs, sides, anchor=None):
    lo = path.lo0.copy()
    hi = path.hi0.copy()
    for sp, right in zip(specs, sides):
        if right:
            np.minimum.at(hi, sp["idx"], sp["lat"] - sp["clear"])
        else:
            np.maximum.at(lo, sp["idx"], sp["lat"] + sp["clear"])
    if anchor is not None:
        alo, ahi = _anchor(path, lo, hi, anchor)
        plo, phi = _propagate(alo, ahi, path.ds, path)
        if np.all(plo <= phi):
            return plo, phi
        # the anchor made it infeasible - the car is already somewhere the
        # signs forbid. Drop it rather than declare the layout impossible.
    return _propagate(lo, hi, path.ds, path)


# A tube narrower than this is not a plan. lo <= hi is satisfiable by a tube
# 3 mm wide, and a 3 mm tube dictates the line exactly - including every kink
# in the bound, which is how feasible tubes were still producing lines at
# 240% of full lock. Requiring real width makes the solver relax (flip a
# sign) instead of accepting a solution with no margin in it.
MIN_TUBE_MM = 60.0


def _violation(lo, hi):
    v = MIN_TUBE_MM - (hi - lo)
    return float(v[v > 0].sum())


def _feasible(lo, hi):
    return bool(np.all(hi - lo >= MIN_TUBE_MM))


def _search_sides(path, specs, max_flips=8, anchor=None):
    """Choose which side of each sign to pass, minimising wrong-side passes.

    The decision has to be made AFTER the rate limit is propagated, not
    before. Two signs 500 mm apart demanding opposite sides do not conflict
    locally at all - their constraint windows never even touch. The conflict
    only exists because the car cannot cross the corridor in 500 mm, and that
    fact lives in the rate limit. An earlier version of this solver decided
    sides from the local windows and therefore never found a flip worth
    making; it reported two thirds of layouts as unsolvable when they were
    merely awkward.

    Greedy on total violation: start from every sign on its correct side,
    then repeatedly flip whichever single sign reduces the violation most.
    """
    # PRE-PASS: a sign whose correct side is unreachable against the bare
    # corridor - before any other sign is considered - is flipped up front.
    # The greedy search below moves one sign at a time and stalls when the
    # violation is caused by several at once, so it never found these.
    #
    # This is the common case at a corner. The seats at a section boundary
    # project onto the corner arc, where the inner-side bound is 319 mm while
    # a sign at lat 189 needs the car at 322 - short by 3 mm, and no amount
    # of moving OTHER signs fixes it.
    sides = []
    for sp in specs:
        want_right = sp["want_right"]
        idx = sp["idx"]
        if want_right:
            reachable = (path.lo0[idx] <= sp["lat"] - sp["clear"]).all()
        else:
            reachable = (path.hi0[idx] >= sp["lat"] + sp["clear"]).all()
        if not reachable:
            other_ok = ((path.hi0[idx] >= sp["lat"] + sp["clear"]).all()
                        if want_right else
                        (path.lo0[idx] <= sp["lat"] - sp["clear"]).all())
            if other_ok:
                want_right = not want_right
        sides.append(want_right)
    lo, hi = _build(path, specs, sides, anchor)
    v = _violation(lo, hi)
    if v <= 0:
        return sides, lo, hi, 0
    # EXHAUSTIVE over small flip sets before falling back to greedy.
    # Greedy moves one sign at a time and stalls whenever the violation is
    # produced by two signs jointly - neither single flip improves it, so it
    # gives up and the whole layout is written off as unsolvable. Most real
    # layouts need one or two flips, and 1 + n + n(n-1)/2 evaluations is
    # about 37 for eight signs, which is affordable.
    import itertools
    m = len(specs)
    for r in (1, 2):
        if m < r:
            break
        best_c = None
        for combo in itertools.combinations(range(m), r):
            trial = list(sides)
            for k in combo:
                trial[k] = not trial[k]
            tlo, thi = _build(path, specs, trial, anchor)
            tv = _violation(tlo, thi)
            if tv <= 0:
                wrong = sum(1 for sp, s_ in zip(specs, trial)
                            if sp["known"] and s_ != sp["want_right"])
                return trial, tlo, thi, wrong
            if best_c is None or tv < best_c[0]:
                best_c = (tv, trial, tlo, thi)
        if best_c is not None and best_c[0] < v:
            v, sides, lo, hi = best_c[0], best_c[1], best_c[2], best_c[3]

    flips = 0
    while flips < max_flips and flips < len(specs):
        best = None
        for k in range(len(specs)):
            if sides[k] != specs[k]["want_right"]:
                continue                      # already flipped
            trial = list(sides)
            trial[k] = not trial[k]
            tlo, thi = _build(path, specs, trial, anchor)
            tv = _violation(tlo, thi)
            if best is None or tv < best[0]:
                best = (tv, k, tlo, thi)
        if best is None or best[0] >= v:
            break
        v, k, lo, hi = best
        sides[k] = not sides[k]
        flips += 1
        if v <= 0:
            break
    wrong = sum(1 for sp, s in zip(specs, sides)
                if sp["known"] and s != sp["want_right"])
    return sides, lo, hi, wrong


ANCHOR_TOL_MM = 25.0
ANCHOR_BACK = 6          # stations behind the car that the anchor pins


def _anchor(path, lo, hi, anchor):
    """Pin the tube to where the car actually is, just behind it.

    Without this the plan is an abstract line re-solved from scratch every
    replan. The moment a new sign enters the map the whole profile can step
    sideways, and the car - which was tracking the OLD line accurately - is
    instantly a couple of hundred millimetres off the new one and swerves to
    catch up. Measured: mean cross-track 26 mm, p95 54 mm, but 246 mm at the
    instant of the only contact in the run.

    Anchoring makes each replan a trajectory FROM THE CURRENT STATE. The
    rate limit then propagates forward from the car's real position, so the
    new plan is something the car can actually pick up smoothly.
    """
    i, lat = anchor
    n = path.n
    idx = np.array([(i - k) % n for k in range(ANCHOR_BACK)])
    lo2, hi2 = lo.copy(), hi.copy()
    np.maximum.at(lo2, idx, lat - ANCHOR_TOL_MM)
    np.minimum.at(hi2, idx, lat + ANCHOR_TOL_MM)
    return lo2, hi2


def _greedy_partial(path, pillars, half_w, anchor):
    """Honour as many signs as will fit, instead of none at all.

    When no assignment satisfies every sign, the old fallback dropped the
    whole sign set and drove the bare corridor. That keeps the line smooth
    and collision-free - measured zero contacts - but throws away every
    passing point on the lap: 5 of 6 signs passed on the wrong side.

    Almost all of those signs were not in conflict with anything. So instead
    of all-or-nothing, add them one at a time, keeping each only if the tube
    survives, and degrade each sign individually through a preference order:

        1. correct side, full clearance      - full points
        2. correct side, bare miss           - full points, tight
        3. wrong side, full clearance        - points lost, safe
        4. wrong side, bare miss             - points lost, tight
        5. keep-out on the roomier side      - points lost, but NOT a contact

    Only if even (5) will not fit is a sign dropped, and a dropped sign is
    the only case that can still produce contact.
    """
    lo = path.lo0.copy()
    hi = path.hi0.copy()
    lo, hi = _propagate(lo, hi, path.ds, path)
    if anchor is not None:
        alo, ahi = _anchor(path, lo, hi, anchor)
        plo, phi = _propagate(alo, ahi, path.ds, path)
        if _feasible(plo, phi):
            lo, hi = plo, phi

    order = sorted(range(len(pillars)),
                   key=lambda k: path.project(pillars[k].x, pillars[k].y)[0])
    wrong = 0
    dropped = 0
    for k in order:
        p = pillars[k]
        placed = False
        for margin, flip in ((PASS_MARGIN_MM, False), (BARE_MARGIN_MM, False),
                             (PASS_MARGIN_MM, True), (BARE_MARGIN_MM, True),
                             (BARE_MARGIN_MM, None)):
            sp = _prep(path, [p], half_w, margin)[0]
            want_right = sp["want_right"]
            if flip is True:
                want_right = not want_right
            elif flip is None:
                room_r = (sp["lat"] - sp["clear"]) - lo[sp["idx"]].max()
                room_l = hi[sp["idx"]].min() - (sp["lat"] + sp["clear"])
                want_right = room_r >= room_l
            t_lo, t_hi = lo.copy(), hi.copy()
            if want_right:
                np.minimum.at(t_hi, sp["idx"], sp["lat"] - sp["clear"])
            else:
                np.maximum.at(t_lo, sp["idx"], sp["lat"] + sp["clear"])
            t_lo, t_hi = _propagate(t_lo, t_hi, path.ds, path)
            # The last option is a bare keep-out on the roomier side. It is
            # the difference between a lost point and a contact, so it is
            # allowed a thinner tube than the others.
            ok = (bool(np.all(t_hi - t_lo >= 18.0)) if flip is None
                  else _feasible(t_lo, t_hi))
            if ok:
                lo, hi = t_lo, t_hi
                if want_right != sp["want_right"] and sp["known"]:
                    wrong += 1
                placed = True
                break
        if not placed:
            dropped += 1
    return lo, hi, wrong, dropped


def solve(path: Path, pillars, half_w: float = 57.1, anchor=None,
          start_stage: int = 0):
    """Lateral profile lat(s) for the whole lap.

    Relaxation order is deliberate, and is the policy the current firmware
    applies inconsistently across three different code paths:
        1. full passing clearance, every sign on its correct side
        2. bare geometric miss - tight, but legal and no contact
        3. deliberate wrong-side passes, fewest first
    A wrong-side pass costs points. A collision costs the run. So the order
    is never "hit it rather than break the rule".
    """
    best_fail = None
    # Graduated relaxation. Dropping straight from "full clearance" to
    # "ignore every sign" throws away the many layouts that are solvable with
    # a tighter but still real margin. Clearance is given up before tube
    # width, and tube width before correctness of side, because that is the
    # order in which the cost goes up: a tighter miss is free, a thinner
    # margin risks contact, a wrong side is certain points lost.
    ladder = ((PASS_MARGIN_MM, 60.0), (BARE_MARGIN_MM, 60.0),
              (BARE_MARGIN_MM, 40.0), (BARE_MARGIN_MM, 22.0))
    global MIN_TUBE_MM
    _saved_tube = MIN_TUBE_MM
    for stage, (margin, tube) in enumerate(ladder):
        if stage < start_stage:
            continue
        MIN_TUBE_MM = tube
        specs = _prep(path, pillars, half_w, margin)
        # unknown-colour signs: put them on whichever side the walls allow,
        # then let the search move them like any other
        for sp in specs:
            if not sp["known"]:
                sp["want_right"] = sp["lat"] > 0
        sides, lo, hi, wrong = _search_sides(path, specs, anchor=anchor)
        if _feasible(lo, hi):
            lat = _line(lo, hi, path)
            kr = line_curvature(path, lat)
            if kr <= 1.0:
                MIN_TUBE_MM = _saved_tube
                return lat, dict(stage=(0 if (stage == 0 and wrong == 0) else
                                        (1 if wrong == 0 else 2)),
                                 wrong=wrong, feasible=True, kratio=kr,
                                 margin=margin, tube=tube)
            if best_fail is None:
                best_fail = (lo, hi, wrong)
        if best_fail is None:
            best_fail = (lo, hi, wrong)
    # Still infeasible. Collapsing the contradictory tube produces a line
    # with kinks in it that the steering cannot follow - measured at 1222% of
    # full lock - and a car flailing at full lock is worse than one driving
    # calmly past a sign on the wrong side. So fall back to the WALL-ONLY
    # tube, which is wide and smooth by construction, and take the points hit.
    # The partial tube can still be too tight to fit a drivable line into, so
    # widen the width requirement until the fitted line is one the steering
    # can actually produce. A car flailing at full lock past a sign is worse
    # than one driving calmly past it on the wrong side.
    best = None
    for tube in (75.0, 60.0, 48.0):
        MIN_TUBE_MM = tube
        lo, hi, wrong, dropped = _greedy_partial(path, pillars, half_w, anchor)
        lat = _line(lo, hi, path)
        kr = line_curvature(path, lat)
        if best is None or kr < best[0]:
            best = (kr, lat, wrong, dropped)
        if kr <= 1.0:
            break
    MIN_TUBE_MM = _saved_tube
    kr, lat, wrong, dropped = best
    return lat, dict(stage=3, wrong=wrong, dropped=dropped, feasible=False,
                     kratio=kr)


# width below which the line stops hugging a bound and moves to the middle
TIGHT_MM = 320.0


def _line(lo, hi, path, iters: int = 45):
    """Pick a DRIVABLE lateral profile inside the feasible tube.

    Two things an earlier version got wrong, both measured rather than
    guessed:

    1. Hugging the bound. clip(0, lo, hi) puts the line exactly ON the
       constraint wherever a sign pushes it, budgeting nothing for tracking
       error. Measured clearances of 2 mm and -29 mm came from that. The line
       is blended toward the MIDDLE of the tube in proportion to how narrow
       the tube is: centred where there is room, splitting the difference
       where there is not.

    2. Curvature, and how it is imposed. Bounding the second difference with
       a local relaxation sweep does not converge: every pass clips back into
       the tube and re-injects the kink the previous pass removed, so the
       fitted line still demanded 220-480% of full lock inside a tube that
       was provably feasible. The smoothing has to be GLOBAL.

       The loop is closed and uniformly sampled, so the second-difference
       operator is circulant and its smoothing inverse is exactly diagonal in
       the Fourier basis - one rfft, one multiply, one irfft, applied to the
       whole lap at once. Alternating that with a clip into the tube
       converges in a few dozen iterations.

       lambda is then searched, smallest first, so the line is only as smooth
       as it has to be: over-smoothing would push it off the centre of the
       tube for no reason.
    """
    n = path.n
    ds2 = path.ds * path.ds
    cap = path.k_budget * ds2

    mid = 0.5 * (lo + hi)
    width = hi - lo
    w = np.clip((TIGHT_MM - width) / TIGHT_MM, 0.0, 1.0)
    tgt = (1.0 - w) * np.clip(0.0, lo, hi) + w * mid

    freq = 2.0 * np.pi * np.fft.rfftfreq(n, d=1.0)
    d2hat = 2.0 * np.cos(freq) - 2.0                 # eigenvalues of D2
    d2sq = d2hat * d2hat

    best = np.clip(tgt, lo, hi)
    for lam in (3.0, 30.0, 300.0, 3.0e3, 3.0e4, 3.0e5, 3.0e6):
        H = 1.0 / (1.0 + lam * d2sq)
        lat = tgt.copy()
        for _ in range(iters):
            lat = np.fft.irfft(np.fft.rfft(lat) * H, n)
            np.clip(lat, lo, hi, out=lat)
        best = lat
        d2 = np.roll(lat, 1) - 2.0 * lat + np.roll(lat, -1)
        if np.all(np.abs(d2) <= cap * 1.25):
            break
    return best


def line_curvature(path, lat):
    """Peak |curvature| of the profile, as a fraction of full lock."""
    _, kap, _ = planned_line(path, lat)
    return float(np.abs(kap).max() / K_CAR)


def planned_line(path: Path, lat):
    """(points, curvature) of the line the car should actually drive.

    Curvature is measured from the resulting points rather than assumed, so
    it already includes both the reference arc and the lateral profile's own
    bending. The tracker feeds it forward, which is what removes pure
    pursuit's standing error on the corners: at a 500 mm radius with a 280 mm
    lookahead that error is L^2/(2R) = 78 mm, and 78 mm is the whole passing
    margin.
    """
    px = path.xy[:, 0] + path.nx * lat
    py = path.xy[:, 1] + path.ny * lat
    x0, x1, x2 = np.roll(px, 1), px, np.roll(px, -1)
    y0, y1, y2 = np.roll(py, 1), py, np.roll(py, -1)
    a = np.hypot(x1 - x0, y1 - y0)
    b = np.hypot(x2 - x1, y2 - y1)
    c = np.hypot(x2 - x0, y2 - y0)
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    denom = a * b * c
    kappa = np.where(denom > 1e-6, 2.0 * area2 / np.maximum(denom, 1e-6), 0.0)
    th = np.arctan2(np.roll(py, -1) - np.roll(py, 1),
                    np.roll(px, -1) - np.roll(px, 1))
    return np.stack([px, py], axis=1), kappa, th


# ------------------------------------------------------------ the tracker
def pure_pursuit(path: Path, lat, x, y, th, wheelbase,
                 lookahead_mm: float = 260.0):
    """Plain pure pursuit. Kept for comparison; track() is what runs."""
    i, _ = path.project(x, y)
    k = max(1, int(round(lookahead_mm / path.ds)))
    j = (i + k) % path.n
    tx, ty = path.offset_point(j, lat[j])
    dx, dy = tx - x, ty - y
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return 0.0
    alpha = math.atan2(dy, dx) - th
    alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
    return math.atan2(2.0 * wheelbase * math.sin(alpha), L)


def track(path: Path, pts, kappa, line_th, x, y, th, wheelbase,
          k_head: float = 0.9, k_cross: float = 0.0035,
          preview_mm: float = 140.0, hint=None):
    """Curvature feedforward + heading and cross-track feedback.

    Three terms, each doing one job:

      feedforward  atan(L * kappa) is the steer angle that holds the planned
                   curvature in steady state. Pure pursuit has to build a
                   standing cross-track error before it generates this, which
                   is precisely the L^2/2R error that was eating the margin.
      heading      corrects the angle between the car and the line.
      cross-track  corrects the remaining offset from the line.

    Preview: the feedforward is taken a little ahead of the car rather than
    at it, so the steering is already turning as the line starts to bend
    instead of a control period after it.
    """
    i = _nearest(pts, x, y, hint)
    n = len(pts)
    j = (i + max(1, int(round(preview_mm / path.ds)))) % n

    ex = x - pts[i, 0]
    ey = y - pts[i, 1]
    lt = line_th[i]
    cross = -math.sin(lt) * ex + math.cos(lt) * ey      # + = car left of line
    hdg = (lt - th + math.pi) % (2 * math.pi) - math.pi

    ff = math.atan(wheelbase * kappa[j])
    return ff + k_head * hdg - math.atan(k_cross * cross), i, cross


def _nearest(pts, x, y, hint=None):
    """Nearest station on the planned line.

    The caller passes the previous index. Keying a cache on the array's
    identity does not work here: every replan builds a NEW array, the cache
    misses, and the search falls back to a global argmin that can land on the
    opposite side of the mat wherever the loop passes close to itself.
    """
    n = len(pts)
    if hint is None:
        d = pts - np.array([x, y])
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))
    idx = (hint + np.arange(-30, 31)) % n
    d = pts[idx] - np.array([x, y])
    return int(idx[np.argmin(np.einsum("ij,ij->i", d, d))])
