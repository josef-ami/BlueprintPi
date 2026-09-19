"""
solver.py — pillar avoidance geometry. Pure functions, no hardware, no state.

THE MODEL
---------
A pillar at bearing b and range r defines a keep-out cone of half-angle

    delta = asin(d_clear / r_axis)

where d_clear is how far the PILLAR AXIS must be from the car's centreline for
the car to slide past it untouched. Aim delta to one side of the pillar and you
are driving the tangent to that keep-out circle: the path grazes it at exactly
d_clear and nowhere closer.

    theta = b + s * delta        s = +1 GREEN (pass on its left), -1 RED (right)
    leg   = sqrt(r_axis^2 - d_clear^2)      distance to the tangency point

Two things fall out of the same delta:

  * "already clear" is |b| >= delta on the correct side. No second test needed —
    if the pillar is further off-axis than the cone is wide, the straight path
    already misses it by more than d_clear.
  * theta is the MINIMUM deviation. If the pillar is already nearly clear, theta
    is small; if it is dead ahead, theta is the full delta. The solver never
    asks for more steering than the geometry needs.

WHY TRACK THEN COMMIT
---------------------
The tangent solve assumes the heading changes instantly. It does not: turning in
by delta costs about R*delta of forward travel (R = 194.4 mm), which is 8% of the
leg at r = 700 mm but 45% at r = 250 mm. So re-solve every tick while the pillar
is far (TRACK) and the error corrects itself, and only freeze the answer once it
is close or has left the camera's field of view (COMMIT), where the leg runs out
on odometry alone.

CONVENTIONS (identical to worldstate.py)
    distances mm, angles deg, bearings 0 = forward, POSITIVE = LEFT.
"""

import math
from dataclasses import dataclass

# Action codes. These are the wire values in the PERCEPT frame's flags byte —
# percept_link.py and ObstacleLap.cpp both depend on them.
AVOID_NONE = 0     # nothing to do; hold the lane heading
AVOID_TRACK = 1    # solution is live and being refreshed every tick
AVOID_COMMIT = 2   # solution is frozen; run the leg out on odometry

SIDE_LEFT = +1     # pass on the pillar's left  -> GREEN
SIDE_RIGHT = -1    # pass on the pillar's right -> RED

COLOR_SIDE = {"GREEN": SIDE_LEFT, "RED": SIDE_RIGHT}


@dataclass
class AvoidCfg:
    """Loaded from config.json -> "avoid". Millimetres and degrees."""
    car_half_width_mm: float = 57.1      # SPECSHEET s3: 114.24 mm scored width
    pillar_half_width_mm: float = 25.0   # 50 mm pillar
    margin_mm: float = 40.0              # the only number you actually tune
    engage_mm: float = 900.0             # start solving at this range
    freeze_mm: float = 350.0             # stop re-solving, commit the answer
    tail_clear_mm: float = 150.0         # keep going past tangency so the tail clears
    max_heading_deg: float = 45.0        # refuse to command more than this off-lane
    max_bearing_deg: float = 50.0        # pillars wider than this are not ours yet
    min_range_mm: float = 150.0          # below this the lidar return is the chassis
    wall_guard: bool = True
    wall_margin_mm: float = 90.0         # closest the car's flank may plan to come

    @property
    def clearance_mm(self) -> float:
        """d_clear: pillar-axis to car-centreline, the number the whole solve turns on."""
        return self.car_half_width_mm + self.pillar_half_width_mm + self.margin_mm

    @classmethod
    def from_config(cls, cfg: dict) -> "AvoidCfg":
        return cls(**{k: v for k, v in cfg.get("avoid", {}).items()
                      if k in cls.__dataclass_fields__})


@dataclass
class Solution:
    action: int = AVOID_NONE
    theta_deg: float = 0.0       # heading offset from CURRENT heading, + = left
    leg_mm: float = 0.0          # how far to travel on that heading
    delta_deg: float = 0.0       # the keep-out half-angle, for logging
    reason: str = ""


def solve(color: str, bearing_deg: float, range_mm: float, cfg: AvoidCfg,
          side_free_mm: float = float("inf")) -> Solution:
    """
    One pillar in, one manoeuvre out.

    range_mm    lidar range at the pillar's bearing. This is the distance to its
                near FACE, so the axis is half a pillar further away.
    side_free_mm  lidar range at +-90 deg on the side we would move toward, used
                by the wall guard. Pass inf to disable it for one call.
    """
    side = COLOR_SIDE.get(color)
    if side is None:
        return Solution(reason=f"colour {color!r} is not a pillar")
    if not math.isfinite(range_mm) or range_mm < cfg.min_range_mm:
        return Solution(reason="no usable range")
    if abs(bearing_deg) > cfg.max_bearing_deg:
        return Solution(reason=f"bearing {bearing_deg:+.0f} outside the working cone")

    r_axis = range_mm + cfg.pillar_half_width_mm
    if r_axis > cfg.engage_mm:
        return Solution(reason=f"r={r_axis:.0f} beyond engage range")

    d = cfg.clearance_mm
    # Inside the keep-out circle the arcsin has no solution. Saturate rather than
    # raise: this only happens if we engaged far too late, and the honest answer
    # is "turn as hard as the guard allows", not a crash.
    ratio = min(d / r_axis, 0.99)
    delta = math.degrees(math.asin(ratio))

    # Already clear: the pillar sits further off-axis than the cone is wide, on
    # the side it is legally allowed to be. RED must end up LEFT of us, GREEN RIGHT.
    if side == SIDE_RIGHT and bearing_deg >= delta:
        return Solution(delta_deg=delta, reason="already clear (pillar left)")
    if side == SIDE_LEFT and bearing_deg <= -delta:
        return Solution(delta_deg=delta, reason="already clear (pillar right)")

    theta = bearing_deg + side * delta
    leg = math.sqrt(max(r_axis * r_axis - d * d, 0.0)) + cfg.tail_clear_mm

    if cfg.wall_guard:
        theta = _wall_guard(theta, leg, side_free_mm, cfg)
        if theta is None:
            return Solution(delta_deg=delta, reason="no room on the legal side")

    theta = max(-cfg.max_heading_deg, min(cfg.max_heading_deg, theta))
    action = AVOID_COMMIT if r_axis <= cfg.freeze_mm else AVOID_TRACK
    return Solution(action=action, theta_deg=theta, leg_mm=leg, delta_deg=delta,
                    reason=f"{color} r={r_axis:.0f} b={bearing_deg:+.0f} "
                           f"d={delta:.1f} -> {theta:+.1f} for {leg:.0f}")


def _wall_guard(theta_deg: float, leg_mm: float, side_free_mm: float,
                cfg: AvoidCfg):
    """
    Clip theta so the planned lateral excursion cannot plant the car's flank in
    the wall. Returns a (possibly reduced) theta, or None if even the reduced
    path has no room — in which case holding the lane and eating the penalty
    beats a collision that ends the run.
    """
    if not math.isfinite(side_free_mm) or abs(theta_deg) < 0.1:
        return theta_deg
    budget = side_free_mm - cfg.car_half_width_mm - cfg.wall_margin_mm
    if budget <= 0.0:
        return None
    shift = leg_mm * math.sin(math.radians(abs(theta_deg)))
    if shift <= budget:
        return theta_deg
    return math.copysign(math.degrees(math.asin(min(budget / leg_mm, 0.99))),
                         theta_deg)


def wrap180(a: float) -> float:
    """Normalise to (-180, +180]. Use for every heading subtraction."""
    a = (a + 180.0) % 360.0 - 180.0
    return a + 360.0 if a <= -180.0 else a
