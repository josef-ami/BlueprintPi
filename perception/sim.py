"""
perception/sim.py - a ground-truth world and a synthetic RPLidar, so the whole
perception pipeline can be exercised and regression-tested with no hardware.

The synthetic scan is built with nav.geom.raycast against the SAME wall
segments the localizer subtracts, plus the segments of whatever pillars and
parking blocks the scenario placed. Feed the result into localize / pillarmap /
perception.state exactly as a real scan, and you have a closed loop where the
true pose and true pillar colours are known, so accuracy is measurable.

Scan convention (matches worldstate.py / localize.py):
    ranges is a 360-long array, index = integer degree in the SENSOR frame,
    0 = sensor forward, increasing anticlockwise. World angle of ray i at
    heading th is radians(i) + th. inf = no return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nav.geom import WALL_SEGMENTS, raycast
from nav import arena
from nav.pillarmap import RED, GREEN, UNKNOWN

# RPLidar C1-ish defaults
MAX_RANGE_MM = 5000.0
RANGE_SIGMA_MM = 12.0
DROPOUT_P = 0.04
ANGLE_JITTER_DEG = 0.0        # kept 0: index maps to integer degree bins


@dataclass
class Scenario:
    direction: str = "CCW"            # or "CW"; free for perception, used by planner later
    pillars: dict = field(default_factory=dict)   # seat_id -> RED|GREEN
    parking_side: str = "S"
    parking_along: float = 0.0
    car_len_mm: float = 175.0

    def pillar_list(self):
        """[(x, y, colour, seat_id)] for the occupied seats."""
        out = []
        by_id = {s.id: s for s in arena.SEATS}
        for sid, col in self.pillars.items():
            s = by_id[sid]
            out.append((s.x, s.y, col, sid))
        return out

    def parking(self):
        return arena.parking_bay(self.parking_side, self.parking_along, self.car_len_mm)


def scenario_segments(sc: Scenario) -> np.ndarray:
    """All ray-castable segments: walls + pillars + the two parking blocks."""
    segs = [WALL_SEGMENTS]
    for (x, y, _c, _id) in sc.pillar_list():
        segs.append(arena.pillar_segments(x, y))
    bay = sc.parking()
    segs.append(bay.left_block)
    segs.append(bay.right_block)
    return np.vstack(segs)


def synth_scan(pose, segs, rng=None, max_range=MAX_RANGE_MM,
               range_sigma=RANGE_SIGMA_MM, dropout_p=DROPOUT_P):
    """(ranges[360], quals[360]) seen from `pose`=(x, y, th) against `segs`."""
    if rng is None:
        rng = np.random.default_rng()
    x0, y0, th = pose
    idx = np.arange(360)
    ang = np.radians(idx.astype(np.float64)) + th
    d = raycast(x0, y0, ang, segs, max_range)          # inf where no hit
    ranges = np.full(360, np.inf)
    quals = np.zeros(360, dtype=np.int32)
    hit = np.isfinite(d) & (d < max_range)
    noisy = d[hit] + rng.normal(0.0, range_sigma, hit.sum())
    ranges[hit] = np.clip(noisy, 1.0, max_range)
    quals[hit] = 47
    # random dropouts
    drop = rng.random(360) < dropout_p
    ranges[drop] = np.inf
    quals[drop] = 0
    return ranges, quals


def random_scenario(seed=None, n_pillars=6) -> Scenario:
    """A rules-plausible layout: a subset of seats occupied, random colours,
    one parking bay in a straight."""
    rng = np.random.default_rng(seed)
    seat_ids = [s.id for s in arena.SEATS]
    k = min(n_pillars, len(seat_ids))
    chosen = rng.choice(len(seat_ids), size=k, replace=False)
    pillars = {seat_ids[i]: int(rng.choice([RED, GREEN])) for i in chosen}
    side = str(rng.choice(list(arena.SIDES)))
    along = float(rng.uniform(-200, 200))
    return Scenario(direction=str(rng.choice(["CW", "CCW"])),
                    pillars=pillars, parking_side=side, parking_along=along)


def corridor_waypoints(step_mm: float = 60.0, turn_r: float = 300.0):
    """A lap of poses round the corridor centre-line, with ROUNDED corners so
    the heading changes gradually (<~8 deg/step). That matters because the
    pose is estimated from LiDAR alone here (no IMU): the scan matcher can only
    follow a heading change that stays inside its per-frame search window, and
    a real car turns on an arc anyway.

    Returns list of (x, y, th).
    """
    cc = arena.CORRIDOR_CENTER
    corners = [(-cc, -cc), (cc, -cc), (cc, cc), (-cc, cc)]   # CCW square
    pts = []
    for i in range(4):
        ax, ay = corners[i]
        bx, by = corners[(i + 1) % 4]
        cx, cy = corners[(i + 2) % 4]
        th_in = math.atan2(by - ay, bx - ax)
        th_out = math.atan2(cy - by, cx - bx)
        # straight part of side i, stopping turn_r short of corner b
        seg = math.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / seg, (by - ay) / seg
        straight = seg - turn_r
        n = max(1, int(straight / step_mm))
        for j in range(n):
            d = j * step_mm
            pts.append((ax + ux * d, ay + uy * d, th_in))
        # rounded corner: pivot heading from th_in to th_out over an arc
        vx, vy = (cx - bx) / math.hypot(cx - bx, cy - by), (cy - by) / math.hypot(cx - bx, cy - by)
        p_in = (bx - ux * turn_r, by - uy * turn_r)
        p_out = (bx + vx * turn_r, by + vy * turn_r)
        k = 14
        dth = ((th_out - th_in + math.pi) % (2 * math.pi) - math.pi) / k
        for j in range(k):
            t = j / k
            x = p_in[0] + t * (p_out[0] - p_in[0])
            y = p_in[1] + t * (p_out[1] - p_in[1])
            pts.append((x, y, th_in + dth * j))
    return pts


if __name__ == "__main__":
    sc = random_scenario(seed=1)
    segs = scenario_segments(sc)
    print(f"scenario: {len(sc.pillars)} pillars, parking {sc.parking_side}@{sc.parking_along:.0f}")
    r, q = synth_scan((-1000.0, -1000.0, 0.0), segs, rng=np.random.default_rng(0))
    finite = np.isfinite(r)
    print(f"scan: {finite.sum()}/360 returns, min={np.nanmin(r[finite]):.0f} "
          f"max={np.nanmax(r[finite]):.0f} mm")
