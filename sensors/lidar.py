"""
LiDAR adapter - distance, wall geometry, and objects nothing has named yet.

The LiDAR does four jobs, in rising order of cleverness:

  1. three beams      0 / 90 / 270 deg, for the corner trigger and the front
                      panic. read_three().
  2. wall cones       a line fitted to every return in a 45 deg cone each
                      side. Gives the PERPENDICULAR distance to each wall
                      (which does not grow when the car is yawed, the way a
                      single beam does) and the car's yaw relative to the
                      walls. fit_wall() / cones().
  3. pillar range     the camera says which direction a pillar is in; this
                      says how far. locate_pillar().
  4. candidates       small clusters inside the corridor that the camera has
                      not named. The camera only covers about +/-48 deg, so a
                      pillar near the far wall can stay out of view until it
                      is too late; the LiDAR sees it from the start line.
                      lidar_candidates().

Liveness / freshness:
  - self.rev           completed revolutions, +1 each time the raw angle wraps
                       360 -> 0. Each bearing bin gets at most one new sample
                       per revolution, so consumers debounce per rev, not per
                       frame.
  - self.last_point_t  time.monotonic() of the last point received, valid or
                       not. If it stops advancing, the scan has stalled even
                       though the bins still hold old values.

Everything below takes its thresholds as arguments rather than reading
params.py, so it can be tested with no robot, no config and no camera.
"""

import asyncio
import json
import math
import os
import threading
import time

import numpy as np

from worldstate import LidarResult, SharedState, WallFit

PUBLISH_PERIOD = 0.05   # 20 Hz SharedState snapshot; the control loop reads
                        # the live bins directly and skips this hop

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config.json")

INVALID = 0xFFFF        # "no return" on the wire
CONE_NONE = 65535
ANG_NONE = 32767
PXY_NONE = 32767

BEARINGS = (("front", 0), ("left", 90), ("right", 270))   # robot frame, CCW+

_DEG = np.radians(np.arange(360))
_COS, _SIN = np.cos(_DEG), np.sin(_DEG)


def load_lidar_config(path=CONFIG_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f).get("lidar", {})
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------
# thread
# --------------------------------------------------------------------------

class LidarThread(threading.Thread):
    def __init__(self, shared: SharedState):
        super().__init__(name="LidarThread", daemon=True)
        self.shared = shared
        self._lidar = None
        self._loop = None
        self._async_stop = None
        self.cfg = load_lidar_config()
        self.port = self.cfg.get("port", "/dev/ttyUSB0")
        self.baud = self.cfg.get("baud", 460800)
        self.mount_offset = self.cfg.get("mount_offset_deg", 0)
        self._ranges = [float("inf")] * 360
        self._quals = [0] * 360

        # freshness / liveness, read from other threads (plain attribute
        # reads/writes are atomic under the GIL)
        self.rev = 0
        self.last_point_t = 0.0
        self._last_a = None

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except BaseException as eg:
            subs = getattr(eg, "exceptions", None)
            if subs:
                for s in subs:
                    print(f"[LidarThread] sub-exception: {type(s).__name__}: {s}")
            else:
                print(f"[LidarThread] fatal: {type(eg).__name__}: {eg}")
        finally:
            try:
                if self._lidar is not None:
                    self._lidar.reset()
            except Exception:
                pass
            self._loop.close()
            print("[LidarThread] stopped")

    async def _async_main(self):
        from rplidarc1 import RPLidar
        self._lidar = RPLidar(self.port, self.baud)
        self._async_stop = self._lidar.stop_event
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._consume(self._lidar.output_queue,
                                         self._async_stop))
            tg.create_task(self._publisher(self._async_stop))
            tg.create_task(self._lidar.simple_scan())

    async def _consume(self, queue, stop_event):
        while not stop_event.is_set():
            try:
                point = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            a = point["a_deg"]
            if self._last_a is not None and a < self._last_a - 180.0:
                self.rev += 1                  # raw angle wrapped: sweep done
            self._last_a = a
            self.last_point_t = time.monotonic()

            d = point["d_mm"]
            idx = int(round(a) + self.mount_offset) % 360
            self._ranges[idx] = float("inf") if d is None else float(d)
            self._quals[idx] = point["q"]

    async def _publisher(self, stop_event):
        while not stop_event.is_set():
            self.shared.set_lidar(LidarResult(
                timestamp=time.time(), ranges=list(self._ranges),
                qualities=list(self._quals), rev=self.rev, ok=True))
            await asyncio.sleep(PUBLISH_PERIOD)

    def queue_depth(self):
        """Points waiting in rplidarc1's queue; should hover near 0. A growing
        number means _consume can't keep up and lag grows over time."""
        try:
            return self._lidar.output_queue.qsize() if self._lidar else -1
        except Exception:
            return -1

    def stop(self):
        if self._loop is not None and self._async_stop is not None:
            self._loop.call_soon_threadsafe(self._async_stop.set)


def lidar_live(lidar, now, stale_s=0.3):
    return lidar.is_alive() and (now - lidar.last_point_t) < stale_s


# --------------------------------------------------------------------------
# 1. beams
# --------------------------------------------------------------------------

def sector_min(ranges, center_deg, half_width_deg):
    """Smallest distance within +/- half_width of center_deg. inf if clear."""
    best = float("inf")
    for offset in range(-half_width_deg, half_width_deg + 1):
        best = min(best, ranges[(center_deg + offset) % 360])
    return best


def pick_bearing(ranges, quals, center_deg, half_width_deg):
    """Highest-quality non-inf return within +/- half_width_deg of center_deg.

    Returns (deg, dist_mm, quality) of the winner, or None if every bin in the
    window is inf. Ties in quality break toward the bin nearest center_deg
    (offsets are visited in order of increasing |offset|, and only a strictly
    higher quality wins).
    """
    center = int(round(center_deg)) % 360
    best = None
    for offset in sorted(range(-half_width_deg, half_width_deg + 1), key=abs):
        deg = (center + offset) % 360
        d = ranges[deg]
        if math.isinf(d):
            continue
        q = quals[deg]
        if best is None or q > best[2]:
            best = (deg, d, q)
    return best


def read_three(ranges, quals, tol):
    """(front_mm, left_mm, right_mm) - each mm or None (no valid return)."""
    out = []
    for _, target in BEARINGS:
        p = pick_bearing(ranges, quals, target, tol)
        out.append(None if p is None else p[1])
    return tuple(out)


def read_three_picks(ranges, quals, tol):
    """read_three, keeping each whole pick_bearing result (deg, mm, quality)."""
    return tuple(pick_bearing(ranges, quals, target, tol)
                 for _, target in BEARINGS)


def u16(mm):
    """mm -> the u16 the wire carries. None / inf / out of range -> INVALID."""
    if mm is None or (isinstance(mm, float) and math.isinf(mm)):
        return INVALID
    v = int(round(mm))
    return INVALID if not (0 <= v < INVALID) else v


# --------------------------------------------------------------------------
# 2. wall cones
# --------------------------------------------------------------------------
#
# Instead of one beam at 90 / 270 deg, take EVERY return in a 45 deg cone
# centred on each side (67.5-112.5 deg left, 247.5-292.5 deg right), convert
# to x (forward) / y (left) and fit a straight line to the wall:
#
#   distance  = perpendicular distance to that line  (doesn't grow when yawed)
#   yaw       = -atan(slope), + = car pointing LEFT of the wall direction
#
# The fit is RANSAC over every point pair (vectorised, ~1k pairs): the line
# with the most points within inlier_mm wins, ties go to the FARTHER line, so
# a pillar between the car and the wall is rejected as outliers instead of
# pulling the fit. Then a least-squares refit on the inliers. A fit needs
# min_inliers points spread over min_span_mm along the wall - a 50 mm pillar
# face cannot pass that on its own.

def fit_wall(ranges, centre_deg, cone_deg=45, min_range_mm=60,
             max_range_mm=1500, inlier_mm=25, min_inliers=8, min_span_mm=150):
    """(perp_mm, yaw_deg, n_inliers, m, c) for the wall in the cone, or None."""
    half = int(cone_deg) // 2
    idx = np.arange(centre_deg - half, centre_deg + half + 1) % 360
    d = np.asarray([ranges[i] for i in idx], dtype=np.float64)
    ok = np.isfinite(d) & (d > min_range_mm) & (d < max_range_mm)
    if ok.sum() < min_inliers:
        return None
    b = np.radians(idx[ok])
    x, y = d[ok] * np.cos(b), d[ok] * np.sin(b)

    i, j = np.triu_indices(len(x), 1)
    keep = np.abs(x[j] - x[i]) > 40.0            # well-separated pairs only
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return None
    m = (y[j] - y[i]) / (x[j] - x[i])
    c = y[i] - m * x[i]
    res = np.abs(y[None, :] - m[:, None] * x[None, :] - c[:, None]) \
        / np.sqrt(1.0 + m[:, None] ** 2)
    inl = res < inlier_mm
    cnt = inl.sum(axis=1)
    far = np.abs(c) / np.sqrt(1.0 + m * m)
    best = np.lexsort((far, cnt))[-1]            # most inliers, then farthest
    mask = inl[best]
    if mask.sum() < min_inliers:
        return None
    xs, ys = x[mask], y[mask]
    wall_d = float(np.median(np.abs(ys)))
    need = min(min_span_mm,
               0.6 * 2 * wall_d * math.tan(math.radians(cone_deg / 2)))
    if xs.max() - xs.min() < need:
        return None
    m, c = np.polyfit(xs, ys, 1)
    return (abs(c) / math.sqrt(1.0 + m * m),
            -math.degrees(math.atan(m)),
            int(mask.sum()),
            float(m), float(c))                  # the line y = m x + c


def cones(ranges, p):
    """WallFit from a PiParams-like mapping. Both walls, plus the car's yaw.

    With both walls fitted the yaw is the inlier-weighted mean of the two, but
    only when they agree to within CONE_AGREE_DEG. When they disagree - at a
    corner, or where the inner wall ends - there is no yaw rather than a bad
    one, because a wrong yaw rotates every pillar sighting.
    """
    kw = dict(cone_deg=p["CONE_DEG"], min_range_mm=p["CONE_MIN_RANGE_MM"],
              max_range_mm=p["CONE_MAX_RANGE_MM"], inlier_mm=p["CONE_INLIER_MM"],
              min_inliers=p["CONE_MIN_INLIERS"], min_span_mm=p["CONE_MIN_SPAN_MM"])
    L = fit_wall(ranges, 90, **kw)
    R = fit_wall(ranges, 270, **kw)
    yaw = None
    if L and R:
        if abs(L[1] - R[1]) <= p["CONE_AGREE_DEG"]:
            yaw = (L[1] * L[2] + R[1] * R[2]) / (L[2] + R[2])
    elif L:
        yaw = L[1]
    elif R:
        yaw = R[1]
    return WallFit(left_mm=L[0] if L else None, right_mm=R[0] if R else None,
                   yaw_deg=yaw,
                   left_line=(L[3], L[4]) if L else None,
                   right_line=(R[3], R[4]) if R else None)


# --------------------------------------------------------------------------
# 3. pillar position: camera bearing + LiDAR range along that ray
# --------------------------------------------------------------------------

def locate_pillar(ranges, bearing_deg, area, p):
    """(x, y) mm of the pillar CENTRE in the car frame, or None.

    Direction comes from the CAMERA (fresh, 30 Hz). Only the DISTANCE comes
    from the LiDAR, because a scan can be up to 100 ms old and while the car
    swerves at ~100 deg/s its bearings are ~10 deg stale - matching an exact
    ray then hits the wall behind the pillar. So: take every return within
    RAY_WINDOW_DEG of the camera ray (seen from the camera), keep those whose
    range agrees with the size estimate (AREA_K / sqrt(area)) within
    [0.5x, 2x], and use the nearest. No agreeing return -> the size estimate.

    The camera sits CAMERA_FWD_MM ahead of the LiDAR, so the ray starts there,
    not at the LiDAR - at 40 cm that parallax is several degrees.
    """
    if bearing_deg is None:
        return None
    cam_fwd = p["CAMERA_FWD_MM"]
    max_mm = p["PILLAR_MAX_MM"]
    b = math.radians(bearing_deg)
    r = np.asarray(ranges, dtype=np.float64)
    ok = np.isfinite(r) & (r > 60) & (r < max_mm + 300)
    # inf * cos(90 deg) is a NaN, and a NaN that survives into arctan2 below
    # would quietly poison the angle test. Zero the misses first; `ok` already
    # excludes them from every decision.
    rs = np.where(ok, r, 0.0)
    px, py = rs * _COS - cam_fwd, rs * _SIN          # relative to the camera
    dist_c = np.hypot(px, py)
    ang = np.degrees(np.arctan2(py, px)) - bearing_deg
    ang = (ang + 180.0) % 360.0 - 180.0
    d_area = p["AREA_K"] / math.sqrt(area) if area and area > 0 else None
    cand = ok & (np.abs(ang) <= p["RAY_WINDOW_DEG"]) & (dist_c > 30)
    if d_area is not None:
        cand &= (dist_c > 0.5 * d_area) & (dist_c < 2.0 * d_area)
    if cand.any():
        dist = float(dist_c[cand].min()) + p["FACE_TO_CENTRE_MM"]
    elif d_area is not None:
        dist = d_area
    else:
        return None
    if dist > max_mm:
        return None
    return (cam_fwd + dist * math.cos(b), dist * math.sin(b))


# --------------------------------------------------------------------------
# 4. LiDAR candidates - objects inside the corridor, colour unknown
# --------------------------------------------------------------------------
#
# A candidate is a small cluster of returns that is:
#   * ahead of the car (x > 0) and closer than CAND_MAX_MM
#   * inside the corridor: more than CAND_WALL_MM from both fitted wall lines
#     (a missing side is placed CORRIDOR_MM from the other)
#   * short of the wall ahead (front distance - CAND_WALL_MM)
#   * no wider than CAND_MAX_WIDTH_MM (a pillar is 50 mm, 71 mm diagonal)
# The nearest one is what the planner lines up with, so the camera can name
# the colour; if it never does, the planner dodges to the roomier side.

def lidar_candidates(ranges, wall: WallFit, p):
    """[(x, y)] pillar-centre candidates in the car frame, nearest first.

    Cluster FIRST, filter second: a wall is one long run of returns and is
    thrown out whole by the width test. Filtering first would chop a wall into
    short fragments at the cut lines, and those look exactly like pillars.
    """
    left_line, right_line = wall.left_line, wall.right_line
    if left_line is None and right_line is None:
        return []                                # corner: no corridor to search
    gap = p["CAND_GAP_MM"]
    wall_mm = p["CAND_WALL_MM"]
    max_mm = p["CAND_MAX_MM"]
    corridor = p["CORRIDOR_MM"]
    face = p["FACE_TO_CENTRE_MM"]

    r = np.asarray(ranges, dtype=np.float64)
    ok = np.isfinite(r) & (r > 80) & (r < max_mm + 400)
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return []
    x, y = r[idx] * _COS[idx], r[idx] * _SIN[idx]
    fwd = r[(np.arange(-3, 4)) % 360]
    fwd = fwd[np.isfinite(fwd)]
    front = float(np.median(fwd)) if fwd.size else np.inf

    clusters, cur = [], [0]
    for k in range(1, idx.size):
        if idx[k] - idx[cur[-1]] <= 3 and \
                math.hypot(x[k] - x[cur[-1]], y[k] - y[cur[-1]]) < gap:
            cur.append(k)
        else:
            clusters.append(cur)
            cur = [k]
    clusters.append(cur)
    if len(clusters) > 1 and idx[0] + 360 - idx[-1] <= 3 and \
            math.hypot(x[0] - x[-1], y[0] - y[-1]) < gap:    # wrap at 0/359
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    out = []
    for c in clusters:
        if len(c) < p["CAND_MIN_POINTS"]:
            continue
        cx, cy = x[c], y[c]
        if math.hypot(cx.max() - cx.min(), cy.max() - cy.min()) > p["CAND_MAX_WIDTH_MM"]:
            continue                                  # wall run
        mx, my = float(cx.mean()), float(cy.mean())
        d = math.hypot(mx, my)
        if d <= 0 or mx <= 0 or d > max_mm or mx > front - wall_mm:
            continue
        if left_line is not None:
            if my > left_line[0] * mx + left_line[1] - wall_mm:
                continue
        elif my > right_line[0] * mx + right_line[1] + corridor - wall_mm:
            continue
        if right_line is not None:
            if my < right_line[0] * mx + right_line[1] + wall_mm:
                continue
        elif my < left_line[0] * mx + left_line[1] - corridor + wall_mm:
            continue
        out.append((mx + face * mx / d, my + face * my / d))   # face -> centre
    out.sort(key=lambda q: math.hypot(*q))
    return out


def unclassified(cands, named_xy, match_mm=200.0):
    """Nearest candidate that is NOT the pillar the camera already named."""
    for c in cands:
        if named_xy is None or \
                math.hypot(c[0] - named_xy[0], c[1] - named_xy[1]) > match_mm:
            return c
    return None
