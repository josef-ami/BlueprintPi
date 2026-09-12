"""
Lidar adapter — now the primary obstacle detector.

Pipeline inside the lidar thread:
  rplidarc1 stream  ->  rolling 360 range+quality arrays
                    ->  filter (quality floor + isolation removal)
                    ->  cluster survivors into discrete objects
                    ->  publish LidarResult{ranges, obstacles}

All filter/cluster parameters come from config.json ("lidar" block) and are
read once at startup, so the calibration dashboard can tune them and a
restart applies them — same contract as the camera.

The camera no longer supplies geometry; it only colours these obstacles
later, in fusion.
"""

import asyncio
import json
import math
import os
import threading
import time

from rplidarc1 import RPLidar
from worldstate import SharedState, LidarResult, LidarObstacle

PORT = "/dev/ttyUSB0"
BAUD = 460800
PUBLISH_PERIOD = 0.05   # 20 Hz obstacle publish

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config.json")


def load_lidar_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg.get("lidar", {})


# --------------------------------------------------------------------------
# filtering + clustering — pure functions, unit-testable without hardware
# --------------------------------------------------------------------------

def _polar_to_xy(deg, mm):
    a = math.radians(deg)
    return (mm * math.sin(a), mm * math.cos(a))


def filter_points(ranges, qualities, cfg):
    """
    Two-stage filter (mirrors the probe tool):
      1. drop points below quality_floor or beyond max_range
      2. drop isolated survivors (fewer than min_neighbours within
         neighbour_dist_mm, measured in real x/y space)
    Returns a list of surviving points: [(deg, mm, q, x, y), ...]
    """
    q_floor = cfg.get("quality_floor", 10)
    max_range = cfg.get("max_range_mm", 3000)
    min_n = cfg.get("min_neighbours", 2)
    tol = cfg.get("neighbour_dist_mm", 150)

    survivors = []
    for deg in range(360):
        mm = ranges[deg]
        if mm is None or math.isinf(mm) or mm <= 0 or mm > max_range:
            continue
        if qualities[deg] < q_floor:
            continue
        x, y = _polar_to_xy(deg, mm)
        survivors.append((deg, mm, qualities[deg], x, y))

    if min_n <= 0:
        return survivors

    kept = []
    for p in survivors:
        n = 0
        for o in survivors:
            if o is p:
                continue
            if math.hypot(p[3] - o[3], p[4] - o[4]) <= tol:
                n += 1
                if n >= min_n:
                    break
        if n >= min_n:
            kept.append(p)
    return kept


def cluster_points(points, cfg):
    """
    Group filtered points into discrete obstacles by walking them in angular
    order and starting a new cluster whenever the gap to the previous point
    (in real x/y space) exceeds cluster_gap_mm. Wraps around 0/360.
    Returns a list of LidarObstacle.
    """
    gap = cfg.get("cluster_gap_mm", 120)
    min_pts = cfg.get("min_cluster_points", 3)
    if not points:
        return []

    pts = sorted(points, key=lambda p: p[0])   # by degree

    clusters = [[pts[0]]]
    for prev, cur in zip(pts, pts[1:]):
        if math.hypot(cur[3] - prev[3], cur[4] - prev[4]) <= gap:
            clusters[-1].append(cur)
        else:
            clusters.append([cur])

    # stitch wrap-around: if first and last clusters touch across 0 deg
    if len(clusters) > 1:
        a, b = clusters[0][0], clusters[-1][-1]
        if math.hypot(a[3] - b[3], a[4] - b[4]) <= gap:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()

    obstacles = []
    for c in clusters:
        if len(c) < min_pts:
            continue
        degs = [p[0] for p in c]
        mms = [p[1] for p in c]
        # bearing: circular mean so a cluster spanning 359..1 deg averages sanely
        sin_s = sum(math.sin(math.radians(d)) for d in degs)
        cos_s = sum(math.cos(math.radians(d)) for d in degs)
        bearing = math.degrees(math.atan2(sin_s, cos_s))
        # width: angular span, wrap-aware
        span = _angular_span(degs)
        obstacles.append(LidarObstacle(
            bearing_deg=bearing,
            distance_mm=min(mms),          # nearest point = what we'd hit
            width_deg=span,
            point_count=len(c),
        ))
    return obstacles


def _angular_span(degs):
    """Smallest arc covering all degrees, handling wrap-around."""
    s = sorted(d % 360 for d in degs)
    if len(s) < 2:
        return 0.0
    gaps = [(s[i + 1] - s[i]) for i in range(len(s) - 1)]
    gaps.append(360 - s[-1] + s[0])       # wrap gap
    return 360 - max(gaps)


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
        self.mount_offset = self.cfg.get("mount_offset_deg", 0)
        self._ranges = [float("inf")] * 360
        self._quals = [0] * 360

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
        self._lidar = RPLidar(PORT, BAUD)
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
            d = point["d_mm"]
            idx = int(round(point["a_deg"]) + self.mount_offset) % 360
            self._ranges[idx] = float("inf") if d is None else float(d)
            self._quals[idx] = point["q"]

    async def _publisher(self, stop_event):
        while not stop_event.is_set():
            ranges = list(self._ranges)
            quals = list(self._quals)
            survivors = filter_points(ranges, quals, self.cfg)
            obstacles = cluster_points(survivors, self.cfg)
            self.shared.set_lidar(LidarResult(
                timestamp=time.time(),
                ranges=ranges,
                qualities=quals,
                obstacles=obstacles,
                ok=True,
            ))
            await asyncio.sleep(PUBLISH_PERIOD)

    def stop(self):
        if self._loop is not None and self._async_stop is not None:
            self._loop.call_soon_threadsafe(self._async_stop.set)


def sector_min(ranges, center_deg, half_width_deg):
    """Kept for the dashboard's front-distance readout."""
    best = float("inf")
    for offset in range(-half_width_deg, half_width_deg + 1):
        best = min(best, ranges[(center_deg + offset) % 360])
    return best
