"""
Lidar adapter — camera-first: supplies DISTANCE only.

The lidar maintains a rolling 360-degree range array (plus per-angle quality
for the dashboard). It no longer detects or clusters obstacles — that's the
camera's job now. Fusion looks up a distance at the camera obstacle's bearing
via select_range.

Liveness / freshness (used by openRound.py):
  - self.rev           completed revolutions, +1 each time the raw angle wraps
                       360 -> 0. Each bearing bin gets at most one new sample
                       per revolution, so consumers debounce per rev, not per
                       send.
  - self.last_point_t  time.monotonic() of the last point received, valid or
                       not. If it stops advancing, the scan has stalled even
                       though the bins still hold old values.
"""

import asyncio
import json
import math
import os
import threading
import time

from rplidarc1 import RPLidar
from worldstate import SharedState, LidarResult

PORT = "/dev/ttyUSB0"
BAUD = 460800
PUBLISH_PERIOD = 0.05   # 20 Hz — SharedState snapshot for main.py / dashboard

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config.json")


def load_lidar_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg.get("lidar", {})


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
                timestamp=time.time(),
                ranges=list(self._ranges),
                qualities=list(self._quals),
                ok=True,
            ))
            await asyncio.sleep(PUBLISH_PERIOD)

    def queue_depth(self):
        """Points waiting in rplidarc1's queue; should hover near 0.
        A growing number means _consume can't keep up (lag grows over time)."""
        try:
            return self._lidar.output_queue.qsize() if self._lidar else -1
        except Exception:
            return -1

    def stop(self):
        if self._loop is not None and self._async_stop is not None:
            self._loop.call_soon_threadsafe(self._async_stop.set)


def sector_min(ranges, center_deg, half_width_deg):
    """Smallest distance within +/- half_width of center_deg. inf if all clear.
    Kept as-is for the dashboard's coarse front-distance readout."""
    best = float("inf")
    for offset in range(-half_width_deg, half_width_deg + 1):
        best = min(best, ranges[(center_deg + offset) % 360])
    return best


def select_range(ranges, center_deg, half_width_deg,
                 floor_mm=0.0, gap_split_mm=0.0):
    """
    Choose the distance to the object at center_deg, robust to the wall behind
    it and to spurious near-returns.

    Pipeline:
      1. gather valid returns in the +/- half_width window
      2. floor: drop anything closer than floor_mm (chassis-radius + margin) so
         a self-occlusion / stray near-return can't win the minimum
      3. gap split: sort survivors; if a jump >= gap_split_mm appears, that jump
         is the pillar->wall separation — keep only the near cluster below it
      4. return the minimum of the near cluster (the pillar is nearest here by
         construction), or inf if nothing valid remains

    With floor_mm=0 and gap_split_mm=0 this reduces to a plain windowed min.
    """
    vals = []
    for offset in range(-half_width_deg, half_width_deg + 1):
        d = ranges[(center_deg + offset) % 360]
        if not math.isinf(d) and d >= floor_mm:
            vals.append(d)
    if not vals:
        return float("inf")

    vals.sort()
    if gap_split_mm > 0:
        # cut at the first large jump; everything before it is the near cluster
        cut = len(vals)
        for i in range(1, len(vals)):
            if vals[i] - vals[i - 1] >= gap_split_mm:
                cut = i
                break
        vals = vals[:cut]
    return vals[0]


def pick_bearing(ranges, quals, center_deg, half_width_deg):
    """
    Highest-quality non-inf return within +/- half_width_deg of center_deg.

    Returns (deg, dist_mm, quality) of the winner, or None if every bin in the
    window is inf. Distinct from select_range: no floor / gap-split logic and no
    nearest-cluster bias — it answers "what is the most confident return near
    this bearing", for a raw sensor-health readout, not a fused pillar distance.

    Ties in quality break toward the bin nearest center_deg (offsets are visited
    in order of increasing |offset|, and only a strictly higher quality wins).
    half_width_deg is an integer because ranges/quals are 1-degree-binned.
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