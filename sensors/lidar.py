"""
Lidar adapter.

Seals the rplidarc1 library's asyncio TaskGroup pattern inside ONE dedicated
thread that runs its own event loop. The rest of the program never touches
asyncio and never sees the library — it only reads the shared LidarResult.

The library streams individual points as dicts: {'q': int, 'a_deg': float,
'd_mm': int|None}. We accumulate them into a rolling 360-element array indexed
by integer degree, converting None -> inf on the way in.
"""

import asyncio
import threading
import time

from rplidarc1 import RPLidar
from worldstate import SharedState, LidarResult

PORT = "/dev/ttyUSB0"
BAUD = 460800

# If your lidar's 0deg doesn't point at robot-forward, set this offset (deg).
# A point's stored index = int(a_deg + MOUNT_OFFSET_DEG) % 360.
MOUNT_OFFSET_DEG = 0

# How often to publish the rolling array to shared state (seconds).
PUBLISH_PERIOD = 0.02   # 50 Hz


class LidarThread(threading.Thread):
    def __init__(self, shared: SharedState):
        super().__init__(name="LidarThread", daemon=True)
        self.shared = shared
        self._lidar = None
        self._loop = None
        self._ranges = [float("inf")] * 360   # owned by this thread only
        self._async_stop = None               # created inside the loop

    # ---- thread entry point: build an event loop, run the async world ----
    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as e:
            print(f"[LidarThread] fatal: {type(e).__name__}: {e}")
        finally:
            # library teardown must happen from a place with no running loop
            try:
                if self._lidar is not None:
                    self._lidar.reset()
            except Exception:
                pass
            self._loop.close()
            print("[LidarThread] stopped")

    async def _async_main(self):
        # RPLidar() does blocking serial I/O in __init__ (connect + healthcheck),
        # so construct it inside run() but off the hot path. Doing it here is fine.
        self._lidar = RPLidar(PORT, BAUD)
        self._async_stop = self._lidar.stop_event  # the library's own asyncio.Event

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._consume(self._lidar.output_queue, self._async_stop))
            tg.create_task(self._publisher(self._async_stop))
            tg.create_task(self._lidar.simple_scan())   # producer

    async def _consume(self, queue, stop_event):
        """Drain points as they stream in, update the rolling array."""
        while not stop_event.is_set():
            try:
                point = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            d = point["d_mm"]
            idx = int(point["a_deg"] + MOUNT_OFFSET_DEG) % 360
            self._ranges[idx] = float("inf") if d is None else float(d)

    async def _publisher(self, stop_event):
        """Periodically copy the rolling array into shared state."""
        while not stop_event.is_set():
            snapshot = list(self._ranges)   # copy so readers can't see mid-update
            self.shared.set_lidar(LidarResult(timestamp=time.time(), ranges=snapshot))
            await asyncio.sleep(PUBLISH_PERIOD)

    # ---- called from the MAIN thread to request shutdown ----
    def stop(self):
        """Thread-safe: schedule the library's stop_event to be set on the loop."""
        if self._loop is not None and self._async_stop is not None:
            self._loop.call_soon_threadsafe(self._async_stop.set)


# convenience for reading a sector min-distance from a ranges array
def sector_min(ranges, center_deg, half_width_deg):
    """Smallest distance within +/- half_width of center_deg. inf if all clear."""
    best = float("inf")
    for offset in range(-half_width_deg, half_width_deg + 1):
        best = min(best, ranges[(center_deg + offset) % 360])
    return best
