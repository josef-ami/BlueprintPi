"""
Main entry point — camera-first architecture.

Camera detects obstacles (colour + bearing via the atan pinhole model).
Lidar supplies a 360-degree range array. Fusion attaches a distance to each
camera obstacle by reading the lidar range at that obstacle's bearing.

Two producer threads write to SharedState; this loop reads snapshot(), fuses,
and (eventually) drives the FSM + MCU. For now it prints so you can watch the
obstacle list track reality before the MCU exists.
"""

import time
import signal

from worldstate import SharedState
from sensors.camera import CameraThread, load_config
from sensors.lidar import LidarThread, sector_min, select_range

LOOP_HZ = 30

_FUSION = load_config().get("fusion", {})
BEARING_MATCH_DEG = _FUSION.get("bearing_match_deg", 8)
RANGE_FLOOR_MM = _FUSION.get("range_floor_mm", 0.0)
GAP_SPLIT_MM = _FUSION.get("gap_split_mm", 0.0)


def fuse(camera_result, lidar_result):
    """
    Attach a distance to each camera obstacle from the lidar returns at its
    bearing, using floor + gap-split selection (see select_range): the pillar
    is nearest by construction, so we take the min of the near cluster and
    ignore the wall behind it. Distance stays inf if nothing valid is found.

    Uses the fusion parameters read from config.json when this module was
    imported. fuse_with() is the same thing with explicit parameters.
    """
    return fuse_with(camera_result, lidar_result,
                     BEARING_MATCH_DEG, RANGE_FLOOR_MM, GAP_SPLIT_MM)


def fuse_with(camera_result, lidar_result, bearing_match_deg, range_floor_mm,
              gap_split_mm):
    """
    fuse() with explicit parameters, for a caller whose config was read later
    than this module's import (dashboard.py loads config.json when a run is
    started, not when the dashboard process started).
    """
    if camera_result is None:
        return []
    obstacles = camera_result.obstacles
    if lidar_result is None:
        return obstacles
    ranges = lidar_result.ranges
    for obs in obstacles:
        center = int(round(obs.bearing_deg)) % 360
        obs.distance_mm = select_range(ranges, center, bearing_match_deg,
                                       floor_mm=range_floor_mm,
                                       gap_split_mm=gap_split_mm)
    return obstacles


def main():
    shared = SharedState()
    cam = CameraThread(shared)
    lidar = LidarThread(shared)

    stop = {"flag": False}

    def handle_sigint(sig, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    cam.start()
    lidar.start()
    print("Threads started. Ctrl-C to stop.")

    period = 1.0 / LOOP_HZ
    try:
        while not stop["flag"]:
            t0 = time.time()

            camera_result, lidar_result = shared.snapshot()
            obstacles = fuse(camera_result, lidar_result)

            front = (sector_min(lidar_result.ranges, 0, 15)
                     if lidar_result else float("inf"))
            summary = ", ".join(
                f"{o.color}@{o.bearing_deg:+.0f}deg/{o.distance_mm:.0f}mm"
                for o in obstacles
            ) or "no obstacles"
            print(f"front={front:7.0f}mm | {summary}")

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        print("\nShutting down...")
        cam.stop()
        lidar.stop()
        cam.join(timeout=2.0)
        lidar.join(timeout=2.0)
        print("Done.")


if __name__ == "__main__":
    main()
