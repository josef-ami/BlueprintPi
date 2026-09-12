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
from sensors.camera import CameraThread
from sensors.lidar import LidarThread, sector_min

LOOP_HZ = 30
BEARING_MATCH_DEG = 8   # lidar sector half-width searched around a bearing


def fuse(camera_result, lidar_result):
    """
    Attach a distance to each camera obstacle from the lidar range at its
    bearing. Obstacle keeps distance=inf if the lidar has no return there.
    """
    if camera_result is None:
        return []
    obstacles = camera_result.obstacles
    if lidar_result is None:
        return obstacles
    ranges = lidar_result.ranges
    for obs in obstacles:
        center = int(round(obs.bearing_deg)) % 360
        obs.distance_mm = sector_min(ranges, center, BEARING_MATCH_DEG)
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
