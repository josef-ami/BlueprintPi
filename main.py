"""
Main entry point — lidar-first architecture.

The LIDAR detects discrete obstacles (bearing + distance + size) from its
filtered, clustered point cloud. The CAMERA reports only colours-at-bearings.
Fusion stamps a colour onto each lidar obstacle by matching bearings.

Two producer threads write to SharedState; this loop reads snapshot(), fuses,
and (eventually) drives the FSM + MCU. For now it prints so you can watch the
coloured obstacle list track reality before the MCU exists.
"""

import time
import signal

from worldstate import SharedState
from sensors.camera import CameraThread
from sensors.lidar import LidarThread, sector_min

LOOP_HZ = 30
BEARING_MATCH_DEG = 10   # camera colour must fall within this of a lidar obstacle


def fuse(camera_result, lidar_result):
    """
    Colour the lidar-detected obstacles.

    For each LidarObstacle, find the camera ColorDetection whose bearing is
    closest (within BEARING_MATCH_DEG) and adopt its colour. Obstacles with no
    colour match stay UNKNOWN — a real object the camera couldn't classify
    (out of frame, mis-tuned HSV, or a wall segment rather than a pillar).
    Returns the list of coloured LidarObstacles.
    """
    if lidar_result is None:
        return []
    obstacles = lidar_result.obstacles
    detections = camera_result.detections if camera_result is not None else []

    for obs in obstacles:
        best = None
        best_err = BEARING_MATCH_DEG
        for det in detections:
            err = abs(det.bearing_deg - obs.bearing_deg)
            if err < best_err:
                best_err = err
                best = det
        obs.color = best.color if best is not None else "UNKNOWN"
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
                f"(w{o.width_deg:.0f} n{o.point_count})"
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
