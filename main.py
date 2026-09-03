"""
Main entry point.

Spins up the camera thread and the lidar thread, then runs the fusion/decision
loop in the main thread. The fusion loop reads snapshot() once per cycle, fuses
camera bearings with lidar ranges, and (eventually) drives the FSM + MCU.

For now it just fuses and prints, so you can watch WorldState track reality
before the MCU exists.
"""

import time
import signal

from worldstate import SharedState
from sensors.camera import CameraThread
from sensors.lidar import LidarThread, sector_min

LOOP_HZ = 30
BEARING_MATCH_DEG = 8   # how close a lidar sector must be to a blob bearing


def fuse(camera_result, lidar_result):
    """
    Attach a real distance to each camera obstacle by looking up the lidar
    range at the same bearing. Returns the enriched obstacle list.
    """
    if camera_result is None:
        return []
    obstacles = camera_result.obstacles
    if lidar_result is None:
        return obstacles   # no ranges yet; bearings only

    ranges = lidar_result.ranges
    for obs in obstacles:
        # lidar array is indexed by integer degree, 0=forward increasing CCW;
        # a +bearing (left) maps to that same degree convention.
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

            # --- placeholder for FSM + MCU command; just observe for now ---
            front = sector_min(lidar_result.ranges, 0, 15) if lidar_result else float("inf")
            summary = ", ".join(
                f"{o.color}@{o.bearing_deg:+.0f}deg/{o.distance_mm:.0f}mm"
                for o in obstacles
            ) or "no obstacles"
            print(f"front={front:7.0f}mm | {summary}")

            # keep a steady loop rate
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
