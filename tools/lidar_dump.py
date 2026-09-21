"""
tools/lidar_dump.py - quick sanity dump of one real RPLidar scan.

Tells you, without any map assumptions, whether the sensor is sane and whether
the car is sitting in something that looks like the WRO corridor: cardinal
distances, return count, min/max, and a 24-sector summary. Also saves the raw
scan to scan.npz for replay through perception.static_perceive --replay.
"""

from __future__ import annotations

import time
import numpy as np

from worldstate import SharedState
from sensors.lidar import LidarThread


def main():
    shared = SharedState()
    lz = LidarThread(shared)
    lz.start()
    print("waiting for a scan ...")
    ranges = None
    t0 = time.time()
    last = -1
    got = []
    while time.time() - t0 < 15 and len(got) < 5:
        _cam, lidar = shared.snapshot()
        if lidar is not None and lidar.rev != last:
            last = lidar.rev
            got.append(np.asarray(lidar.ranges, dtype=np.float64))
        time.sleep(0.05)
    lz.stop()
    if not got:
        print("NO SCAN - lidar not streaming (check power/port/baud)")
        return
    r = got[-1]
    finite = np.isfinite(r)
    print(f"returns: {finite.sum()}/360   min={np.nanmin(r[finite]):.0f}  "
          f"max={np.nanmax(r[finite]):.0f} mm")
    print("cardinal (deg:mm):", {d: (round(float(r[d])) if np.isfinite(r[d]) else None)
                                  for d in (0, 45, 90, 135, 180, 225, 270, 315)})
    # 24 sectors of 15 deg, min distance in each
    print("24-sector min (mm):")
    for s in range(24):
        seg = r[s * 15:(s + 1) * 15]
        seg = seg[np.isfinite(seg)]
        v = f"{np.min(seg):5.0f}" if seg.size else "   inf"
        print(f"  {s*15:3d}-{s*15+14:3d}: {v}", end="\n" if s % 4 == 3 else "  ")
    np.savez_compressed("scan.npz", ranges=np.array(got), qualities=np.zeros((len(got), 360)))
    print(f"\nsaved {len(got)} revs -> scan.npz")


if __name__ == "__main__":
    main()
