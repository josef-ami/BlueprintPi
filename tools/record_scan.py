"""
tools/record_scan.py - capture real RPLidar scans on the Pi to a .npz file, so
they can be replayed through perception.static_perceive / the tests off the
hardware (regression fixtures, and a way to debug a bad mat placement later).

    python -m tools.record_scan --out scan.npz --revs 10

Saves arrays: ranges (revs, 360), qualities (revs, 360).
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scan.npz")
    ap.add_argument("--revs", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    from worldstate import SharedState
    from sensors.lidar import LidarThread

    shared = SharedState()
    lz = LidarThread(shared)
    lz.start()

    ranges, quals = [], []
    last = -1
    t0 = time.time()
    print(f"recording {args.revs} revolutions ...")
    while len(ranges) < args.revs and time.time() - t0 < args.timeout:
        _cam, lidar = shared.snapshot()
        if lidar is None or lidar.rev == last:
            time.sleep(0.02)
            continue
        last = lidar.rev
        ranges.append(np.asarray(lidar.ranges, dtype=np.float64))
        quals.append(np.asarray(lidar.qualities, dtype=np.int32))
        print(f"  rev {len(ranges)}/{args.revs}  "
              f"{np.isfinite(ranges[-1]).sum()}/360 returns")
    lz.stop()

    if not ranges:
        print("no scans captured - is the RPLidar connected?")
        return
    np.savez_compressed(args.out, ranges=np.array(ranges), qualities=np.array(quals))
    print(f"saved {len(ranges)} revs -> {args.out}")


if __name__ == "__main__":
    main()
