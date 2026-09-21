"""
tools/calibrate_self.py - find the bearings where the LiDAR sees the car.

RUN THIS ON AN EMPTY MAT. Anything close that survives is the robot; if a sign
is standing nearby it will be masked out for the rest of the season and every
detection at that bearing will be silently lost. That mistake has already been
made once by eye - two side bands looked exactly like self-structure and were
actually a sign placed close to the car.

    python -m tools.calibrate_self --moves 4        # live, move the car between
    python -m tools.calibrate_self --replay a.npz b.npz

Move the car (and turn it) between captures. That is what makes the arena vary
while the car stays constant, so only the car is close in every capture.
Writes the bands to config.json under lidar.blind_deg when --write is given.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from perception import selfmask

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "config.json")


def capture_live(moves, per_move=3):
    from worldstate import SharedState
    from sensors.lidar import LidarThread
    shared = SharedState()
    lz = LidarThread(shared)
    lz.start()
    scans = []
    try:
        for m in range(moves):
            input(f"\n[{m+1}/{moves}] MOVE AND TURN the car to a new spot on an "
                  f"EMPTY mat, then press Enter...")
            got, last = 0, -1
            t0 = time.time()
            while got < per_move and time.time() - t0 < 10:
                _c, lidar = shared.snapshot()
                if lidar is not None and lidar.rev != last:
                    last = lidar.rev
                    r = np.asarray(lidar.ranges, dtype=np.float64)
                    if np.isfinite(r).sum() > 30:
                        scans.append(r)
                        got += 1
                time.sleep(0.03)
            print(f"    captured {got} scans")
    finally:
        lz.stop()
    return scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", type=int, default=4)
    ap.add_argument("--replay", nargs="*", default=None,
                    help=".npz files from tools/record_scan.py")
    ap.add_argument("--close-mm", type=float, default=selfmask.SELF_MAX_MM)
    ap.add_argument("--min-frac", type=float, default=0.8)
    ap.add_argument("--write", action="store_true", help="save to config.json")
    args = ap.parse_args()

    if args.replay:
        scans = []
        for p in args.replay:
            d = np.load(p)
            rr = d["ranges"]
            scans.extend(list(rr) if rr.ndim == 2 else [rr])
    else:
        scans = capture_live(args.moves)

    if len(scans) < 2:
        print("need at least two captures from DIFFERENT positions")
        return

    bands = selfmask.calibrate(scans, close_mm=args.close_mm,
                               min_frac=args.min_frac)
    print(f"\n{len(scans)} scans -> blind bands (close in >= "
          f"{args.min_frac:.0%} of them):")
    if not bands:
        print("   none - nothing is consistently close. Good: a clean mount.")
    for lo, hi in bands:
        rs = [float(np.min(s[lo:hi + 1][np.isfinite(s[lo:hi + 1])]))
              for s in scans if np.isfinite(s[lo:hi + 1]).any()]
        spread = (max(rs) - min(rs)) if rs else 0.0
        verdict = "chassis" if (rs and max(rs) < 60) else (
            "SELF (range repeats)" if spread < 20 else
            "SUSPECT - range varies %.0f mm, may be a real object" % spread)
        print(f"   {lo:3d}-{hi:3d} deg   {verdict}")

    if args.write and bands:
        cfg = {}
        if os.path.exists(CFG):
            with open(CFG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        cfg.setdefault("lidar", {})["blind_deg"] = [list(b) for b in bands]
        with open(CFG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nwritten to {CFG} (lidar.blind_deg)")
    elif bands:
        print("\nre-run with --write to save these to config.json")


if __name__ == "__main__":
    main()
