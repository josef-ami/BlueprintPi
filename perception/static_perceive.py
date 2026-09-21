"""
perception/static_perceive.py - the on-Pi static perception test.

Place the car (RPLidar C1) somewhere on the arena, tell it roughly where that
is, and this answers the one question that has to be true before anything else:
can it perceive? It captures a scan, localizes against the known matrix, reads
the occupied seats and their colours, and locates the parking bay, then prints a
report. Success = the printed pose + pillars + parking match the physical mat.

    python -m perception.static_perceive --sim                 # synthetic scan
    python -m perception.static_perceive --replay scan.npz     # a recorded scan
    python -m perception.static_perceive --live --guess -1000,-1000,0

`--guess x,y,deg` seeds the global fix; on a square-in-square mat a scan alone
cannot tell the four corridors apart (4-fold symmetry), so tell it which
corridor the car is in - anywhere within ~150 mm is plenty.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from perception.state import WorldBelief
from perception import sim
from nav.pillarmap import RED, GREEN


def parse_guess(s):
    if not s:
        return None
    x, y, d = (float(v) for v in s.split(","))
    return (x, y, math.radians(d))


def _sim_source(seed, guess):
    sc = sim.random_scenario(seed=seed, n_pillars=6)
    segs = sim.scenario_segments(sc)
    truth = guess or (-1000.0, -1000.0, 0.0)
    rng = np.random.default_rng(seed)

    def gen():
        while True:
            r, _q = sim.synth_scan(truth, segs, rng=rng)
            cam = []
            for (px, py, col, _sid) in sc.pillar_list():
                b = math.degrees((math.atan2(py - truth[1], px - truth[0])
                                  - truth[2] + math.pi) % (2 * math.pi) - math.pi)
                if abs(b) < 48 and math.hypot(px - truth[0], py - truth[1]) < 1400:
                    cam.append((col, b))
            yield r, cam
    return gen(), truth, sc


def _replay_source(path):
    data = np.load(path, allow_pickle=True)
    ranges = data["ranges"]           # (N, 360) or (360,)
    if ranges.ndim == 1:
        ranges = ranges[None, :]

    def gen():
        i = 0
        while True:
            yield list(ranges[i % len(ranges)]), None
            i += 1
    return gen(), None, None


def _live_source():
    from worldstate import SharedState
    from sensors.lidar import LidarThread
    shared = SharedState()
    lz = LidarThread(shared)
    lz.start()

    def gen():
        last = -1
        while True:
            _cam, lidar = shared.snapshot()
            if lidar is None or lidar.rev == last:
                time.sleep(0.05)
                continue
            last = lidar.rev
            yield list(lidar.ranges), None
    return gen(), None, lz


def run(source, guess, frames=12, settle=6):
    gen, truth, sc = source
    wb = WorldBelief()
    first = True
    for i in range(frames):
        ranges, cam = next(gen)
        if first:
            wb.global_init(ranges, guess=guess)
            first = False
        else:
            wb.track(ranges)                # LiDAR-only (no IMU/encoder)
        wb.update(ranges, cam_dets=cam)
        last_ranges = ranges
    return wb, last_ranges, truth, sc


def report(wb, truth=None, sc=None):
    x, y, th = wb.pose
    print("\n=== STATIC PERCEPTION REPORT ===")
    print(f"pose      x={x:7.1f}  y={y:7.1f}  heading={math.degrees(th):6.1f} deg"
          f"   match score={wb.score:.2f}   healthy={wb.healthy}")
    if truth is not None:
        err = math.hypot(x - truth[0], y - truth[1])
        dth = abs((math.degrees(th - truth[2]) + 180) % 360 - 180)
        print(f"truth     x={truth[0]:7.1f}  y={truth[1]:7.1f}  "
              f"heading={math.degrees(truth[2]):6.1f} deg"
              f"   -> pos err {err:.0f} mm, heading err {dth:.1f} deg")
    snap = wb.snapshot()
    occ = [s for s in snap["seats"] if s["state"] not in ("unknown", "empty")]
    emp = [s for s in snap["seats"] if s["state"] == "empty"]
    print(f"\nseats     {len(occ)} occupied, {len(emp)} seen-empty, "
          f"{24 - len(occ) - len(emp)} unknown")
    for s in occ:
        print(f"   {s['id']:>3}  {s['state']:<6} at ({s['x']:.0f},{s['y']:.0f})  conf {s['conf']}")
    if snap.get("parking"):
        pk = snap["parking"]
        print(f"\nparking   side {pk['side']}  rect {tuple(round(v) for v in pk['rect'])}")
    else:
        print("\nparking   not located from this vantage")
    if sc is not None:
        truth_occ = {sid: ("red" if c == RED else "green") for sid, c in sc.pillars.items()}
        print(f"\nground truth pillars: {truth_occ}")
    print("================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--replay", metavar="NPZ")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--guess", help="x,y,deg  (seed the fix / which corridor)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()

    guess = parse_guess(args.guess)
    if args.replay:
        src = _replay_source(args.replay)
    elif args.live:
        src = _live_source()
        if guess is None:
            guess = (-1000.0, -1000.0, 0.0)
    else:
        src = _sim_source(args.seed, guess)

    wb, ranges, truth, sc = run(src, guess, frames=args.frames)
    report(wb, truth=truth, sc=sc)


if __name__ == "__main__":
    main()
