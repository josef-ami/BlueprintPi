"""
perception/mat_view.py - the live virtual-mat web dashboard.

A tiny Flask app that streams the WorldBelief snapshot (pose + seat states +
parking + the live scan superimposed on the matrix) to a browser canvas over
Server-Sent Events. It is additive - it does not touch the existing
dashboard.py / templates/index.html.

Run it three ways:
    python -m perception.mat_view --sim              # simulated car driving a lap
    python -m perception.mat_view --sim --static     # simulated car parked in place
    python -m perception.mat_view --live             # real RPLidar on the Pi

Then open http://<host>:8080/ .
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time

import numpy as np
from flask import Flask, Response, render_template

from perception.state import WorldBelief
from perception import sim
from nav.pillarmap import RED, GREEN

HERE = os.path.dirname(__file__)
app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))

_latest = {"mode": "starting"}
_lock = threading.Lock()


def publish(snap: dict):
    with _lock:
        _latest.clear()
        _latest.update(snap)


def current():
    with _lock:
        return dict(_latest)


# ------------------------------- sources --------------------------------
class SimSource(threading.Thread):
    """Drives a WorldBelief from synthetic scans - a lap of the corridor, or
    static in one spot - so the whole render path works with no hardware."""

    daemon = True

    def __init__(self, static=False, seed=1, hz=10.0):
        super().__init__()
        self.static = static
        self.hz = hz
        self.rng = np.random.default_rng(seed)
        self.sc = sim.random_scenario(seed=seed, n_pillars=6)
        self.segs = sim.scenario_segments(self.sc)
        self.wb = WorldBelief()
        self.wb.set_parking(self.sc.parking())
        self._stop = threading.Event()

    def _truth_poses(self):
        if self.static:
            # a fixed, interesting vantage: near a corner so along is observable
            while not self._stop.is_set():
                yield (-1000.0, -1000.0, math.radians(0))
        else:
            wpts = sim.corridor_waypoints(step_mm=70.0)
            i = 0
            while not self._stop.is_set():
                yield wpts[i % len(wpts)]
                i += 1

    def run(self):
        gen = self._truth_poses()
        first = True
        dt = 1.0 / self.hz
        for truth in gen:
            r, q = sim.synth_scan(truth, self.segs, rng=self.rng)
            if first:
                self.wb.global_init(r, guess=truth)   # seed which corridor
            else:
                self.wb.track(r)                      # LiDAR-only, no odometry
            cam = self._camera(truth)
            self.wb.update(r, cam_dets=cam)
            snap = self.wb.snapshot(ranges=r)
            snap["mode"] = "SIM · static" if self.static else "SIM · driving (LiDAR only)"
            snap["truth"] = {"x": round(truth[0], 1), "y": round(truth[1], 1),
                             "th_deg": round(math.degrees(truth[2]), 1)}
            publish(snap)
            first = False
            time.sleep(dt)

    def _camera(self, truth):
        tx, ty, tth = truth
        out = []
        for (px, py, col, sid) in self.sc.pillar_list():
            b = math.degrees((math.atan2(py - ty, px - tx) - tth + math.pi)
                             % (2 * math.pi) - math.pi)
            if abs(b) < 48 and math.hypot(px - tx, py - ty) < 1400:
                out.append((col, b))
        return out

    def stop(self):
        self._stop.set()


class LiveSource(threading.Thread):
    """Real RPLidar on the Pi. Imports the hardware stack lazily so --sim never
    needs rplidar installed."""

    daemon = True

    def __init__(self, guess=None, cw=None, hz=10.0, sides=("N", "E", "S", "W")):
        super().__init__()
        self.hz = hz
        self.guess = guess          # None -> systematic fit_start
        self.cw_hint = cw           # None -> work the direction out from the scan
        self.sides = sides          # restrict to one straight to fix the rotation
        self._stop = threading.Event()
        self.wb = WorldBelief(sensor_ahead=0.0)

    def run(self):
        from worldstate import SharedState
        from sensors.lidar import LidarThread
        shared = SharedState()
        lz = LidarThread(shared)
        lz.start()
        first = True
        last_rev = -1
        dt = 1.0 / self.hz
        while not self._stop.is_set():
            _cam, lidar = shared.snapshot()
            if lidar is None or lidar.rev == last_rev:
                time.sleep(dt)
                continue
            last_rev = lidar.rev
            ranges = lidar.ranges
            if first:
                if self.guess is not None:
                    _p, sc = self.wb.global_init(ranges, guess=self.guess)
                    first = sc <= 0.0            # retry until a real fix lands
                else:
                    res = self.wb.fit_start(ranges, cw=self.cw_hint, sides=self.sides)
                    first = res is None
                    if res is not None:
                        print("=== START FIT ===\n" + res.explain())
                if first:
                    continue                    # empty/spin-up scan, wait
            else:
                self.wb.track(ranges)       # LiDAR-only pose (no IMU/encoder)
            self.wb.update(ranges)          # camera wiring is a later step
            snap = self.wb.snapshot(ranges=ranges)
            snap["mode"] = "LIVE · RPLidar"
            publish(snap)
            first = False
            time.sleep(dt)
        lz.stop()

    def stop(self):
        self._stop.set()


# ------------------------------- routes ---------------------------------
@app.route("/")
def index():
    return render_template("mat.html")


@app.route("/snapshot")
def snapshot():
    return Response(json.dumps(current()), mimetype="application/json")


@app.route("/events")
def events():
    def stream():
        while True:
            yield "data: " + json.dumps(current()) + "\n\n"
            time.sleep(0.1)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="synthetic scans, no hardware")
    ap.add_argument("--static", action="store_true", help="sim: park in one spot")
    ap.add_argument("--live", action="store_true", help="real RPLidar on the Pi")
    ap.add_argument("--guess", help="x,y,deg  (override auto-init with an explicit pose)")
    ap.add_argument("--cw", action="store_true", help="force clockwise (default: infer from the scan)")
    ap.add_argument("--ccw", action="store_true", help="force counter-clockwise")
    ap.add_argument("--start", choices=["N","E","S","W"],
                    help="which straight the car starts in (resolves the rotation)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    guess = None
    if args.guess:
        gx, gy, gd = (float(v) for v in args.guess.split(","))
        guess = (gx, gy, math.radians(gd))

    cw_hint = True if args.cw else (False if args.ccw else None)
    if args.live:
        sides = (args.start,) if args.start else ("N", "E", "S", "W")
        src = LiveSource(guess=guess, cw=cw_hint, sides=sides)
    else:
        src = SimSource(static=args.static, seed=args.seed)
    src.start()
    print(f"virtual mat on http://{args.host}:{args.port}/  "
          f"({'LIVE' if args.live else 'SIM'})")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
