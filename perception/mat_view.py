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
from perception.nav import Navigator
from perception import sim
from nav.pillarmap import RED, GREEN

HERE = os.path.dirname(__file__)
app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))

_latest = {"mode": "starting"}
_raw = {"ranges": None, "t": 0.0}
_lock = threading.Lock()


def publish(snap: dict):
    with _lock:
        _latest.clear()
        _latest.update(snap)


def current():
    with _lock:
        return dict(_latest)


def publish_raw(ranges):
    """Keep the last RAW scan so it can be pulled for offline diagnosis - far
    more reliable than trying to capture one over ssh while the dashboard owns
    the serial port."""
    with _lock:
        _raw["ranges"] = [None if (r is None or not (r == r) or r == float("inf"))
                          else round(float(r), 1) for r in ranges]
        _raw["t"] = time.time()


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
        self.nav = Navigator()
        self.wb = self.nav.wb
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
            publish_raw(r)
            self.nav.step(r, cam_dets=self._camera(truth))
            snap = self.nav.snapshot(ranges=r)
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

    def __init__(self, guess=None, cw=None, hz=10.0, sides=("N", "E", "S", "W"),
                 telemetry_port=None, telemetry_baud=115200):
        super().__init__()
        self.hz = hz
        self.telemetry_port = telemetry_port
        self.telemetry_baud = telemetry_baud
        self.telem = None
        self.guess = guess          # None -> systematic fit_start
        self.cw_hint = cw           # None -> work the direction out from the scan
        self.sides = sides          # restrict to one straight to fix the rotation
        self._stop = threading.Event()
        self.nav = Navigator(sides=self.sides, cw=self.cw_hint)
        self.wb = self.nav.wb

    def run(self):
        from worldstate import SharedState
        from sensors.lidar import LidarThread
        from perception.telemetry import TelemetryReader, camera_dets
        shared = SharedState()
        lz = LidarThread(shared)
        lz.start()
        telem = None
        if self.telemetry_port:
            telem = TelemetryReader(self.nav, port=self.telemetry_port,
                                    baud=self.telemetry_baud)
            telem.start()
        self.telem = telem
        announced = False
        last_rev = -1
        dt = 1.0 / self.hz
        while not self._stop.is_set():
            _cam, lidar = shared.snapshot()
            if lidar is None or lidar.rev == last_rev:
                time.sleep(dt)
                continue
            last_rev = lidar.rev
            ranges = lidar.ranges
            publish_raw(ranges)
            # colours from whatever vision thread is publishing into SharedState
            self.nav.step(ranges, cam_dets=camera_dets(shared))
            if not announced and self.nav.fit is not None:
                print("=== START FIT ===\n" + self.nav.fit.explain())
                announced = True
            snap = self.nav.snapshot(ranges=ranges)
            snap["mode"] = "LIVE · RPLidar"
            if self.telem is not None:
                snap["telem"] = self.telem.status()
            publish(snap)
            time.sleep(dt)
        lz.stop()

    def stop(self):
        self._stop.set()


# ------------------------------- routes ---------------------------------
@app.route("/")
def index():
    return render_template("mat.html")


@app.route("/raw")
def raw():
    with _lock:
        return Response(json.dumps(dict(_raw)), mimetype="application/json")


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
    ap.add_argument("--telemetry", nargs="?", const="/dev/ttyACM0", default=None,
                    metavar="PORT",
                    help="read STM32 TELEM (IMU heading + encoder) for odometry; "
                         "defaults to /dev/ttyACM0")
    ap.add_argument("--telemetry-baud", type=int, default=115200)
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
        src = LiveSource(guess=guess, cw=cw_hint, sides=sides,
                         telemetry_port=args.telemetry,
                         telemetry_baud=args.telemetry_baud)
    else:
        src = SimSource(static=args.static, seed=args.seed)
    src.start()
    print(f"virtual mat on http://{args.host}:{args.port}/  "
          f"({'LIVE' if args.live else 'SIM'})")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
