"""
perception/seat_dash.py - capture the real sign positions, with your eyes on it.

The seat grid in nav/arena.py was invented, and it has already thrown away a
correct 60 mm detection for landing 198 mm from an imaginary seat. This
replaces it with measurement you can watch happen.

HOW IT WORKS
    The car sits still. It localizes against the MEASURED walls (geom.py reads
    arena_cal.json), so everything below is in real mat coordinates. Put one
    sign down, see it appear as a crosshair, press Record.

ONLY ONE CORRIDOR IS NEEDED
    The arena is 4-fold symmetric, so a seat captured in one straight IS a seat
    in all four, rotated. Capture one corridor and the other three are filled by
    turning each point through 90, 180 and 270 degrees. That also means the
    rotational ambiguity that haunts localization does not matter here at all:
    whichever corridor the localizer thinks it is in, the captured geometry is
    the same geometry.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from perception.state import WorldBelief, scan_points
from perception.tower import find_towers, tower_points
from perception.nav import LIDAR_AHEAD_MM
from nav import arena, geom

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SEATS_FILE = os.path.join(ROOT, "arena_seats.json")

app = Flask(__name__, template_folder=os.path.join(HERE, "templates"))
_lock = threading.Lock()
_state = {"snap": {"mode": "starting"}, "seats": [], "log": []}


def rotate90(x, y, k):
    """A point turned through k quarter-turns about the mat centre."""
    for _ in range(k % 4):
        x, y = -y, x
    return x, y


def replicated(seats):
    """One corridor's captures, mirrored into all four by symmetry."""
    out = []
    for s in seats:
        for k in range(4):
            x, y = rotate90(s["x"], s["y"], k)
            out.append({"x": round(x, 1), "y": round(y, 1),
                        "src": s.get("id", "?"), "rot": k * 90})
    return out


def note(msg):
    with _lock:
        _state["log"].insert(0, f"{time.strftime('%H:%M:%S')}  {msg}")
        del _state["log"][12:]
    print("[seats] " + msg)


class Capture(threading.Thread):
    daemon = True

    def __init__(self, hz=10.0, sides=("N", "E", "S", "W")):
        super().__init__()
        self.hz = hz
        self.sides = sides
        self.wb = WorldBelief(sensor_ahead=LIDAR_AHEAD_MM)
        self.located = False
        self.last_scan = None
        self.towers = []
        self._stop = threading.Event()
        self._relocate = threading.Event()

    def run(self):
        from worldstate import SharedState
        from sensors.lidar import LidarThread
        shared = SharedState()
        lz = LidarThread(shared)
        lz.start()
        last_rev, dt = -1, 1.0 / self.hz
        while not self._stop.is_set():
            _c, li = shared.snapshot()
            if li is None or li.rev == last_rev:
                time.sleep(dt)
                continue
            last_rev = li.rev
            r = np.asarray(li.ranges, dtype=np.float64)
            if np.isfinite(r).sum() < 30:
                continue
            self.last_scan = r

            if not self.located or self._relocate.is_set():
                res = self.wb.fit_start(r, sides=self.sides)
                self._relocate.clear()
                if res is not None:
                    self.located = True
                    note(f"located on straight {res.side}, walls "
                         f"{res.wall:.2f}, direction {res.direction}")
            else:
                self.wb.track(r)

            ts = find_towers(r)
            self.towers = list(zip(ts, tower_points(r, self.wb.sensor_pose, ts)))
            self.publish(r)
            time.sleep(dt)
        lz.stop()

    def publish(self, r):
        x, y, th = self.wb.pose
        with _lock:
            seats = list(_state["seats"])
        snap = {
            "mode": "SEAT CAPTURE",
            "arena": {"outer": geom.OUTER, "inner": geom.INNER,
                      "corridor": geom.CORRIDOR},
            "pose": {"x": round(x, 1), "y": round(y, 1),
                     "th_deg": round(math.degrees(th), 1)},
            "score": round(self.wb.score, 3),
            "located": self.located,
            "scan": scan_points(self.wb.sensor_pose, r),
            "towers": [{"x": round(p[0], 1), "y": round(p[1], 1),
                        "size_mm": round(t.size_mm, 1),
                        "range_mm": round(t.range_mm, 1),
                        "bearing_deg": round(t.bearing_deg, 1)}
                       for t, p in self.towers],
            "seats": seats,
            "replicated": replicated(seats),
        }
        with _lock:
            _state["snap"] = snap

    def record(self, samples=5):
        """Average the sign currently in view over a few revolutions."""
        if not self.located:
            return False, "not located yet"
        pts, t0 = [], time.time()
        while len(pts) < samples and time.time() - t0 < 3.0:
            if self.towers and len(self.towers) == 1:
                pts.append(self.towers[0][1])
            time.sleep(0.12)
        if not pts:
            n = len(self.towers)
            return False, ("no sign in view" if n == 0 else
                           f"{n} signs in view - record one at a time")
        P = np.array(pts)
        mx, my = float(P[:, 0].mean()), float(P[:, 1].mean())
        spread = float(np.hypot(P[:, 0].std(), P[:, 1].std()))
        near, d = arena.nearest_seat(mx, my, tol=1e9)
        with _lock:
            sid = f"S{len(_state['seats']) + 1}"
            _state["seats"].append({"id": sid, "x": round(mx, 1),
                                    "y": round(my, 1),
                                    "spread_mm": round(spread, 1),
                                    "samples": len(P)})
        note(f"{sid} at ({mx:.0f}, {my:.0f})  spread {spread:.0f} mm  "
             f"[invented grid's nearest: {near.id} at {d:.0f} mm]")
        return True, sid

    def relocate(self):
        self._relocate.set()


CAP: Capture | None = None


@app.route("/")
def index():
    return render_template("seats.html")


@app.route("/events")
def events():
    def stream():
        while True:
            with _lock:
                payload = dict(_state["snap"])
                payload["log"] = list(_state["log"])
            yield "data: " + json.dumps(payload) + "\n\n"
            time.sleep(0.1)
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/record", methods=["POST"])
def record():
    ok, msg = CAP.record()
    if not ok:
        note(msg)
    return jsonify(ok=ok, msg=msg)


@app.route("/undo", methods=["POST"])
def undo():
    with _lock:
        s = _state["seats"].pop() if _state["seats"] else None
    note(f"removed {s['id']}" if s else "nothing to undo")
    return jsonify(ok=bool(s))


@app.route("/relocate", methods=["POST"])
def relocate():
    CAP.relocate()
    return jsonify(ok=True)


@app.route("/save", methods=["POST"])
def save():
    with _lock:
        seats = list(_state["seats"])
    full = replicated(seats)
    with open(SEATS_FILE, "w", encoding="utf-8") as f:
        json.dump({"captured": seats, "seats": full,
                   "arena": {"outer_mm": geom.OUTER, "inner_mm": geom.INNER,
                             "corridor_mm": geom.CORRIDOR},
                   "note": "captured in ONE corridor, replicated by 4-fold "
                           "symmetry into the other three",
                   "source": "perception/seat_dash.py"}, f, indent=2)
    note(f"saved {len(seats)} captured -> {len(full)} seats in {SEATS_FILE}")
    return jsonify(ok=True, captured=len(seats), total=len(full))


def main(host="0.0.0.0", port=8081, sides=("N", "E", "S", "W")):
    global CAP
    if os.path.exists(SEATS_FILE):
        try:
            with open(SEATS_FILE, "r", encoding="utf-8") as f:
                _state["seats"] = json.load(f).get("captured", [])
            note(f"loaded {len(_state['seats'])} previously captured")
        except Exception:                                # noqa: BLE001
            pass
    CAP = Capture(sides=sides)
    CAP.start()
    print(f"[seats] map {geom.OUTER:.0f}/{geom.INNER:.0f}/{geom.CORRIDOR:.0f} mm")
    print(f"[seats] open http://<pi>:{port}/")
    app.run(host=host, port=port, threaded=True, debug=False)
