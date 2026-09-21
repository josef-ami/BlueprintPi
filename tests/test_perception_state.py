"""Semantic belief: pillar detection at seats, colour, parking, occupancy."""

import math

import numpy as np
import pytest

from perception import sim
from perception.state import WorldBelief
from nav.pillarmap import RED, GREEN


def _run(sc, truth, frames=10, seed=0, camera=True):
    segs = sim.scenario_segments(sc)
    wb = WorldBelief()
    wb.set_parking(sc.parking())
    rng = np.random.default_rng(seed)
    for i in range(frames):
        r, _q = sim.synth_scan(truth, segs, rng=rng)
        if i == 0:
            wb.global_init(r, guess=truth)
        else:
            wb.correct(r)
        cam = []
        if camera:
            for (px, py, col, _sid) in sc.pillar_list():
                b = math.degrees((math.atan2(py - truth[1], px - truth[0])
                                  - truth[2] + math.pi) % (2 * math.pi) - math.pi)
                if abs(b) < 48 and math.hypot(px - truth[0], py - truth[1]) < 1400:
                    cam.append((col, b))
        wb.update(r, cam_dets=cam)
    return wb


def test_pillar_recall_and_precision_at_seats():
    # pillars the car can see from the SW corner; no false seats occupied
    sc = sim.Scenario(pillars={"S0": RED, "S2": GREEN, "W0": RED},
                      parking_side="N")
    wb = _run(sc, (-1000.0, -1000.0, 0.0))
    snap = wb.snapshot()
    occ = {s["id"] for s in snap["seats"] if s["state"] not in ("unknown", "empty")}
    # recall: every truth seat detected
    assert set(sc.pillars) <= occ
    # precision: nothing occupied that was not a real pillar
    assert occ <= set(sc.pillars)


def test_colour_labelled_when_in_camera_fov():
    # one pillar straight ahead in the east straight
    sc = sim.Scenario(pillars={"E4": RED}, parking_side="N")
    wb = _run(sc, (1000.0, -200.0, math.radians(90)))
    reds = [p for p in wb.snapshot()["pillars"] if p["color"] == "red"]
    assert len(reds) == 1


def test_seat_seen_empty_becomes_empty():
    sc = sim.Scenario(pillars={"S2": RED}, parking_side="N")   # only S2 occupied
    wb = _run(sc, (-1000.0, -1000.0, 0.0), camera=False)
    seats = {s["id"]: s["state"] for s in wb.snapshot()["seats"]}
    assert seats["S2"] in ("pillar", "red", "green")
    # a clearly visible neighbouring seat with nothing on it reads empty
    assert seats["S0"] == "empty"


def test_parking_located_when_both_blocks_visible():
    sc = sim.Scenario(pillars={}, parking_side="S", parking_along=0.0)
    # sit in the south straight looking east; both south-wall blocks in view
    wb = _run(sc, (0.0, -1000.0, 0.0), camera=False)
    pk = wb.snapshot().get("parking")
    assert pk is not None and pk["side"] == "S"
