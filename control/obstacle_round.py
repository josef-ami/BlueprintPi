#!/usr/bin/env python3
"""
obstacle_round.py - the obstacle round's control loop.

    VisionThread ─┐
                  ├─> perception ─> Ctx ─> FSM ─> ActionIntent ─> mapper ─┐
    LidarThread  ─┘        ^                                              │
                           │                                             v
                      WallFit, pillar xy, candidates                  Link ─> STM32
                                                                          <─ TELEM

One tick per 1/SEND_HZ. The loop itself lives in class ObstacleRound so
dashboard.py runs exactly this code - same perception reads, same FSM, same
frames - on the camera and LiDAR it already owns. main() below is the CLI
around it.

Run this INSTEAD of dashboard.py: one process may hold the camera and the
LiDAR at a time.

    python3 -m control.obstacle_round            # normal run
    python3 -m control.obstacle_round --dry      # no serial: decide and print
    python3 -m control.obstacle_round --quiet    # one status line per second
"""

import argparse
import math
import threading
import time
from dataclasses import dataclass, field

import params as prm
from worldstate import SharedState, WallFit
import sensors.camera as camera
import sensors.lidar as lidar_mod
from sensors.lidar import (LidarThread, cones, lidar_candidates, locate_pillar,
                           read_three_picks, unclassified, lidar_live)

from .fsm import FSM, Ctx, State
from .intent import ActionIntent, SteerMode
from .link import Link, Telemetry, CMD_NONE, CMD_REBOOT
from .mapper import CommandMapper


@dataclass
class Tick:
    """Everything one pass of the loop saw, decided and sent."""
    t: float = 0.0
    lidar_live: bool = False
    cam_live: bool = False
    new_frame: bool = False
    new_rev: bool = False
    picks: tuple = ()
    front: object = None
    left: object = None
    right: object = None
    wall: object = None
    pillar_colour: str = None
    pillar_xy: tuple = None
    pillar_area: int = 0
    sec_colour: str = None
    sec_xy: tuple = None
    unknown_xy: tuple = None
    n_candidates: int = 0
    telem: object = None
    telem_fresh: bool = False
    intent: object = None
    clamped: str = ""
    sent: bool = False
    rev: int = 0
    rev_per_s: int = 0
    queue_depth: int = 0
    tick_ms: float = 0.0
    fsm: dict = field(default_factory=dict)


class ObstacleRound:
    """The control loop. One tick() per 1/SEND_HZ.

    p        PiParams, live - a change on the Tune tab lands on the next tick
    stm      StmParams, the firmware mirror
    shared   SharedState the vision thread publishes into
    lidar    the running LidarThread (its live bins are read directly, which
             skips the 20 Hz SharedState hop and saves 0-50 ms)
    link     an opened, started Link; ignored when dry
    emit     where printed lines go (print, for the CLI)
    """

    def __init__(self, p, stm, shared, lidar, link=None, dry=False,
                 quiet=False, emit=print):
        self.p = p
        self.stm = stm
        self.shared = shared
        self.lidar = lidar
        self.link = link
        self.dry = dry
        self.quiet = quiet
        self.emit = emit

        self.fsm = FSM(p, emit=emit)
        self.mapper = CommandMapper()

        self.last = None                 # the newest Tick
        self.last_log = 0.0
        self.last_rev = 0
        self.rev_per_s = 0
        self._cone_rev = -1
        self._wall = WallFit()
        self._cands = []
        self._last_cam_seq = -1
        self._reboot_frames = 0

    # ---- the page's buttons ----

    def start(self):
        self.fsm.request_start()

    def stop(self):
        self.fsm.request_stop()

    def reboot(self):
        """Hold CMD_REBOOT for a few frames; the firmware acts on a run of
        identical values so one corrupt frame can never reset the car."""
        self._reboot_frames = int(self.p["CMD_REPEAT"]) * 2

    def banner(self):
        p = self.p
        self.emit(f"[round] obstacle round. send={p['SEND_HZ']} Hz  "
                  f"tol=+/-{p['BEARING_TOL_DEG']} deg  "
                  f"corridor={p['CORRIDOR_MM']:.0f}  "
                  f"pass_clear={p.derived['PASS_CLEAR_MM']:.0f}  "
                  f"lane_limit={p.derived['LANE_LIMIT_MM']:.0f}  "
                  f"link={'DRY' if self.dry else 'serial'}")

    # ------------------------------------------------------------ one pass

    def tick(self) -> Tick:
        t0 = time.time()
        mono = time.monotonic()
        p = self.p
        lidar = self.lidar

        # ---- perception -------------------------------------------------
        live = lidar_live(lidar, mono, p["LIDAR_STALE_S"])
        dead = not lidar.is_alive() or \
            (mono - lidar.last_point_t) >= p["LIDAR_DEAD_S"]
        picks = (None, None, None)
        front = left = right = None
        new_rev = False
        if live:
            picks = read_three_picks(lidar._ranges, lidar._quals,
                                     p["BEARING_TOL_DEG"])
            front, left, right = (None if q is None else q[1] for q in picks)
            # The cone fits and the candidate search only change once per
            # revolution, so they run once per revolution and not 50 times.
            if lidar.rev != self._cone_rev:
                self._cone_rev = lidar.rev
                new_rev = True
                self._wall = cones(lidar._ranges, p)
                self._cands = lidar_candidates(lidar._ranges, self._wall, p)

        cam_res, _ = self.shared.snapshot()
        cam_live = cam_res is not None and \
            (t0 - cam_res.timestamp) < p["VISION_STALE_S"]
        new_frame = cam_live and cam_res.seq != self._last_cam_seq
        if cam_res is not None:
            self._last_cam_seq = cam_res.seq

        pillar_colour, pillar_xy, pillar_area = None, None, 0
        if cam_live and cam_res.best is not None and live:
            best = cam_res.best
            pillar_xy = locate_pillar(lidar._ranges, best.bearing_deg,
                                      best.area, p)
            if pillar_xy is not None:
                pillar_colour = best.colour
                pillar_area = best.area

        # The second pillar, located the same way. It steers nothing - the
        # planner decides whether it is the next straight's, and only its
        # COLOUR is ever used - but it has to be put in the lane frame to be
        # judged, so it needs a position like any other sighting.
        sec_colour, sec_xy = None, None
        if cam_live and cam_res.second is not None and live:
            sec = cam_res.second
            sec_xy = locate_pillar(lidar._ranges, sec.bearing_deg, sec.area, p)
            if sec_xy is not None:
                sec_colour = sec.colour

        unknown_xy = unclassified(self._cands, pillar_xy) if live else None

        telem = Telemetry() if self.dry else self.link.telemetry()
        if self.dry:
            # No STM32 on the bench, so nothing would ever report a heading
            # and the FSM would sit blind. Pretend the link is alive so the
            # planner and the states can be exercised, which is the point of
            # this mode.
            telem.stamp = t0
        telem_fresh = telem.fresh(t0, p["TELEM_STALE_S"])

        # A firmware reset puts the parameter table back at compiled-in
        # defaults. boot_id is in every TELEM frame, so we notice within one
        # frame rather than waiting for a ?V round trip.
        if telem_fresh and telem.boot_id:
            note = self.stm.on_boot(None, None, telem.boot_id)
            if note:
                self.emit(note)

        # ---- decide -----------------------------------------------------
        ctx = Ctx(p=p, now=t0, telem=telem, telem_fresh=telem_fresh,
                  lidar_live=live, lidar_dead=dead, new_frame=new_frame or
                  (live and new_rev), new_rev=new_rev,
                  front_mm=front if front is not None else float("inf"),
                  left_mm=left if left is not None else float("inf"),
                  right_mm=right if right is not None else float("inf"),
                  cone_left=self._wall.left_mm, cone_right=self._wall.right_mm,
                  wall_ang=self._wall.yaw_deg, pillar_xy=pillar_xy,
                  pillar_colour=pillar_colour, sec_xy=sec_xy,
                  sec_colour=sec_colour, unknown_xy=unknown_xy,
                  ticks_per_mm=self.stm.ticks_per_mm())
        raw = self.fsm.step(ctx)
        intent = self.mapper.map(raw)

        # ---- send -------------------------------------------------------
        cmd = CMD_NONE
        if self._reboot_frames > 0:
            cmd = CMD_REBOOT
            self._reboot_frames -= 1

        sent = False
        if not self.dry:
            for line in self.stm.drain(int(p["STM_PUSH_PER_LOOP"])):
                self.link.push_params([line])
            sent = self.link.send(
                intent, left_mm=left, front_mm=front, right_mm=right,
                rev=lidar.rev, lidar_ok=live, cam_ok=cam_live,
                pillar_seen=pillar_colour is not None, cmd=cmd)

        # ---- observe ----------------------------------------------------
        queue_depth = lidar.queue_depth()
        tick_log = (t0 - self.last_log) >= 1.0
        if tick_log or not self.quiet:
            self._status(ctx, intent, front, left, right, queue_depth)
        if tick_log:
            self.rev_per_s = lidar.rev - self.last_rev
            self.last_rev = lidar.rev
            self.last_log = t0

        self.last = Tick(
            t=t0, lidar_live=live, cam_live=cam_live, new_frame=new_frame,
            new_rev=new_rev, picks=picks, front=front, left=left, right=right,
            wall=self._wall, pillar_colour=pillar_colour, pillar_xy=pillar_xy,
            pillar_area=pillar_area, sec_colour=sec_colour, sec_xy=sec_xy,
            unknown_xy=unknown_xy,
            n_candidates=len(self._cands), telem=telem,
            telem_fresh=telem_fresh, intent=intent, clamped=self.mapper.clamped,
            sent=sent, rev=lidar.rev, rev_per_s=self.rev_per_s,
            queue_depth=queue_depth, tick_ms=(time.time() - t0) * 1000.0,
            fsm=self.fsm.snapshot())
        return self.last

    def _status(self, ctx, intent, front, left, right, queue_depth):
        f = self.fsm
        pl = f.planner
        fmt = lambda v: "----" if v is None else f"{int(v):4d}"
        cone = lambda v: "----" if v is None else f"{int(v):4d}"
        self.emit(
            f"{f.state.name:<16} c{f.corner_count:02d} | "
            f"F {fmt(front)} L {fmt(left)} R {fmt(right)} | "
            f"cone {cone(self._wall.left_mm)}/{cone(self._wall.right_mm)} "
            f"off {pl.lane_off:+5.0f}{'' if pl.lane_off_ok else '?'} "
            f"along {pl.lane_along:6.0f} | "
            f"hd={ctx.heading:+6.1f} yaw={pl.lat_yaw_cmd:+5.1f} "
            f"tgt={pl.lat_target:+5.0f} | "
            f"trk={len(pl.tracks)} | {intent.mode.name[:4]} {intent.reason}")

    # ------------------------------------------------------- the paced loop

    def run(self, stop_event=None):
        """tick() at SEND_HZ until stop_event is set (forever if None)."""
        while stop_event is None or not stop_event.is_set():
            period = 1.0 / max(1, self.p["SEND_HZ"])
            t = self.tick().t
            dt = time.time() - t
            if dt < period:
                time.sleep(period - dt)


# ================================================================== main ====

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="decide and print, but never open the serial port")
    ap.add_argument("--port", default=None)
    ap.add_argument("--quiet", action="store_true", help="one line per second")
    ap.add_argument("--start", action="store_true",
                    help="request Start as soon as the lidar is live")
    args = ap.parse_args()

    p, stm = prm.PiParams(), prm.StmParams()
    print(prm.load(p, stm))

    cfg = prm.read_config()
    port = args.port or cfg.get("serial", {}).get("port", "/dev/ttyACM0")
    baud = cfg.get("serial", {}).get("baud", 115200)

    shared = SharedState()
    lidar = LidarThread(shared)
    vision = camera.VisionThread(p, shared)

    link = None
    if not args.dry:
        link = Link(port, baud, on_log=lambda s: print("[stm32] " + s),
                    on_param=lambda s: _param_line(stm, s))
        try:
            link.open()
        except Exception as e:
            import glob
            found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
            raise SystemExit(f"[round] cannot open {port}: {e}\n"
                             f"        serial devices present: "
                             f"{', '.join(found) or 'none'}")
        link.start()
        stm.request_dump()

    loop = ObstacleRound(p, stm, shared, lidar, link=link, dry=args.dry,
                         quiet=args.quiet)
    lidar.start()
    vision.start()
    loop.banner()
    print("[round] Ctrl-C to stop.\n")

    if args.start:
        threading.Thread(target=_autostart, args=(loop, lidar),
                         daemon=True).start()

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[round] stopping...")
        loop.stop()
        for _ in range(5):              # hold STOP so the car really halts
            try:
                loop.tick()
            except Exception:
                break
            time.sleep(0.02)
        if link is not None:
            link.stop()
            link.close()
        vision.stop()
        lidar.stop()
        vision.join(timeout=2.0)
        lidar.join(timeout=2.0)
        print("[round] done.")


def _param_line(stm, line):
    note = stm.on_line(line)
    if note:
        print(note)
    if stm.synced and not _param_line.pushed:
        _param_line.pushed = True
        stm.queue_all()
        print("[pi] pushing saved tuning to the STM32")
    if not stm.synced:
        _param_line.pushed = False


_param_line.pushed = False


def _autostart(loop, lidar):
    while not lidar_live(lidar, time.monotonic(), loop.p["LIDAR_STALE_S"]):
        time.sleep(0.1)
    time.sleep(0.5)
    loop.start()


if __name__ == "__main__":
    main()
