#!/usr/bin/env python3
"""
obstacle_lap.py — Pi entry point for the WRO obstacle round, LAP ONLY.
(No parking, no magenta, no start button, no rear ToF.)

    CameraThread ─┐
                  ├─> SharedState ─> fuse() ─> AvoidSupervisor ─┐
    LidarThread  ─┘         │                                   │
                            └─> left/front/right bearings ──────┤
                                                                v
    STM32  <──  PERCEPT frame  <──────────────────────  PerceptLink  ──> TELEM

WHO DOES WHAT
    Pi     perception, pillar tracking, the avoidance solve.
    STM32  the state machine, the heading PID, the 90 deg arc, odometry, the
           lidar/link staleness logic. It is OpenRound.cpp with one state added,
           so everything already calibrated for the open round still runs the car.

The loop itself lives in class ObstacleLap, so dashboard.py runs exactly this
code — same perception reads, same supervisor, same frames, same printed lines —
on the camera and lidar it already owns. main() below is the CLI around it and
prints exactly what it always has.

Run this INSTEAD of main.py / openRound.py / dashboard.py — one process may
hold the camera and the lidar at a time.

    python3 obstacle_lap.py              # normal run
    python3 obstacle_lap.py --dry        # no serial: solve and print, send nothing
    python3 obstacle_lap.py --no-avoid   # send perception only; car runs an open lap
"""

import argparse
import threading
import time
from dataclasses import dataclass

from worldstate import SharedState, Obstacle
from sensors.camera import CameraThread, load_config
from sensors.lidar import LidarThread, pick_bearing
from main import fuse_with

from control.percept_link import (PerceptLink, Telemetry, ST_HEADING, ST_BOOT,
                                  ST_FINISH, ST_STOPPED, CMD_NONE, CMD_RERUN,
                                  CMD_STOP, CMD_NAMES)
from control.solver import AvoidCfg, AVOID_NONE, AVOID_TRACK, AVOID_COMMIT
from control.supervisor import AvoidSupervisor

SEND_HZ = 50
UART_PORT = "/dev/ttyACM0"      # Blackpill native USB CDC. NOT ttyUSB0 (lidar).
UART_BAUD = 115200              # ignored by native CDC; matters only for a bridge
LIDAR_STALE_S = 0.3             # no point for this long -> the lidar is not scanning
CAMERA_STALE_S = 0.5
CMD_TIMEOUT_S = 3.0             # a held command the STM32 never acknowledges is dropped

# Base (straight) speed in PWM, sent to the STM32 in every PERCEPT frame. The
# firmware clamps to its own [SPEED_MIN, SPEED_MAX] band and scales the avoid
# speed by the same fraction; these must match ObstacleLap.cpp so the dashboard
# slider's ends line up with what the firmware will actually accept.
DEFAULT_BASE_SPEED = 70
SPEED_MIN = 40
SPEED_MAX = 150

BEARINGS = (("front", 0), ("left", 90), ("right", 270))   # robot frame, CCW+


def read_three(ranges, quals, tol):
    """(front, left, right) in mm, each a float or None (no valid return)."""
    out = []
    for _, target in BEARINGS:
        p = pick_bearing(ranges, quals, target, tol)
        out.append(None if p is None else p[1])
    return tuple(out)


def read_three_picks(ranges, quals, tol):
    """read_three, keeping each whole pick_bearing result (deg, mm, quality)."""
    return tuple(pick_bearing(ranges, quals, target, tol) for _, target in BEARINGS)


# ============================================================ one pass ====

@dataclass
class Tick:
    """Everything one pass of the loop saw, decided and sent."""
    t: float                    # time.time() at the top of the pass
    lidar_live: bool
    cam_live: bool
    picks: tuple                # (front, left, right): (deg, mm, q) or None
    front: object               # mm or None
    left: object
    right: object
    cam_res: object             # the CameraResult fused this pass (or None)
    lidar_res: object           # the LidarResult fused this pass (or None)
    obstacles: list             # fused obstacles (copies)
    near_idx: int               # the supervisor's pick in obstacles, -1 = none
    telem: Telemetry
    status: object              # FsmStatus, or None (dry / older firmware)
    action: int
    color: str
    heading: float
    leg: float
    note: str
    sent: bool                  # a PERCEPT frame went out this pass
    hello_sent: bool
    cmd: int                    # CMD byte carried this pass
    rev: int
    rev_per_s: int              # lidar revolutions in the last full second
    queue_depth: int
    link_state: str             # "dry" | "ok" | "STALE"
    sup: dict                   # AvoidSupervisor.snapshot()
    tick_ms: float              # how long this pass took to compute


class ObstacleLap:
    """
    The obstacle-lap control loop, one tick() per 1/SEND_HZ. Reads live lidar
    bins and the SharedState camera result, fuses, runs the supervisor, sends
    one PERCEPT frame, and emits the same lines the CLI always printed.

    cfg        config.json as a dict, read once by the caller
    shared     SharedState the camera (and lidar) publish into
    lidar      the running LidarThread (its live bins are read directly)
    link       an opened, started PerceptLink; ignored when dry
    emit       where printed lines go (print, for the CLI)
    """

    def __init__(self, cfg, shared, lidar, link=None, dry=False, no_avoid=False,
                 quiet=False, port=UART_PORT, emit=print):
        self.cfg = cfg
        self.acfg = AvoidCfg.from_config(cfg)
        self.tol = max(0, int(cfg.get("lidar", {}).get("bearing_tol_deg", 2)))
        fcfg = cfg.get("fusion", {})
        self.fusion = (fcfg.get("bearing_match_deg", 8),
                       fcfg.get("range_floor_mm", 0.0),
                       fcfg.get("gap_split_mm", 0.0))
        self.sup = AvoidSupervisor(
            self.acfg,
            confirm_ticks=int(cfg.get("avoid", {}).get("confirm_ticks", 3)),
            refractory_mm=float(cfg.get("avoid", {}).get("refractory_mm", 150.0)))
        self.shared = shared
        self.lidar = lidar
        self.link = link
        self.dry = dry
        self.no_avoid = no_avoid
        self.quiet = quiet
        self.port = port
        self.emit = emit

        self.hello_sent = False
        self.last_log = 0.0
        self.last_rev = 0
        self.last_state = -1
        self.last_action = AVOID_NONE
        self.rev_per_s = 0
        self.last = None                    # the newest Tick

        self._cmd_lock = threading.Lock()
        self._cmd = CMD_NONE
        self._cmd_t0 = 0.0

        # Base (straight) speed sent every frame; read config, else the default.
        self._speed = int(cfg.get("run", {}).get("base_speed", DEFAULT_BASE_SPEED))
        self._speed = max(SPEED_MIN, min(SPEED_MAX, self._speed))

    # ---- straight-line speed (PERCEPT byte 16) ----

    def set_speed(self, pwm):
        """Set the base (straight) speed the STM32 runs, clamped to the band the
        firmware accepts. Applied live: the next frame carries it, and the
        firmware re-asserts it every tick, so it takes effect mid-run."""
        v = max(SPEED_MIN, min(SPEED_MAX, int(pwm)))
        self._speed = v
        return v

    @property
    def speed(self):
        return self._speed

    # ---- the banner the CLI has always printed first ----

    def banner(self):
        self.emit(f"[lap] obstacle lap. send={SEND_HZ} Hz  tol=+/-{self.tol} deg  "
                  f"d_clear={self.acfg.clearance_mm:.0f} mm  "
                  f"engage={self.acfg.engage_mm:.0f}  freeze={self.acfg.freeze_mm:.0f}  "
                  f"link={'DRY' if self.dry else self.port}"
                  f"{'  AVOIDANCE DISABLED' if self.no_avoid else ''}")

    # ---- commands in PERCEPT byte 15 ----

    def request(self, cmd):
        """
        Hold `cmd` (CMD_RERUN or CMD_STOP) in every frame until TELEM shows it
        took effect — RERUN: state BOOT; STOP: STOPPED or FINISH — or until
        CMD_TIMEOUT_S passes. While one is held and the lidar is not live, the
        frame goes out anyway as a command-only frame (LIDAR_OK clear, no ranges,
        no HELLO), so a command never waits on perception.
        """
        if cmd not in (CMD_RERUN, CMD_STOP):
            raise ValueError(f"cannot hold command {cmd!r}")
        if self.dry:
            self.emit(f"[lap] {CMD_NAMES[cmd]} ignored: --dry has no STM32")
            return False
        with self._cmd_lock:
            self._cmd = cmd
            self._cmd_t0 = time.time()
        self.emit(f"[lap] sending {CMD_NAMES[cmd]}")
        return True

    def pending_command(self):
        with self._cmd_lock:
            return self._cmd

    def _command(self, telem, now):
        with self._cmd_lock:
            cmd, t0 = self._cmd, self._cmd_t0
        if cmd == CMD_NONE:
            return CMD_NONE
        done = telem.fresh() and telem.stamp > t0 and (
            (cmd == CMD_RERUN and telem.state == ST_BOOT) or
            (cmd == CMD_STOP and telem.state in (ST_STOPPED, ST_FINISH)))
        if done:
            self.emit(f"[lap] {CMD_NAMES[cmd]} acknowledged (STM32 {telem.state_name})")
        elif now - t0 > CMD_TIMEOUT_S:
            self.emit(f"[lap] {CMD_NAMES[cmd]} not acknowledged after "
                      f"{CMD_TIMEOUT_S:.0f} s (STM32 {telem.state_name}"
                      f"{'' if telem.fresh() else ', TELEM stale'}) - dropped")
            done = True
        if done:
            with self._cmd_lock:
                if self._cmd == cmd and self._cmd_t0 == t0:
                    self._cmd = CMD_NONE
            return CMD_NONE
        return cmd

    # ---- one pass ----

    def tick(self) -> Tick:
        t0 = time.time()
        mono = time.monotonic()
        lidar = self.lidar

        # ---- perception -----------------------------------------------------
        lidar_live = lidar.is_alive() and (mono - lidar.last_point_t) < LIDAR_STALE_S
        picks = (None, None, None)
        front = left = right = None
        if lidar_live:
            # Live bins, not the 20 Hz SharedState hop: saves 0-50 ms.
            picks = read_three_picks(lidar._ranges, lidar._quals, self.tol)
            front, left, right = (None if p is None else p[1] for p in picks)

        cam_res, lidar_res = self.shared.snapshot()
        cam_live = cam_res is not None and (t0 - cam_res.timestamp) < CAMERA_STALE_S
        obstacles = fuse_with(cam_res, lidar_res, *self.fusion) if cam_live else []

        # In --dry there is no STM32, so nothing would ever leave BOOT and
        # the supervisor would sit idle. Pretend the car is driving so the
        # solver can be exercised on the bench, which is the whole point of
        # this mode.
        telem = (Telemetry(stamp=t0, state=ST_HEADING) if self.dry
                 else self.link.telemetry())
        status = None if self.dry else self.link.status()

        # ---- decide ---------------------------------------------------------
        if self.no_avoid:
            action, color, heading, leg, note = AVOID_NONE, "", 0.0, 0.0, "disabled"
        else:
            action, color, heading, leg, note = self.sup.tick(
                obstacles, telem, t0,
                left if left is not None else float("inf"),
                right if right is not None else float("inf"))
        cmd = CMD_NONE if self.dry else self._command(telem, t0)

        # ---- send -----------------------------------------------------------
        # Silence is the failure signal: with no lidar the STM32 must see
        # its feed go dead and fall back, not act on frozen distances. The
        # one exception is a committed leg, which is odometry-terminated and
        # needs no perception at all. A held command also goes out, but as a
        # command-only frame (no ranges, LIDAR_OK clear), so it never makes a
        # dead lidar look alive.
        sent = False
        if lidar_live or action == AVOID_COMMIT:
            if not self.hello_sent and lidar_live and cam_live:
                self.hello_sent = True
            if not self.dry:
                self.link.send(left_mm=left, front_mm=front, right_mm=right,
                               rev=lidar.rev, lidar_ok=lidar_live, cam_ok=cam_live,
                               hello=self.hello_sent, action=action,
                               green=(color == "GREEN"),
                               target_heading_deg=heading, leg_mm=leg, cmd=cmd,
                               base_speed=self._speed)
                sent = True
        elif cmd != CMD_NONE and not self.dry:
            # rev repeats the lidar's current count so this frame cannot look
            # like a new revolution to the side-open counter.
            self.link.send(left_mm=None, front_mm=None, right_mm=None,
                           rev=lidar.rev, lidar_ok=False, cam_ok=False,
                           hello=False, action=AVOID_NONE, green=False,
                           target_heading_deg=0.0, leg_mm=0, cmd=cmd)
            sent = True

        # ---- observe --------------------------------------------------------
        if action != self.last_action:
            self.emit(f"  [avoid] {_aname(self.last_action)} -> {_aname(action)}  {note}")
            self.last_action = action
        if telem.state != self.last_state:
            self.emit(f"  [stm32] state {telem.state_name}  corner {telem.corner_count}")
            self.last_state = telem.state

        near = self.sup.pick(obstacles)
        queue_depth = lidar.queue_depth()
        link_state = 'dry' if self.dry else ('ok' if telem.fresh() else 'STALE')
        tick_log = (t0 - self.last_log) >= 1.0
        if tick_log or not self.quiet:
            near_s = (f"{near.color[0]}@{near.bearing_deg:+.0f}/"
                      f"{near.distance_mm:.0f}" if near else "-")
            fmt = lambda v: "----" if v is None else f"{int(v):4d}"
            self.emit(f"{telem.state_name:<8} c{telem.corner_count:02d} | "
                      f"F {fmt(front)} L {fmt(left)} R {fmt(right)} | "
                      f"hd={telem.heading_deg:+6.1f} odo={telem.odo_mm:7.0f} | "
                      f"obs={len(obstacles)} {near_s:<14} | "
                      f"{_aname(action)} {heading:+6.1f} {leg:5.0f} | "
                      f"rev/s {lidar.rev - self.last_rev:2d} q{queue_depth} "
                      f"{link_state}"
                      f" | {note}")
        if tick_log:
            self.rev_per_s = lidar.rev - self.last_rev
            self.last_rev = lidar.rev
            self.last_log = t0

        near_idx = next((i for i, o in enumerate(obstacles) if o is near), -1)
        self.last = Tick(
            t=t0, lidar_live=lidar_live, cam_live=cam_live, picks=picks,
            front=front, left=left, right=right,
            cam_res=cam_res, lidar_res=lidar_res,
            obstacles=[Obstacle(o.color, o.bearing_deg, o.distance_mm, o.confidence)
                       for o in obstacles],
            near_idx=near_idx, telem=telem, status=status,
            action=action, color=color, heading=heading, leg=leg, note=note,
            sent=sent, hello_sent=self.hello_sent, cmd=cmd,
            rev=lidar.rev, rev_per_s=self.rev_per_s, queue_depth=queue_depth,
            link_state=link_state, sup=self.sup.snapshot(),
            tick_ms=(time.time() - t0) * 1000.0)
        return self.last

    # ---- the paced loop ----

    def run(self, stop_event=None):
        """tick() at SEND_HZ until stop_event is set (forever if None)."""
        period = 1.0 / SEND_HZ
        while stop_event is None or not stop_event.is_set():
            t = self.tick().t
            dt = time.time() - t
            if dt < period:
                time.sleep(period - dt)


# ================================================================== main ====

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="solve and print, but never open the serial port")
    ap.add_argument("--no-avoid", action="store_true",
                    help="send perception only; the car runs a plain open lap")
    ap.add_argument("--port", default=UART_PORT)
    ap.add_argument("--quiet", action="store_true", help="one line per second")
    args = ap.parse_args()

    cfg = load_config()

    shared = SharedState()
    cam = CameraThread(shared)
    lidar = LidarThread(shared)
    link = PerceptLink(args.port, UART_BAUD)

    if not args.dry:
        try:
            link.open()
        except Exception as e:
            import glob
            found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
            raise SystemExit(f"[lap] cannot open {args.port}: {e}\n"
                             f"      serial devices present: "
                             f"{', '.join(found) or 'none'}")
        link.start()

    lap = ObstacleLap(cfg, shared, lidar, link=link, dry=args.dry,
                      no_avoid=args.no_avoid, quiet=args.quiet, port=args.port)

    cam.start()
    lidar.start()
    lap.banner()
    print("[lap] Ctrl-C to stop.\n")

    try:
        lap.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[lap] stopping...")
        if not args.dry:
            link.stop()
            # The port closes. NOTE: ObstacleLap.cpp has no link-loss motor
            # cut, so a car that is mid-lap carries on under its own logic.
            # dashboard.py's "Stop car" / "End session" send CMD_STOP first.
            link.close()
        cam.stop()
        lidar.stop()
        cam.join(timeout=2.0)
        lidar.join(timeout=2.0)
        print("[lap] done.")


def _aname(a):
    return {AVOID_NONE: "----", AVOID_TRACK: "TRCK", AVOID_COMMIT: "CMIT"}[a]


if __name__ == "__main__":
    main()
