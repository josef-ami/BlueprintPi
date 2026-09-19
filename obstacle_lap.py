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
           watchdog. It is OpenRound.cpp with one state added, so everything
           already calibrated for the open round still runs the car.

Run this INSTEAD of main.py / openRound.py / dashboard.py — one process may
hold the camera and the lidar at a time.

    python3 obstacle_lap.py              # normal run
    python3 obstacle_lap.py --dry        # no serial: solve and print, send nothing
    python3 obstacle_lap.py --no-avoid   # send perception only; car runs an open lap
"""

import argparse
import time

from worldstate import SharedState
from sensors.camera import CameraThread, load_config
from sensors.lidar import LidarThread, pick_bearing
from main import fuse

from control.percept_link import PerceptLink, Telemetry, ST_HEADING
from control.solver import AvoidCfg, AVOID_NONE, AVOID_TRACK, AVOID_COMMIT
from control.supervisor import AvoidSupervisor

SEND_HZ = 50
UART_PORT = "/dev/ttyACM0"      # Blackpill native USB CDC. NOT ttyUSB0 (lidar).
UART_BAUD = 115200              # ignored by native CDC; matters only for a bridge
LIDAR_STALE_S = 0.3             # no point for this long -> the lidar is not scanning
CAMERA_STALE_S = 0.5

BEARINGS = (("front", 0), ("left", 90), ("right", 270))   # robot frame, CCW+


def read_three(ranges, quals, tol):
    """(front, left, right) in mm, each a float or None (no valid return)."""
    out = []
    for _, target in BEARINGS:
        p = pick_bearing(ranges, quals, target, tol)
        out.append(None if p is None else p[1])
    return tuple(out)


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
    acfg = AvoidCfg.from_config(cfg)
    tol = max(0, int(cfg.get("lidar", {}).get("bearing_tol_deg", 2)))
    sup = AvoidSupervisor(acfg,
                          confirm_ticks=int(cfg.get("avoid", {}).get("confirm_ticks", 3)),
                          refractory_mm=float(cfg.get("avoid", {}).get("refractory_mm", 150.0)))

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

    cam.start()
    lidar.start()
    print(f"[lap] obstacle lap. send={SEND_HZ} Hz  tol=+/-{tol} deg  "
          f"d_clear={acfg.clearance_mm:.0f} mm  "
          f"engage={acfg.engage_mm:.0f}  freeze={acfg.freeze_mm:.0f}  "
          f"link={'DRY' if args.dry else args.port}"
          f"{'  AVOIDANCE DISABLED' if args.no_avoid else ''}")
    print("[lap] Ctrl-C to stop.\n")

    period = 1.0 / SEND_HZ
    hello_sent = False
    last_log = 0.0
    last_rev = 0
    last_state = -1
    last_action = AVOID_NONE

    try:
        while True:
            t0 = time.time()
            mono = time.monotonic()

            # ---- perception -------------------------------------------------
            lidar_live = lidar.is_alive() and (mono - lidar.last_point_t) < LIDAR_STALE_S
            front = left = right = None
            if lidar_live:
                # Live bins, not the 20 Hz SharedState hop: saves 0-50 ms.
                front, left, right = read_three(lidar._ranges, lidar._quals, tol)

            cam_res, lidar_res = shared.snapshot()
            cam_live = cam_res is not None and (t0 - cam_res.timestamp) < CAMERA_STALE_S
            obstacles = fuse(cam_res, lidar_res) if cam_live else []

            # In --dry there is no STM32, so nothing would ever leave BOOT and
            # the supervisor would sit idle. Pretend the car is driving so the
            # solver can be exercised on the bench, which is the whole point of
            # this mode.
            telem = (Telemetry(stamp=t0, state=ST_HEADING) if args.dry
                     else link.telemetry())

            # ---- decide -----------------------------------------------------
            if args.no_avoid:
                action, color, heading, leg, note = AVOID_NONE, "", 0.0, 0.0, "disabled"
            else:
                action, color, heading, leg, note = sup.tick(
                    obstacles, telem, t0,
                    left if left is not None else float("inf"),
                    right if right is not None else float("inf"))

            # ---- send -------------------------------------------------------
            # Silence is the failure signal: with no lidar the STM32 must see
            # its feed go dead and fall back, not act on frozen distances. The
            # one exception is a committed leg, which is odometry-terminated and
            # needs no perception at all.
            if lidar_live or action == AVOID_COMMIT:
                if not hello_sent and lidar_live and cam_live:
                    hello_sent = True
                if not args.dry:
                    link.send(left_mm=left, front_mm=front, right_mm=right,
                              rev=lidar.rev, lidar_ok=lidar_live, cam_ok=cam_live,
                              hello=hello_sent, action=action,
                              green=(color == "GREEN"),
                              target_heading_deg=heading, leg_mm=leg)

            # ---- observe ----------------------------------------------------
            if action != last_action:
                print(f"  [avoid] {_aname(last_action)} -> {_aname(action)}  {note}")
                last_action = action
            if telem.state != last_state:
                print(f"  [stm32] state {telem.state_name}  corner {telem.corner_count}")
                last_state = telem.state

            tick_log = (t0 - last_log) >= 1.0
            if tick_log or not args.quiet:
                near = sup.pick(obstacles)
                near_s = (f"{near.color[0]}@{near.bearing_deg:+.0f}/"
                          f"{near.distance_mm:.0f}" if near else "-")
                fmt = lambda v: "----" if v is None else f"{int(v):4d}"
                print(f"{telem.state_name:<8} c{telem.corner_count:02d} | "
                      f"F {fmt(front)} L {fmt(left)} R {fmt(right)} | "
                      f"hd={telem.heading_deg:+6.1f} odo={telem.odo_mm:7.0f} | "
                      f"obs={len(obstacles)} {near_s:<14} | "
                      f"{_aname(action)} {heading:+6.1f} {leg:5.0f} | "
                      f"rev/s {lidar.rev - last_rev:2d} q{lidar.queue_depth()} "
                      f"{'dry' if args.dry else ('ok' if telem.fresh() else 'STALE')}"
                      f" | {note}")
            if tick_log:
                last_rev = lidar.rev
                last_log = t0

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[lap] stopping...")
        if not args.dry:
            link.stop()
            link.close()      # port closes -> STM32 watchdog cuts the motor
        cam.stop()
        lidar.stop()
        cam.join(timeout=2.0)
        lidar.join(timeout=2.0)
        print("[lap] done.")


def _aname(a):
    return {AVOID_NONE: "----", AVOID_TRACK: "TRCK", AVOID_COMMIT: "CMIT"}[a]


if __name__ == "__main__":
    main()
