#!/usr/bin/env python3
"""
main_obstacle.py — the obstacle-round entry point.

Wires the whole chain and runs it at a fixed tick:

    CameraThread ─┐
                  ├─> SharedState ─> fuse() ─┐
    LidarThread  ─┘                          ├─> Ctx ─> FSM ─> ActionIntent
    Link (STM32) ──> Telemetry ──────────────┘                      │
                                                                    v
    STM32 <── DRIVE frame <── link.send_intent() <── CommandMapper ──┘

Run this INSTEAD of main.py or dashboard.py — one process may hold the camera
and the lidar at a time.

    python3 main_obstacle.py            # normal run
    python3 main_obstacle.py --dry      # no STM32: FSM runs, nothing is sent
    python3 main_obstacle.py --open-loop  # speed as raw duty, PID not engaged

Bring-up order that avoids chasing ghosts:
    1. --dry first. Watch the state line. No wheels turn.
    2. --open-loop with wheels off the ground. Confirm steering + direction.
    3. Tune the STM32 speed PID (see docs/PI_STM32_PROTOCOL.md), then drop the
       flag and run closed-loop.
"""

import argparse
import signal
import sys
import time

from worldstate import SharedState
from sensors.camera import CameraThread
from sensors.lidar import LidarThread
from main import fuse                     # reuse the existing fusion, unchanged

from control import FSM, Ctx, CommandMapper, Link, ActionIntent

TICK_HZ = 30
UART_PORT = "/dev/ttyACM0"      # Blackpill native USB CDC. NOT ttyUSB0 (lidar).
UART_BAUD = 115200              # ignored by native CDC; matters only for a bridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="run the FSM but never open the serial port")
    ap.add_argument("--open-loop", action="store_true",
                    help="send speed as open-loop duty instead of closed-loop mm/s")
    ap.add_argument("--port", default=UART_PORT)
    ap.add_argument("--quiet", action="store_true", help="one status line per second")
    args = ap.parse_args()

    shared = SharedState()
    cam = CameraThread(shared)
    lidar = LidarThread(shared)
    mapper = CommandMapper()

    def on_transition(old, new):
        print(f"  [FSM] {old.name} -> {new.name}")

    fsm = FSM(on_transition=on_transition)

    link = Link(args.port, UART_BAUD)
    if not args.dry:
        try:
            link.open()
        except Exception as e:
            print(f"[main] cannot open {args.port}: {e}")
            print("       is the STM32 plugged in? try --dry to run without it.")
            sys.exit(1)
        link.start()

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))

    cam.start()
    lidar.start()
    print(f"[main] obstacle round. tick={TICK_HZ} Hz  "
          f"link={'DRY' if args.dry else args.port}  "
          f"speed={'open-loop' if args.open_loop else 'closed-loop'}")
    print("[main] Ctrl-C to stop.\n")

    period = 1.0 / TICK_HZ
    prev = time.time()
    last_log = 0.0

    try:
        while not stop["flag"]:
            t0 = time.time()

            # ---- build this tick's world ----
            cam_res, lidar_res = shared.snapshot()
            obstacles = fuse(cam_res, lidar_res)
            telem = link.telemetry()

            ctx = Ctx(obstacles=obstacles, telem=telem, lidar=lidar_res,
                      now=t0, dt=t0 - prev)
            prev = t0

            # ---- decide ----
            intent = fsm.step(ctx)
            if args.open_loop:
                intent.closed_loop = False

            # ---- act ----
            steer_deg, speed_mmps = mapper.map(intent)
            if not args.dry:
                link.send_intent(intent, steer_deg, speed_mmps)

            # ---- observe ----
            if not args.quiet or (t0 - last_log) >= 1.0:
                last_log = t0
                n = len(obstacles)
                near = ctx.nearest("RED", "GREEN")
                near_s = (f"{near.color[0]}@{near.bearing_deg:+.0f}/"
                          f"{near.distance_mm:.0f}mm" if near else "-")
                link_s = ("dry" if args.dry
                          else ("ok" if telem.fresh() else "STALE"))
                print(f"{fsm.st.state.name:<12} "
                      f"st={steer_deg:+5.1f}d v={speed_mmps:+6.1f} | "
                      f"hd={telem.heading_deg:+6.1f} odo={telem.distance_mm:7.0f} "
                      f"F={telem.tof_front_mm:5.0f} | obs={n} {near_s:<16} "
                      f"c{fsm.st.corner_count:02d} link={link_s} | {intent.reason}")

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        print("\n[main] stopping...")
        if not args.dry:
            link.send_stop()        # explicit all-stop before we let go
            time.sleep(0.05)
            link.stop()
            link.close()
        cam.stop()
        lidar.stop()
        cam.join(timeout=2.0)
        lidar.join(timeout=2.0)
        print("[main] done.")


if __name__ == "__main__":
    main()
