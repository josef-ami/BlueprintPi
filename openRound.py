#!/usr/bin/env python3
"""
openRound.py - Open-challenge LiDAR feeder.

ONE job: stream the three bearing distances (0 deg forward, 90 deg left,
270 deg right) to the STM32 over USB, continuously, for the WRO Open round.
No camera, no planner, no FSM here - the open-round firmware owns the driving
and this process is a pure sensor pipe (plus it echoes the STM32's '#' log
lines to stdout).

This is deliberately NOT the obstacle round. The open round's firmware is
OpenRound.cpp, which drives itself from these three numbers and has been
calibrated that way; nothing about the obstacle round's split - the Pi-side
FSM, the binary DRIVE/TELEM frames, the lane planner - applies to it. Keeping
this file small and separate is what stops one round's changes breaking the
other.

Run standalone; it owns the LiDAR itself, so run it INSTEAD of dashboard.py -
one process may hold the LiDAR at a time.

Wire frame - one ASCII line per send, SEND_HZ times a second:

    left,front,right,rev\\n          e.g.  "412,1873,655,1234\\n"

  - left/front/right in mm; 65535 = no valid return within tolerance (the
    STM32 sanitizes that to LIDAR_FAR and treats it as NO EVIDENCE, never as
    "side open").
  - rev = LiDAR revolution counter. Each bearing gets at most one new sample
    per revolution (~10 Hz on the C1), so the STM32 debounces the turn on
    distinct rev values, not on frames - at 50 Hz, consecutive frames are
    mostly the same measurement resent.

Silence is the failure signal: if the LiDAR thread has died or no point has
arrived for STALE_S, NOTHING is sent. That keeps the STM32's startup gate
closed until the LiDAR is really scanning, and lets its stale / dead logic
engage mid-run instead of acting on frozen values.

Latency: bins are read live from LidarThread (no 20 Hz SharedState hop), so
what is left is the rotation phase (0-100 ms) plus the send period.

The geometry helpers this used to define - read_three, pick_bearing, u16,
lidar_live - moved into sensors/lidar.py, because the obstacle round needs
them too and one copy is better than two.

Serial port: a Blackpill on native USB CDC enumerates as /dev/ttyACM0, a
different namespace from the LiDAR's /dev/ttyUSB0, so they never collide. If
you ever go through a USB-UART bridge instead, it becomes /dev/ttyUSBn and can
swap order with the LiDAR across reboots - pin it via /dev/serial/by-id/.
    ls /dev/ttyACM* /dev/ttyUSB*     (or: dmesg | grep tty)

1 Hz status line:
    F/L/R   current distances
    rev/s   should sit near 10. Lower = motor/scan trouble.
    queue   rplidarc1 backlog. Should hover near 0; if it climbs, the
            consumer is falling behind and lag grows the longer it runs.
"""

import argparse
import time

import serial

import params as prm
from sensors.lidar import LidarThread, lidar_live, read_three, u16
from worldstate import SharedState

SEND_HZ = 50
STALE_S = 0.3                  # no LiDAR point for this long -> go silent


def pack_frame(front_mm, left_mm, right_mm, rev):
    """ASCII 'left,front,right,rev\\n' - field order matches the STM32 parser."""
    return (f"{u16(left_mm)},{u16(front_mm)},{u16(right_mm)},{rev}\n"
            .encode("ascii"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--tol", type=int, default=None,
                    help="bearing tolerance in whole degrees "
                         "(default: BEARING_TOL_DEG from config.json)")
    args = ap.parse_args()

    cfg = prm.read_config()
    p = prm.PiParams()
    p.set_many(cfg.get("params", {}))
    tol = args.tol if args.tol is not None else int(p["BEARING_TOL_DEG"])
    port = args.port or cfg.get("serial", {}).get("port", "/dev/ttyACM0")
    baud = cfg.get("serial", {}).get("baud", 115200)
    period = 1.0 / SEND_HZ

    try:
        ser = serial.Serial(port, baud, timeout=0)
    except serial.SerialException as e:
        import glob
        found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        raise SystemExit(f"[openRound] cannot open {port}: {e}\n"
                         f"           USB serial devices present: "
                         f"{', '.join(found) or 'none found'}")

    shared = SharedState()          # LidarThread still publishes; unused here
    lidar = LidarThread(shared)
    lidar.start()

    print(f"[openRound] 0/90/270 deg -> {port}, {SEND_HZ} Hz, "
          f"tol=+/-{tol} deg, silent when the lidar is stale > {STALE_S}s. "
          f"Ctrl-C to stop.")

    last_log = time.monotonic()
    last_rev = 0
    was_live = False
    f = l = r = None
    rx_buf = b""

    try:
        while True:
            now = time.monotonic()
            live = lidar_live(lidar, now, STALE_S)

            if live:
                # Live bins, not the SharedState snapshot: skips a 0-50 ms hop.
                # Element reads are atomic under the GIL; a mixed-age window of
                # 1-degree bins is fine for a nearest-return pick.
                f, l, r = read_three(lidar._ranges, lidar._quals, tol)
                ser.write(pack_frame(f, l, r, lidar.rev))
            if live != was_live:
                print("[openRound] lidar LIVE - streaming" if live else
                      "[openRound] lidar STALE - silent (the STM32 watchdog "
                      "will trip)")
                was_live = live

            # echo the STM32's '#' state-transition prints
            n = ser.in_waiting
            if n:
                rx_buf += ser.read(n)
                *lines, rx_buf = rx_buf.split(b"\n")
                for line in lines:
                    print("[stm32] " + line.decode("ascii", "replace").rstrip())
                if len(rx_buf) > 512:           # no newline for ages: flush
                    rx_buf = b""

            if now - last_log >= 1.0:
                fmt = lambda v: "----" if v is None else f"{int(v):4d}"
                print(f"[openRound] F {fmt(f)}  L {fmt(l)}  R {fmt(r)} mm | "
                      f"rev/s {lidar.rev - last_rev:2d}  "
                      f"queue {lidar.queue_depth()}"
                      f"{'' if live else '  (STALE)'}")
                last_rev = lidar.rev
                last_log = now

            time.sleep(period)
    except KeyboardInterrupt:
        pass
    except serial.SerialException as e:
        print(f"[openRound] serial error: {e} (STM32 unplugged/reset?)")
    finally:
        lidar.stop()
        lidar.join(timeout=2.0)
        ser.close()
        print("[openRound] stopped")


if __name__ == "__main__":
    main()
