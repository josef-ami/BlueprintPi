"""
openRound.py — Open-challenge lidar feeder.

ONE job: stream the three bearing distances (0 deg forward, 90 deg left,
270 deg right) to the STM32 over USB, continuously, for the WRO Open round.
No camera, no fusion, no FSM here — the STM32 owns the driving; this process
is a pure sensor pipe (plus it echoes the STM32's '#' log lines to stdout).

Run standalone, it owns the lidar (LidarThread) itself, so run it INSTEAD of
main.py / dashboard.py — one process may hold the lidar at a time.

dashboard.py can ALSO drive this same wire protocol from inside its own
process (Start/Stop toggle in the UI), reusing the pure functions below
(load_tol, read_three, pack_frame) against the LidarThread it already owns,
instead of duplicating this logic. Do not run openRound.py standalone at the
same time as that dashboard toggle — both would try to open UART_PORT.

Wire frame — one ASCII line per send, SEND_HZ times a second:

    left,front,right,rev\n          e.g.  "412,1873,655,1234\n"

  - left/front/right in mm; 65535 = no valid return within tolerance
    (the STM32 sanitizes that to LIDAR_FAR and treats it as NO EVIDENCE,
    never as "side open").
  - rev = lidar revolution counter. Each bearing gets at most one new sample
    per revolution (~10 Hz on the C1), so the STM32 debounces the turn on
    distinct rev values, not on frames — at 50 Hz, consecutive frames are
    mostly the same measurement resent.

Silence is the failure signal: if the lidar thread has died or no point has
arrived for STALE_S, NOTHING is sent. That keeps the STM32's startup gate
closed until the lidar is really scanning, and lets its stale (200 ms) /
dead (1000 ms) logic engage mid-run instead of acting on frozen values.

Latency: bins are read live from LidarThread (no 20 Hz SharedState hop), so
what's left is the rotation phase (0-100 ms) plus the send period (0-20 ms).

Bearing tolerance (+/- degrees) is read once at startup from
    config.json -> lidar.bearing_tol_deg
which the dashboard slider sets and saves. Re-run to pick up a new saved value.

Serial port: a Blackpill on native USB CDC enumerates as /dev/ttyACM0, a
different namespace from the lidar's /dev/ttyUSB0, so they never collide.
If you ever go through a USB-UART bridge instead, it becomes /dev/ttyUSBn and
can swap order with the lidar across reboots — pin it via /dev/serial/by-id/.
    ls /dev/ttyACM* /dev/ttyUSB*     (or: dmesg | grep tty)

1 Hz status line:
    F/L/R   current distances
    rev/s   should sit near 10. Lower = motor/scan trouble.
    queue   rplidarc1 backlog. Should hover near 0; if it climbs, the
            consumer is falling behind and lag grows the longer it runs.
"""

import json
import math
import os
import time

import serial

from worldstate import SharedState
from sensors.lidar import LidarThread, pick_bearing

UART_PORT = "/dev/ttyACM0"     # Blackpill native USB CDC; NOT ttyUSB0 (lidar)
UART_BAUD = 115200             # ignored by native CDC; must match a bridge
SEND_HZ = 50
STALE_S = 0.3                  # no lidar point for this long -> go silent

BEARINGS = (("front", 0), ("left", 90), ("right", 270))   # robot frame, CCW+
INVALID = 0xFFFF

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config.json")


def load_tol(path=CONFIG_PATH, default=2):
    """+/- bearing tolerance in whole degrees, from lidar.bearing_tol_deg."""
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        return max(0, int(cfg.get("lidar", {}).get("bearing_tol_deg", default)))
    except (OSError, ValueError, TypeError):
        return default


def _u16(mm):
    if mm is None or math.isinf(mm):
        return INVALID
    v = int(round(mm))
    return INVALID if not (0 <= v < INVALID) else v


def pack_frame(front_mm, left_mm, right_mm, rev):
    """ASCII 'left,front,right,rev\\n' — field order matches the STM32 parser."""
    return (f"{_u16(left_mm)},{_u16(front_mm)},{_u16(right_mm)},{rev}\n"
            .encode("ascii"))


def read_three(ranges, quals, tol):
    """(front_mm, left_mm, right_mm) — each mm or None (no valid return)."""
    out = []
    for _, target in BEARINGS:
        p = pick_bearing(ranges, quals, target, tol)
        out.append(None if p is None else p[1])
    return tuple(out)


def lidar_live(lidar, now):
    return lidar.is_alive() and (now - lidar.last_point_t) < STALE_S


def main():
    tol = load_tol()
    period = 1.0 / SEND_HZ

    try:
        ser = serial.Serial(UART_PORT, UART_BAUD, timeout=0)
    except serial.SerialException as e:
        import glob
        found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
        hint = ", ".join(found) if found else "none found"
        raise SystemExit(f"[openRound] cannot open {UART_PORT}: {e}\n"
                         f"           USB serial devices present: {hint}")

    shared = SharedState()          # LidarThread still publishes; unused here
    lidar = LidarThread(shared)
    lidar.start()

    print(f"[openRound] 0/90/270 deg -> {UART_PORT}, {SEND_HZ} Hz, "
          f"tol=+/-{tol} deg, silent when lidar stale > {STALE_S}s. Ctrl-C to stop.")

    last_log = time.monotonic()
    last_rev = 0
    was_live = False
    f = l = r = None
    rx_buf = b""

    try:
        while True:
            now = time.monotonic()
            live = lidar_live(lidar, now)

            if live:
                # live bins, not the SharedState snapshot: skips a 0-50 ms hop.
                # Element reads are atomic under the GIL; a mixed-age window
                # of 1-degree bins is fine for a nearest-return pick.
                f, l, r = read_three(lidar._ranges, lidar._quals, tol)
                ser.write(pack_frame(f, l, r, lidar.rev))
            if live != was_live:
                print("[openRound] lidar LIVE - streaming" if live else
                      "[openRound] lidar STALE - silent (STM32 watchdog will trip)")
                was_live = live

            # echo the STM32's log lines (its '#' state-transition prints)
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
                revs = lidar.rev - last_rev
                print(f"[openRound] F {fmt(f)}  L {fmt(l)}  R {fmt(r)} mm | "
                      f"rev/s {revs:2d}  queue {lidar.queue_depth()}"
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