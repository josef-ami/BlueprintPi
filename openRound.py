"""
openRound.py — Open-challenge lidar feeder.

ONE job: stream the three bearing distances (0 deg forward, 90 deg left,
270 deg right) to the STM32 over the Pi's hardware UART, continuously, for the
WRO Open round. No camera, no fusion, no FSM here — the STM32 owns the driving;
this process is a pure sensor pipe.

It owns the lidar (LidarThread), so run it INSTEAD of main.py / dashboard.py —
one process may hold the lidar at a time.

Wire frame — 9 bytes, sent every 1/SEND_HZ seconds:

    byte 0..1 : sync   0xAA 0x55
    byte 2..3 : front  uint16 little-endian, mm
    byte 4..5 : left   uint16 little-endian, mm
    byte 6..7 : right  uint16 little-endian, mm
    byte 8    : xor8   XOR of bytes 2..7

  - distance is in millimetres; 0xFFFF (65535) means "no valid return within
    tolerance" (also sent for every field when the lidar isn't publishing yet,
    so the STM32 keeps getting a steady heartbeat and can trip its watchdog on
    silence rather than on stale data).
  - xor8 lets the STM32 reject partial/corrupt frames; resync on 0xAA 0x55.

Bearing tolerance (+/- degrees) is read once at startup from
    config.json -> lidar.bearing_tol_deg
which the dashboard slider sets and saves. Re-run to pick up a new saved value.

Note: the STM32 reaches the Pi over USB. A Blackpill running native USB CDC
enumerates as /dev/ttyACM0 — a different namespace from the lidar's
/dev/ttyUSB0 (USB-UART adapter), so the two never collide. If instead you reach
the STM32 through a USB-UART bridge on PA9/PA10, it's /dev/ttyUSB1 (ttyUSB0 is
the lidar) and the two can swap order across reboots — pin it with a
/dev/serial/by-id/... path. Confirm the name with:
    ls /dev/ttyACM* /dev/ttyUSB*     (or: dmesg | grep tty)
"""

import json
import math
import os
import struct
import time

import serial

from worldstate import SharedState
from sensors.lidar import LidarThread, pick_bearing

UART_PORT = "/dev/ttyACM0"     # Blackpill native USB CDC (see Note); NOT ttyUSB0
UART_BAUD = 115200             # ignored by native USB CDC; must match a bridge
SEND_HZ = 50                   # steady cadence for the STM32 watchdog

BEARINGS = (("front", 0), ("left", 90), ("right", 270))   # robot frame
SYNC = b"\xAA\x55"
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


def pack_frame(front_mm, left_mm, right_mm):
    payload = struct.pack("<HHH", _u16(front_mm), _u16(left_mm), _u16(right_mm))
    xor = 0
    for b in payload:
        xor ^= b
    return SYNC + payload + bytes([xor])


def read_three(lidar_result, tol):
    """(front_mm, left_mm, right_mm) — each mm or None (no valid return)."""
    if lidar_result is None:
        return None, None, None
    ranges = lidar_result.ranges
    quals = getattr(lidar_result, "qualities", [0] * 360)
    out = []
    for _, target in BEARINGS:
        p = pick_bearing(ranges, quals, target, tol)
        out.append(None if p is None else p[1])
    return tuple(out)


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

    shared = SharedState()
    lidar = LidarThread(shared)
    lidar.start()

    print(f"[openRound] streaming 0/90/270 deg -> {UART_PORT} @ {UART_BAUD} "
          f"baud, {SEND_HZ} Hz, tol=+/-{tol} deg. Ctrl-C to stop.")

    last_log = 0.0
    try:
        while True:
            _, lidar_result = shared.snapshot()
            f, l, r = read_three(lidar_result, tol)
            ser.write(pack_frame(f, l, r))

            now = time.time()
            if now - last_log >= 1.0:        # 1 Hz sanity line; safe to remove
                fmt = lambda v: "----" if v is None else f"{int(v):4d}"
                print(f"[openRound] F {fmt(f)}  L {fmt(l)}  R {fmt(r)}  (mm)")
                last_log = now

            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        lidar.stop()
        lidar.join(timeout=2.0)
        ser.close()
        print("[openRound] stopped")


if __name__ == "__main__":
    main()
