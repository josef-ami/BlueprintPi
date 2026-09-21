"""
calibrationmat.py - build the REAL arena map by measuring it.

STEP 1: accumulate. Drive the open round as many times as you like; every LiDAR
revolution is stored raw. Later steps fit the walls to that cloud and capture
the true traffic-sign positions, so the map stops being my assumptions and
becomes your mat.

WHY THIS FILE DRIVES THE CAR ITSELF
    openRound.py already owns BOTH the STM32 serial port and the LiDAR, and a
    serial port cannot be shared - so a separate recorder process cannot run
    alongside it. This file therefore reproduces openRound.py's loop exactly:
    the same read_three() bearings, the same 'left,front,right,rev' frame at
    the same 50 Hz, so the STM32 (flashed with OpenRound.cpp) drives precisely
    as it normally does. The only addition is that each new revolution is also
    kept.

    Keep the mat EMPTY for this step. Signs come later, deliberately, one at a
    time - guessing where they go is the mistake this whole exercise exists to
    undo.

USAGE (on the Pi)
    python calibrationmat.py                     # drive + record, appends
    python calibrationmat.py --secs 120          # stop after 2 minutes
    python calibrationmat.py --no-drive          # record only (push it by hand)
    python calibrationmat.py --status            # what is in the file so far

Run it ~20 times, or fewer long runs; it APPENDS to mat_session.npz each time,
so the cloud keeps growing. Ctrl-C stops a run cleanly and still saves.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

DEFAULT_OUT = "mat_session.npz"
SEND_HZ = 50
STALE_S = 0.3
MIN_RETURNS = 30          # fewer than this is a spin-up / dropout frame


# --------------------------------------------------------------- storage
def load_session(path):
    if not os.path.exists(path):
        return {"ranges": np.empty((0, 360)), "t": np.empty(0),
                "run": np.empty(0, dtype=np.int32)}
    d = np.load(path)
    return {k: d[k] for k in ("ranges", "t", "run")}


def save_session(path, old, ranges, ts, run_id):
    if not ranges:
        print("[cal] nothing new to save")
        return
    new_r = np.asarray(ranges, dtype=np.float64)
    new_t = np.asarray(ts, dtype=np.float64)
    new_run = np.full(len(ranges), run_id, dtype=np.int32)
    np.savez_compressed(
        path,
        ranges=np.vstack([old["ranges"], new_r]) if len(old["ranges"]) else new_r,
        t=np.concatenate([old["t"], new_t]) if len(old["t"]) else new_t,
        run=np.concatenate([old["run"], new_run]) if len(old["run"]) else new_run,
    )
    total = len(old["ranges"]) + len(new_r)
    print(f"[cal] saved {len(new_r)} new revolutions (run {run_id}) -> {path}")
    print(f"[cal] session now holds {total} revolutions "
          f"across {run_id + 1} run(s)")


def status(path):
    if not os.path.exists(path):
        print(f"[cal] no session yet at {path}")
        return
    s = load_session(path)
    runs = np.unique(s["run"])
    print(f"[cal] {path}: {len(s['ranges'])} revolutions, {len(runs)} run(s)")
    for r in runs:
        m = s["run"] == r
        print(f"      run {int(r):2d}: {int(m.sum()):5d} revs, "
              f"{s['t'][m].max() - s['t'][m].min():5.1f}s")
    fin = np.isfinite(s["ranges"]).sum(axis=1)
    print(f"      returns per rev: mean {fin.mean():.0f}, "
          f"min {fin.min()}, max {fin.max()}")


# ----------------------------------------------------------------- record
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--secs", type=float, default=0.0,
                    help="stop after this many seconds (0 = until Ctrl-C)")
    ap.add_argument("--no-drive", action="store_true",
                    help="do not open the STM32 / do not drive; record only")
    ap.add_argument("--port", default=None)
    ap.add_argument("--tol", type=int, default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status(args.out)
        return

    old = load_session(args.out)
    run_id = int(old["run"].max()) + 1 if len(old["run"]) else 0

    from worldstate import SharedState
    from sensors.lidar import LidarThread, lidar_live, read_three, u16

    ser = None
    tol = args.tol
    if not args.no_drive:
        import serial
        import params as prm
        cfg = prm.read_config()
        p = prm.PiParams()
        p.set_many(cfg.get("params", {}))
        if tol is None:
            tol = int(p["BEARING_TOL_DEG"])
        port = args.port or cfg.get("serial", {}).get("port", "/dev/ttyACM0")
        baud = cfg.get("serial", {}).get("baud", 115200)
        try:
            ser = serial.Serial(port, baud, timeout=0)
        except serial.SerialException as e:
            import glob
            found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
            raise SystemExit(f"[cal] cannot open {port}: {e}\n"
                             f"      USB serial present: {', '.join(found) or 'none'}")
        print(f"[cal] driving via {port} at {SEND_HZ} Hz (same frames as openRound)")
    else:
        tol = tol if tol is not None else 2
        print("[cal] --no-drive: recording only, the car will not be commanded")

    shared = SharedState()
    lidar = LidarThread(shared)
    lidar.start()

    print(f"[cal] run {run_id} -> {args.out}. KEEP THE MAT EMPTY. Ctrl-C to stop.")

    ranges, ts = [], []
    period = 1.0 / SEND_HZ
    t0 = time.time()
    last_rev = -1
    last_log = time.monotonic()
    rx = b""
    try:
        while True:
            if args.secs and (time.time() - t0) >= args.secs:
                break
            now = time.monotonic()
            live = lidar_live(lidar, now, STALE_S)

            if live and ser is not None:
                f, l, r = read_three(lidar._ranges, lidar._quals, tol)
                ser.write(f"{u16(l)},{u16(f)},{u16(r)},{lidar.rev}\n".encode("ascii"))

            # capture each new revolution, whole scan, unfiltered
            if lidar.rev != last_rev:
                last_rev = lidar.rev
                scan = np.asarray(list(lidar._ranges), dtype=np.float64)
                if np.isfinite(scan).sum() >= MIN_RETURNS:
                    ranges.append(scan)
                    ts.append(time.time() - t0)

            if ser is not None:
                n = ser.in_waiting
                if n:
                    rx += ser.read(n)
                    *lines, rx = rx.split(b"\n")
                    for ln in lines:
                        print("[stm32] " + ln.decode("ascii", "replace").rstrip())
                    if len(rx) > 512:
                        rx = b""

            if now - last_log >= 2.0:
                print(f"[cal] {len(ranges):5d} revs captured  "
                      f"{time.time() - t0:5.1f}s"
                      f"{'' if live else '   (LIDAR STALE)'}")
                last_log = now

            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[cal] stopping")
    finally:
        lidar.stop()
        lidar.join(timeout=2.0)
        if ser is not None:
            try:
                ser.write(b"0,0,0,0\n")      # let the watchdog stop the car
                ser.close()
            except Exception:                # noqa: BLE001
                pass
        save_session(args.out, old, ranges, ts, run_id)
        print("[cal] next: run it again for another lap, or "
              "`python calibrationmat.py --status` to see the total")


if __name__ == "__main__":
    main()
