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
import math
import os
import time

import numpy as np

DEFAULT_OUT = "mat_session.npz"
SEND_HZ = 50
STALE_S = 0.3
MIN_RETURNS = 30          # fewer than this is a spin-up / dropout frame

# ---- outlier rejection, done here on the Pi -----------------------------
# Every rule below is PHYSICAL, not a guess about what a reading "should" be.
# That matters: masking by eye already nearly deleted a real traffic sign that
# merely looked like robot structure. A ray is only discarded when the arena
# makes it impossible, or when its own neighbours contradict it.
MIN_MM = 90.0             # inside the car; below the scanner's useful minimum
MAX_MM = 3400.0           # the racetrack is 3000 mm across - a longer return
                          # came through a gap or off a reflection
SPIKE_MM = 250.0          # a ray this far from BOTH neighbours...
AGREE_MM = 120.0          # ...while those neighbours agree with each other


def clean_scan(scan):
    """Drop impossible rays and isolated spikes. Returns (clean, n_dropped).

    Conservative on purpose: a spike is only removed when both neighbours are
    valid AND agree with each other, so a genuine narrow object - which reads
    as several consistent rays, not one - is never touched.
    """
    r = np.asarray(scan, dtype=np.float64).copy()
    n = len(r)
    before = int(np.isfinite(r).sum())

    bad = np.isfinite(r) & ((r < MIN_MM) | (r > MAX_MM))
    r[bad] = np.inf

    ok = np.isfinite(r)
    out = r.copy()
    for i in range(n):
        if not ok[i]:
            continue
        a, b = r[(i - 1) % n], r[(i + 1) % n]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if abs(a - b) <= AGREE_MM and \
                abs(r[i] - a) > SPIKE_MM and abs(r[i] - b) > SPIKE_MM:
            out[i] = np.inf                 # neighbours outvote it
    return out, before - int(np.isfinite(out).sum())


# ------------------------------------------------------------ telemetry
def drain_telem(buf, head, odo, count):
    """Pull complete TELEM frames out of the RX buffer, newest wins.

    Returns (remaining_bytes, n_frames). The frames are removed so the caller
    can treat what is left as ASCII log text.
    """
    from perception.telemetry import SYNC, TELEM_LEN, decode
    got = 0
    while True:
        i = buf.find(SYNC)
        if i < 0 or len(buf) - i < TELEM_LEN:
            break
        frame = bytes(buf[i:i + TELEM_LEN])
        t = decode(frame)
        if t is None:                      # bad checksum: not a frame after all
            buf = buf[:i] + buf[i + 2:]    # drop the false sync, keep the rest
            continue
        buf = buf[:i] + buf[i + TELEM_LEN:]
        head[0] = t["heading_deg"]
        odo[0] = float(t["odo_ticks"])
        count[0] += 1
        got += 1
    return buf, got


# ----------------------------------------------------- wall fitting (step 2)
LEFT_SECTOR = (60, 120)       # bearings that look at the left-hand wall
RIGHT_SECTOR = (240, 300)
MIN_SECTOR_PTS = 8
PARALLEL_TOL_DEG = 12.0


def _fit_line(pts, trims=3):
    """Total-least-squares line through pts, trimmed. Returns (dist, ang, rms).

    dist = perpendicular distance from the sensor to the line
    ang  = direction of the line, degrees
    """
    p = np.asarray(pts, dtype=np.float64)
    for _ in range(trims):
        if len(p) < MIN_SECTOR_PTS:
            return None
        c = p.mean(axis=0)
        u, s_, vt = np.linalg.svd(p - c, full_matrices=False)
        d = vt[0]                       # direction of greatest spread
        nrm = np.array([-d[1], d[0]])   # unit normal
        res = (p - c) @ nrm
        rms = float(np.sqrt(np.mean(res ** 2)))
        keep = np.abs(res) <= max(3.0 * rms, 25.0)
        if keep.all():
            break
        p = p[keep]
    c = p.mean(axis=0)
    u, s_, vt = np.linalg.svd(p - c, full_matrices=False)
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    res = (p - c) @ nrm
    return (abs(float(c @ nrm)), math.degrees(math.atan2(d[1], d[0])),
            float(np.sqrt(np.mean(res ** 2))), len(p))


def _sector_points(scan, lo, hi):
    idx = np.arange(lo, hi + 1) % 360
    r = scan[idx]
    ok = np.isfinite(r)
    a = np.radians(idx[ok].astype(np.float64))
    return np.stack([r[ok] * np.cos(a), r[ok] * np.sin(a)], axis=1)


def fit_walls(path):
    """STEP 2, per-scan: fit the two corridor walls in EVERY scan on its own.

    No pose is needed and none is trusted. Each revolution independently sees a
    wall on its left and one on its right; fitting both and adding the two
    perpendicular distances measures the corridor where the car happens to be.
    Pooling thousands of those measurements says what the corridor really is -
    and whether it is the 1000 mm the rulebook claims and geom.py assumes.
    """
    s = load_session(path)
    scans = s["ranges"]
    if not len(scans):
        print("[fit] no data - record some laps first")
        return
    widths, lefts, rights, rmss, skew = [], [], [], [], []
    for scan in scans:
        L = _fit_line(_sector_points(scan, *LEFT_SECTOR))
        R = _fit_line(_sector_points(scan, *RIGHT_SECTOR))
        if L is None or R is None:
            continue
        dl, al, rl, _nl = L
        dr, ar, rr, _nr = R
        da = abs((al - ar + 90.0) % 180.0 - 90.0)
        if da > PARALLEL_TOL_DEG:          # not a straight: a corner is in view
            continue
        widths.append(dl + dr)
        lefts.append(dl)
        rights.append(dr)
        rmss.append(0.5 * (rl + rr))
        skew.append(da)

    if len(widths) < 20:
        print(f"[fit] only {len(widths)} usable scans - drive more laps")
        return
    w = np.array(widths)
    print(f"\n[fit] {len(scans)} revolutions, {len(w)} usable "
          f"(the rest had a corner in view)")
    print(f"      corridor width : median {np.median(w):7.1f} mm   "
          f"IQR {np.percentile(w,25):.0f}-{np.percentile(w,75):.0f}   "
          f"p10-p90 {np.percentile(w,10):.0f}-{np.percentile(w,90):.0f}")
    print(f"      left / right   : {np.median(lefts):7.1f} / "
          f"{np.median(rights):.1f} mm (median)")
    print(f"      wall flatness  : {np.median(rmss):7.1f} mm rms residual "
          f"(how straight the walls fit)")
    print(f"      wall skew      : {np.median(skew):7.2f} deg between the two")
    assumed = 1000.0
    err = np.median(w) - assumed
    print(f"\n      geom.py assumes CORRIDOR = {assumed:.0f} mm -> measured is "
          f"{err:+.1f} mm {'WIDER' if err > 0 else 'NARROWER'}")
    if abs(err) < 15:
        print("      within +/-15 mm: the rulebook figure is good, keep it.")
    else:
        print("      OUTSIDE +/-15 mm: set CORRIDOR in nav/geom.py to the "
              "measured value before trusting any map-based pose.")


# --------------------------------------------------------------- storage
def load_session(path):
    if not os.path.exists(path):
        return {"ranges": np.empty((0, 360)), "raw": np.empty((0, 360)),
                "t": np.empty(0), "run": np.empty(0, dtype=np.int32),
                "heading": np.empty(0), "ticks": np.empty(0)}
    d = np.load(path)
    out = {k: d[k] for k in ("ranges", "t", "run")}
    # older sessions have no raw copy; fall back to the cleaned one
    out["raw"] = d["raw"] if "raw" in d.files else out["ranges"]
    for k in ("heading", "ticks"):
        out[k] = d[k] if k in d.files else np.full(len(out["t"]), np.nan)
    return out


def save_session(path, old, ranges, raws, ts, heads, ticks, run_id):
    if not ranges:
        print("[cal] nothing new to save")
        return
    new_r = np.asarray(ranges, dtype=np.float64)
    new_w = np.asarray(raws, dtype=np.float64)
    new_t = np.asarray(ts, dtype=np.float64)
    new_run = np.full(len(ranges), run_id, dtype=np.int32)
    cat = lambda o, n: np.vstack([o, n]) if len(o) else n
    np.savez_compressed(
        path,
        ranges=cat(old["ranges"], new_r),      # outliers removed
        raw=cat(old["raw"], new_w),            # untouched, so a filtering
                                               # mistake never costs a re-drive
        t=np.concatenate([old["t"], new_t]) if len(old["t"]) else new_t,
        run=np.concatenate([old["run"], new_run]) if len(old["run"]) else new_run,
        heading=np.concatenate([old["heading"], np.asarray(heads, dtype=np.float64)]),
        ticks=np.concatenate([old["ticks"], np.asarray(ticks, dtype=np.float64)]),
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
              f"{s['t'][m].max() - s['t'][m].min():5.1f}s  (3 laps)")
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
    ap.add_argument("--fit", action="store_true",
                    help="STEP 2: fit the corridor walls from the recorded laps")
    args = ap.parse_args()

    if args.status:
        status(args.out)
        return
    if args.fit:
        fit_walls(args.out)
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

    ranges, raws, ts, heads, ticks = [], [], [], [], []
    n_dropped, n_rejected = [0], [0]
    period = 1.0 / SEND_HZ
    t0 = time.time()
    last_rev = -1
    last_log = time.monotonic()
    rx = b""
    tel_head, tel_odo, tel_count = [np.nan], [np.nan], [0]
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
                    clean, dropped = clean_scan(scan)
                    if np.isfinite(clean).sum() >= MIN_RETURNS:
                        ranges.append(clean)
                        raws.append(scan)
                        ts.append(time.time() - t0)
                        heads.append(tel_head[0])
                        ticks.append(tel_odo[0])
                        n_dropped[0] += dropped
                    else:
                        n_rejected[0] += 1
                else:
                    n_rejected[0] += 1

            if ser is not None:
                n = ser.in_waiting
                if n:
                    rx += ser.read(n)
                    # OpenRound.cpp interleaves binary TELEM frames with its '#'
                    # log lines on this one port. 0x55 0xAA cannot occur in
                    # ASCII, so pull the frames out FIRST and treat only what is
                    # left as text - otherwise the frames print as garbage and
                    # are lost.
                    rx, _got = drain_telem(rx, tel_head, tel_odo, tel_count)
                    *lines, rx = rx.split(b"\n")
                    for ln in lines:
                        txt = ln.decode("ascii", "replace").rstrip()
                        if txt:
                            print("[stm32] " + txt)
                    if len(rx) > 512:
                        rx = b""

            if now - last_log >= 2.0:
                od = "-" if np.isnan(tel_odo[0]) else f"{int(tel_odo[0])}"
                hd = "-" if np.isnan(tel_head[0]) else f"{tel_head[0]:.0f}"
                print(f"[cal] {len(ranges):5d} revs  {time.time() - t0:5.1f}s  "
                      f"telem {tel_count[0]}  hdg {hd}  odo {od}"
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
        if ranges:
            kept = sum(int(np.isfinite(x).sum()) for x in ranges)
            print(f"[cal] outliers: dropped {n_dropped[0]} rays "
                  f"({100.0 * n_dropped[0] / max(1, kept + n_dropped[0]):.1f}%), "
                  f"rejected {n_rejected[0]} whole revolutions")
        save_session(args.out, old, ranges, raws, ts, heads, ticks, run_id)
        print("[cal] next: run it again for another lap, or "
              "`python calibrationmat.py --status` to see the total")


if __name__ == "__main__":
    main()
