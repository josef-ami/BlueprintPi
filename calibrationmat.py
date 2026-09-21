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


# ------------------------------------------------------- drawing the map
TICKS_PER_MM = 1.4853          # matches perception/nav.py


def deadreckon(heading_deg, ticks):
    """Path from the IMU heading and the cumulative encoder.

    Heading is ABSOLUTE from the IMU, so it does not accumulate error the way
    an integrated turn rate would; only the along-track distance drifts. Over
    three laps that is good enough to see the arena's shape, which is all this
    has to do - it is a drawing, not a pose estimate.
    """
    n = len(ticks)
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(1, n):
        if not (np.isfinite(ticks[i]) and np.isfinite(ticks[i - 1])
                and np.isfinite(heading_deg[i])):
            x[i], y[i] = x[i - 1], y[i - 1]
            continue
        d = (ticks[i] - ticks[i - 1]) / TICKS_PER_MM
        if abs(d) > 200.0:                 # a reset or a dropped frame
            d = 0.0
        th = math.radians(heading_deg[i])
        x[i] = x[i - 1] + d * math.cos(th)
        y[i] = y[i - 1] + d * math.sin(th)
    return x, y


def draw_map(path, out="mat_map.png", stride=2, max_scans=1200):
    """Stitch the recorded scans into one picture and fit lines to it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = load_session(path)
    R, H, K, run = s["ranges"], s["heading"], s["ticks"], s["run"]
    have = np.isfinite(H) & np.isfinite(K)
    print(f"[draw] {len(R)} revolutions, {int(have.sum())} with odometry")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), facecolor="#0e1116")
    for ax in axes:
        ax.set_facecolor("#0b0e13")
        ax.tick_params(colors="#8b93a1")
        for sp in ax.spines.values():
            sp.set_color("#2a2f3a")
        ax.set_aspect("equal")

    # ---- left: one scan, in the sensor frame, with the two walls fitted ----
    ax = axes[0]
    mid = len(R) // 2
    scan = R[mid]
    pts = []
    idx = np.arange(360)
    ok = np.isfinite(scan)
    a = np.radians(idx[ok].astype(np.float64))
    px, py = scan[ok] * np.cos(a), scan[ok] * np.sin(a)
    ax.scatter(px, py, s=6, c="#4aa3ff", label="scan")
    ax.scatter([0], [0], s=90, c="#ffd23f", marker="^", label="LiDAR")
    for (lo, hi), col, nm in ((LEFT_SECTOR, "#30a46c", "left wall"),
                              (RIGHT_SECTOR, "#e5484d", "right wall")):
        f = _fit_line(_sector_points(scan, lo, hi))
        if f is None:
            continue
        dist, ang, rms, _n = f
        t = np.linspace(-900, 900, 2)
        nx, ny = -math.sin(math.radians(ang)), math.cos(math.radians(ang))
        # the fitted line passes at perpendicular distance `dist`; pick the
        # sign that puts it on the same side as the points it came from
        sp_ = _sector_points(scan, lo, hi)
        sgn = 1.0 if (sp_ @ np.array([nx, ny])).mean() > 0 else -1.0
        cx, cy = sgn * dist * nx, sgn * dist * ny
        dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        ax.plot(cx + t * dx, cy + t * dy, c=col, lw=2,
                label=f"{nm} {dist:.0f}mm rms{rms:.0f}")
    ax.set_title("one revolution, sensor frame", color="#e6e9ee")
    ax.legend(facecolor="#171b22", labelcolor="#e6e9ee", fontsize=8)

    # ---- right: every scan stitched by dead reckoning ----
    ax = axes[1]
    if have.sum() > 50:
        X, Y = deadreckon(H, K)
        sel = np.where(have)[0][:max_scans]
        allx, ally = [], []
        for i in sel:
            sc = R[i]
            o = np.isfinite(sc)
            ii = idx[o][::stride]
            rr = sc[o][::stride]
            th = math.radians(H[i])
            aa = np.radians(ii.astype(np.float64)) + th
            allx.append(X[i] + rr * np.cos(aa))
            ally.append(Y[i] + rr * np.sin(aa))
        allx = np.concatenate(allx)
        ally = np.concatenate(ally)
        ax.scatter(allx, ally, s=1, c="#4aa3ff", alpha=0.25)
        ax.plot(X[sel], Y[sel], c="#ffd23f", lw=1.2, label="path (dead reckoned)")
        ax.legend(facecolor="#171b22", labelcolor="#e6e9ee", fontsize=8)
        ax.set_title(f"{len(sel)} revolutions stitched by IMU heading + encoder",
                     color="#e6e9ee")
    else:
        ax.text(0.5, 0.5, "no odometry in this session\n"
                          "(flash firmware/OpenRound.cpp and re-record)",
                ha="center", va="center", color="#8b93a1", transform=ax.transAxes)
        ax.set_title("stitched map", color="#e6e9ee")

    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="#0e1116")
    print(f"[draw] wrote {out}")


# ------------------------------------------- build the map, pixel by pixel
GRID_MM = 10.0                 # one pixel
GRID_HALF = 2000.0             # half-extent of the canvas


def lap_segments(heading, ticks, run):
    """Split the session into individual laps.

    Heading is absolute from the IMU, so unwrapping it and counting whole turns
    gives lap boundaries directly - no map and no pose needed. Runs are kept
    apart because the encoder restarts and the car is re-placed between them.
    """
    segs = []
    for r in np.unique(run):
        idx = np.where((run == r) & np.isfinite(heading) & np.isfinite(ticks))[0]
        if len(idx) < 50:
            continue
        h = np.unwrap(np.radians(heading[idx]))
        turns = (h - h[0]) / (2 * math.pi)
        for lap in range(int(abs(turns).max()) + 1):
            m = (np.abs(turns) >= lap) & (np.abs(turns) < lap + 1)
            if m.sum() >= 50:
                segs.append(idx[m])
    return segs


def _min_area_rect(P):
    """Rotation and centre of the tightest rectangle around P.

    The arena IS a rectangle, so the orientation that minimises the bounding
    box is its orientation, and that box's centre is its centre. This is what
    lets separately-drifted laps be brought onto each other without knowing
    where any of them actually started.
    """
    best = None
    for deg in np.arange(0.0, 90.0, 0.5):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        x = P[:, 0] * c + P[:, 1] * s
        y = -P[:, 0] * s + P[:, 1] * c
        lo = np.percentile(np.stack([x, y], 1), 1.0, axis=0)
        hi = np.percentile(np.stack([x, y], 1), 99.0, axis=0)
        area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        if best is None or area < best[0]:
            mid = 0.5 * (lo + hi)
            best = (area, deg, mid, hi - lo)
    _a, deg, mid, size = best
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    cx = mid[0] * c - mid[1] * s          # centre back in the original frame
    cy = mid[0] * s + mid[1] * c
    return deg, np.array([cx, cy]), size


def scan_cloud(R, H, X, Y, idx, stride=2):
    """Points of the given revolutions, placed by their pose."""
    out = []
    bins = np.arange(360)
    for i in idx:
        sc = R[i]
        o = np.isfinite(sc)
        ii = bins[o][::stride]
        rr = sc[o][::stride]
        aa = np.radians(ii.astype(np.float64)) + math.radians(H[i])
        out.append(np.stack([X[i] + rr * np.cos(aa),
                             Y[i] + rr * np.sin(aa)], axis=1))
    return np.vstack(out) if out else np.empty((0, 2))


def build_map(path, out="mat_grid.png", npy="mat_grid.npy"):
    """Stitch every lap onto a common centre and accumulate a pixel map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = load_session(path)
    R, H, K, run = s["ranges"], s["heading"], s["ticks"], s["run"]
    segs = lap_segments(H, K, run)
    if not segs:
        print("[build] no laps with odometry - flash firmware/OpenRound.cpp")
        return
    print(f"[build] {len(segs)} laps found")

    n = int(2 * GRID_HALF / GRID_MM)
    grid = np.zeros((n, n), dtype=np.float32)
    kept = 0
    for k, idx in enumerate(segs):
        X, Y = deadreckon(H, K)                  # whole-session path
        P = scan_cloud(R, H, X, Y, idx)
        if len(P) < 500:
            continue
        deg, ctr, size = _min_area_rect(P)
        # normalise: centre it, then rotate its walls onto the axes
        Q = P - ctr
        a = math.radians(-deg)
        c, si = math.cos(a), math.sin(a)
        Q = np.stack([Q[:, 0] * c - Q[:, 1] * si,
                      Q[:, 0] * si + Q[:, 1] * c], axis=1)
        ix = ((Q[:, 0] + GRID_HALF) / GRID_MM).astype(int)
        iy = ((Q[:, 1] + GRID_HALF) / GRID_MM).astype(int)
        m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        np.add.at(grid, (iy[m], ix[m]), 1.0)
        kept += 1
        print(f"   lap {k:2d}: {len(P):6d} pts  rect {size[0]:.0f}x{size[1]:.0f} mm"
              f"  rot {deg:4.1f} deg")
    print(f"[build] {kept} laps stitched")
    np.save(npy, grid)

    # ---- read the geometry back off the accumulated pixels ----
    # The arena is a square annulus, so the Chebyshev radius max(|x|,|y|)
    # collapses BOTH walls onto one axis: the inner block and the outer wall
    # become two peaks. Anything between them is corner smear, which is why
    # the two dominant well-separated peaks are taken rather than an average.
    ys, xs = np.nonzero(grid)
    if len(ys):
        wgt = grid[ys, xs]
        px = (xs * GRID_MM) - GRID_HALF
        py = (ys * GRID_MM) - GRID_HALF
        cheb = np.maximum(np.abs(px), np.abs(py))
        hist, edges = np.histogram(cheb, bins=np.arange(0, 2000, GRID_MM),
                                   weights=wgt)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        inner_i = int(np.argmax(np.where(ctr < 900, hist, 0)))
        outer_i = int(np.argmax(np.where(ctr > 1200, hist, 0)))
        ih, oh = ctr[inner_i], ctr[outer_i]
        print("\n[build] geometry read off the accumulated pixels:")
        print(f"      inner block wall : {ih:7.1f} mm from centre "
              f"-> block {2*ih:.0f} mm")
        print(f"      outer wall       : {oh:7.1f} mm from centre "
              f"-> arena {2*oh:.0f} mm")
        print(f"      corridor width   : {oh - ih:7.1f} mm")
        print(f"\n      geom.py assumes   OUTER 3000 / INNER 1000 / CORRIDOR 1000")
        print(f"      measured          OUTER {2*oh:.0f} / INNER {2*ih:.0f} / "
              f"CORRIDOR {oh-ih:.0f}")
        d = (oh - ih) - 1000.0
        if abs(d) > 25:
            print(f"      corridor is {abs(d):.0f} mm "
                  f"{'WIDER' if d > 0 else 'NARROWER'} than assumed - verify "
                  f"with a tape measure before changing nav/geom.py, since a "
                  f"wrong corridor biases every map-based pose.")

    fig, ax = plt.subplots(figsize=(8.6, 8.6), facecolor="#0e1116")
    ax.set_facecolor("#0b0e13")
    ax.imshow(np.log1p(grid), origin="lower", cmap="magma",
              extent=[-GRID_HALF, GRID_HALF, -GRID_HALF, GRID_HALF])
    ax.set_title(f"{kept} laps stitched on a common centre, "
                 f"{GRID_MM:.0f} mm pixels", color="#e6e9ee")
    ax.tick_params(colors="#8b93a1")
    for sp in ax.spines.values():
        sp.set_color("#2a2f3a")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="#0e1116")
    print(f"[build] wrote {out} and {npy}")


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
    ap.add_argument("--build", action="store_true",
                    help="stitch every lap on a common centre -> mat_grid.png")
    ap.add_argument("--draw", action="store_true",
                    help="draw the recorded laps to mat_map.png")
    ap.add_argument("--fit", action="store_true",
                    help="STEP 2: fit the corridor walls from the recorded laps")
    args = ap.parse_args()

    if args.status:
        status(args.out)
        return
    if args.build:
        build_map(args.out)
        return
    if args.draw:
        draw_map(args.out)
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
