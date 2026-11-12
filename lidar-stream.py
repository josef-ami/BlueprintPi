#!/usr/bin/env python3
"""Live view of the RPLIDAR C1 data streamed by lidar_stream.ino.

Install once:   pip install pyserial numpy matplotlib
Run:            python lidar_view.py                (finds the ESP32 port itself)
                python lidar_view.py --port COM5
                python lidar_view.py --port COM5 --log scan.csv     (also saves every point)
                python lidar_view.py --demo                         (fake room, no hardware)

Close the Arduino Serial Monitor first: only one program can hold the port.
Picture: top view, robot at the centre, forward = up, angles clockwise (RPLIDAR convention).
If the lidar is mounted upside-down the picture is mirrored: use --mirror. Use --rotate DEG to turn it.
"""
import argparse
import csv
import math
import struct
import threading
import time

import numpy as np

HDR = b"\xAA\x55"


def find_port():
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:                                   # CP210x (10C4) and CH340 (1A86) are the usual ESP32 bridges
        if p.vid in (0x10C4, 0x1A86):
            return p.device
    return ports[0].device if ports else None


class SerialReader(threading.Thread):
    def __init__(self, port, baud, log_path=None):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.lock = threading.Lock()
        self.frame = None            # (angles_deg, dist_mm)
        self.rev = 0
        self.bad = 0
        self.hz = 0.0
        self.running = True
        self.log = open(log_path, "w", newline="") if log_path else None
        self.writer = csv.writer(self.log) if self.log else None
        if self.writer:
            self.writer.writerow(["rev", "angle_deg", "distance_mm"])

    def run(self):
        buf = bytearray()
        last = None
        while self.running:
            buf += self.ser.read(self.ser.in_waiting or 1)
            while True:
                i = buf.find(HDR)
                if i < 0:
                    del buf[:-1]                       # keep a possible half header
                    break
                if i > 0:
                    del buf[:i]                        # junk before the header (boot text etc.)
                if len(buf) < 4:
                    break
                n = struct.unpack_from("<H", buf, 2)[0]
                if n > 2000:
                    del buf[:2]
                    continue
                total = 4 + n * 4 + 1
                if len(buf) < total:
                    break
                x = 0
                for b in buf[2:total - 1]:
                    x ^= b
                if x != buf[total - 1]:
                    self.bad += 1
                    del buf[:2]
                    continue
                arr = np.frombuffer(bytes(buf[4:total - 1]), dtype="<u2").reshape(-1, 2)
                ang = arr[:, 0] / 10.0
                dist = arr[:, 1].astype(float)
                now = time.time()
                with self.lock:
                    self.frame = (ang, dist)
                    self.rev += 1
                    if last:
                        self.hz = 0.8 * self.hz + 0.2 / max(now - last, 1e-3) if self.hz else 1.0 / max(now - last, 1e-3)
                    if self.writer:
                        for a, d in zip(ang, dist):
                            self.writer.writerow([self.rev, f"{a:.1f}", int(d)])
                last = now
                del buf[:total]

    def stop(self):
        self.running = False
        try:
            self.ser.write(b"x")                       # stop the lidar motor
            self.ser.close()
        except Exception:
            pass
        if self.log:
            self.log.close()


class DemoReader(threading.Thread):
    """Fake 10 Hz scans of a rectangular room, for trying the viewer without hardware."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.frame = None
        self.rev = 0
        self.bad = 0
        self.hz = 10.0
        self.running = True

    def run(self):
        rng = np.random.default_rng(1)
        t = 0.0
        while self.running:
            ang = np.arange(0, 360, 0.72) + rng.normal(0, 0.05, 500)
            th = np.radians(ang)
            dx, dy = np.sin(th), np.cos(th)
            with np.errstate(divide="ignore"):
                walls = []
                for pos, d in ((2300, dx), (-1700, dx), (1800, dy), (-1200, dy)):
                    tt = pos / d
                    walls.append(np.where(tt > 0, tt, np.inf))
            dist = np.min(walls, axis=0)
            # a box obstacle ahead-right, a thin pole behind
            for cx, cy, r in ((600 + 100 * math.sin(t), 900, 160), (-500, -700, 40)):
                b = dx * cx + dy * cy
                disc = b * b - (cx * cx + cy * cy - r * r)
                hit = disc > 0
                tt = np.where(hit, b - np.sqrt(np.where(hit, disc, 0)), np.inf)
                dist = np.where((tt > 0) & (tt < dist), tt, dist)
            dist = dist + rng.normal(0, 4, dist.size)
            dist[rng.random(dist.size) < 0.1] = 0       # some points with no return
            with self.lock:
                self.frame = (ang % 360, dist)
                self.rev += 1
            t += 0.1
            time.sleep(0.1)

    def stop(self):
        self.running = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--range", type=float, default=4.0, help="plot radius in metres")
    ap.add_argument("--mirror", action="store_true", help="mirror left/right (upside-down lidar)")
    ap.add_argument("--rotate", type=float, default=0.0, help="rotate the picture by this many degrees")
    ap.add_argument("--log", help="save every point to this CSV file")
    ap.add_argument("--demo", action="store_true", help="fake data, no hardware")
    ap.add_argument("--snapshot", help="save a PNG after 3 s and exit (for testing)")
    args = ap.parse_args()

    if args.snapshot:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Polygon

    if args.demo:
        reader = DemoReader()
    else:
        port = args.port or find_port()
        if not port:
            raise SystemExit("No serial port found. Plug the ESP32 in or pass --port COM5")
        print(f"reading {port} at {args.baud} baud (close the Arduino Serial Monitor first)")
        reader = SerialReader(port, args.baud, args.log)
    reader.start()

    R = args.range
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_xlabel("m (right)")
    ax.set_ylabel("m (forward)")
    ax.grid(alpha=0.25)
    for r in range(1, int(R) + 1):
        ax.add_patch(Circle((0, 0), r, fill=False, ls="--", lw=0.6, alpha=0.35))
        ax.text(0, r, f"{r} m", fontsize=8, alpha=0.5, ha="center", va="bottom")
    ax.add_patch(Polygon([[0, 0.12], [-0.08, -0.08], [0.08, -0.08]], closed=True, color="tab:red"))   # robot, nose up
    sc = ax.scatter([], [], s=6, c=[], cmap="viridis", vmin=0, vmax=R)
    fig.colorbar(sc, ax=ax, label="distance (m)", shrink=0.8)
    info = ax.text(0.01, 0.99, "waiting for frames...", transform=ax.transAxes, va="top", fontsize=10,
                   family="monospace", bbox=dict(facecolor="white", alpha=0.75, lw=0))
    state = {"seen": 0}

    def update(_):
        with reader.lock:
            frame, rev, hz, bad = reader.frame, reader.rev, reader.hz, reader.bad
        if frame is None or rev == state["seen"]:
            return (sc, info)
        state["seen"] = rev
        ang, dist = frame
        valid = dist > 0
        th = np.radians(ang[valid] + args.rotate)
        if args.mirror:
            th = -th
        d = dist[valid] / 1000.0
        x, y = d * np.sin(th), d * np.cos(th)           # forward = up, clockwise positive
        keep = d <= R
        sc.set_offsets(np.c_[x[keep], y[keep]])
        sc.set_array(d[keep])
        near = d.min() if d.size else float("nan")
        info.set_text(f"rev {rev}   {hz:4.1f} rev/s\npoints {ang.size}   valid {int(valid.sum())}\n"
                      f"nearest {near:5.2f} m   bad frames {bad}")
        return (sc, info)

    ani = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
    fig.canvas.manager.set_window_title("RPLIDAR C1")
    try:
        if args.snapshot:
            t0 = time.time()
            while time.time() - t0 < 3:
                update(None)
                time.sleep(0.1)
            fig.savefig(args.snapshot, dpi=110)
        else:
            plt.show()
    finally:
        reader.stop()


if __name__ == "__main__":
    main()
