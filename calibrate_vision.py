#!/usr/bin/env python3
"""
calibrate_vision.py — find the pillar-PD numbers for ObstacleRound.cpp:

    VIS_AREA_START       area where vision STARTS steering  (weight 0)
    VIS_AREA_FULL        area where vision has FULL control  (weight 1)
    VIS_RED_TARGET_PX    pixel offset to hold a red pillar at   (pass it on the right)
    VIS_GREEN_TARGET_PX  pixel offset to hold a green pillar at (pass it on the left)

It runs obstacleRound.py's OWN VisionThread (same resize, same HSV from
config.json, same largest-blob pick), so the area and err you read here are
exactly the numbers the STM32 will receive. The LiDAR is optional; if it is
running, the pillar's distance is shown next to its area so you can pick the
thresholds by distance instead of guessing.

Put it in the blueprintpi repo root (next to obstacleRound.py) and run it
INSTEAD of obstacleRound.py / dashboard.py (one process owns the camera):

    python3 calibrate_vision.py            # camera + lidar
    python3 calibrate_vision.py --no-lidar # camera only

Then open  http://<pi>:5000  on your laptop.

PROCEDURE (car still, on the mat, wheels straight along the lane)

  1. MIN AREA — put a pillar straight ahead at the distance where the car
     should START reacting to it (typically ~80-100 cm). Press "Capture MIN".
  2. MAX AREA — move the pillar closer, to where the car must be FULLY
     committed to passing it (typically ~35-45 cm). Press "Capture MAX".
  3. RED OFFSET — red pillar at roughly the MAX distance. Slide the car
     sideways (keep it parallel to the walls) until it sits where you want
     it to drive past: pillar on the car's LEFT with your chosen clearance.
     Press "Capture RED". The result should be negative (pillar left of centre).
  4. GREEN OFFSET — same with a green pillar on the car's RIGHT.
     Press "Capture GREEN". The result should be positive.

Each capture takes the median of CAPTURE_FRAMES fresh camera frames. Results
are saved to vision_cal.json and printed as paste-ready C++ lines.

Note on offsets: the PD holds a FIXED pixel offset, which on this lens is
close to a fixed bearing, so the sideways clearance shrinks as the pillar
gets closer. Capture the offsets at the distance where the pillar is
actually being passed (around the MAX distance), not far away.
"""

import argparse
import json
import math
import os
import statistics
import threading
import time

import cv2
from flask import Flask, Response, jsonify

import obstacleRound as ob
import sensors.camera as camera

PORT = 5000
CAPTURE_FRAMES = 30
CAPTURE_TIMEOUT_S = 6.0
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vision_cal.json")

KINDS = {                     # kind -> (colour required, field measured)
    "min":   (None,         "area"),
    "max":   (None,         "area"),
    "red":   (ob.COL_RED,   "err"),
    "green": (ob.COL_GREEN, "err"),
}
NAMES = {ob.COL_RED: "RED", ob.COL_GREEN: "GREEN", ob.COL_NONE: "none"}

cfg = camera.load_config()
FUSION = cfg.get("fusion", {})
lidar = None

results = {}
capture_state = {"busy": False, "kind": None, "n": 0, "msg": ""}
res_lock = threading.Lock()


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def pillar_distance_mm(v):
    """LiDAR distance at the pillar's bearing (same fusion as main.fuse)."""
    if lidar is None or v["box"] is None or v["color"] == ob.COL_NONE:
        return None
    if time.monotonic() - lidar.last_point_t > 0.3:
        return None
    x0, y0, x1, y1 = v["box"]
    bearing = camera.px_to_bearing((x0 + x1) / 2.0, camera.FRAME_W,
                                   cfg.get("hfov_deg", 160),
                                   cy=(y0 + y1) / 2.0,
                                   offset_deg=cfg.get("camera_offset_deg", 0.0))
    from sensors.lidar import select_range
    d = select_range(lidar._ranges, int(round(bearing)) % 360,
                     FUSION.get("bearing_match_deg", 8),
                     floor_mm=FUSION.get("range_floor_mm", 0.0),
                     gap_split_mm=FUSION.get("gap_split_mm", 0.0))
    return None if math.isinf(d) else int(d)


def live():
    v = ob.vision_now(time.monotonic())
    v["dist_mm"] = pillar_distance_mm(v)
    return v


def summarise(vals):
    return {"median": int(round(statistics.median(vals))),
            "lo": int(min(vals)), "hi": int(max(vals)), "n": len(vals)}


def do_capture(kind):
    want, field = KINDS[kind]
    vals, dists, colours = [], [], set()
    last_seq = None
    t0 = time.monotonic()
    while len(vals) < CAPTURE_FRAMES and time.monotonic() - t0 < CAPTURE_TIMEOUT_S:
        v = live()
        if v["seq"] != last_seq:
            last_seq = v["seq"]
            ok = v["color"] != ob.COL_NONE and (want is None or v["color"] == want)
            if ok:
                vals.append(v[field])
                colours.add(v["color"])
                if v["dist_mm"] is not None:
                    dists.append(v["dist_mm"])
                with res_lock:
                    capture_state["n"] = len(vals)
        time.sleep(0.005)

    with res_lock:
        capture_state["busy"] = False
        if len(vals) < CAPTURE_FRAMES // 2:
            need = NAMES[want] if want is not None else "a red or green"
            capture_state["msg"] = (f"{kind}: only {len(vals)} frames saw {need} "
                                    f"pillar — check the stream / HSV and retry")
            return
        r = summarise(vals)
        r["field"] = field
        r["colour"] = "/".join(NAMES[c] for c in sorted(colours))
        r["dist_mm"] = int(statistics.median(dists)) if dists else None
        results[kind] = r
        capture_state["msg"] = f"{kind}: {field} = {r['median']} (spread {r['lo']}..{r['hi']})"
        save()
    print_constants()


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def warnings():
    w = []
    mn, mx = results.get("min"), results.get("max")
    if mn and mx and mn["median"] >= mx["median"]:
        w.append("MIN area must be smaller than MAX area — pillar for MIN goes further away.")
    if mn and mn["median"] <= 300:
        w.append("MIN area is at/below VIS_AREA_MIN (300) — lower VIS_AREA_MIN too.")
    if "red" in results and results["red"]["median"] >= 0:
        w.append("RED offset should be NEGATIVE (pillar on the car's left).")
    if "green" in results and results["green"]["median"] <= 0:
        w.append("GREEN offset should be POSITIVE (pillar on the car's right).")
    return w


def constants_text():
    g = lambda k: str(results[k]["median"]) if k in results else "/* not captured */"
    lines = [
        f"const long  VIS_AREA_START      = {g('min')};",
        f"const long  VIS_AREA_FULL       = {g('max')};",
        f"const int   VIS_RED_TARGET_PX   = {g('red')};",
        f"const int   VIS_GREEN_TARGET_PX = {g('green')};",
    ]
    return "\n".join(lines + [f"// WARNING: {x}" for x in warnings()])


def print_constants():
    print("\n---- paste into ObstacleRound.cpp ----")
    print(constants_text())
    print("--------------------------------------\n")


def save():
    try:
        with open(OUT_PATH, "w") as f:
            json.dump({"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "results": results}, f, indent=2)
    except OSError as e:
        print(f"[cal] could not save {OUT_PATH}: {e}")


def load():
    try:
        with open(OUT_PATH) as f:
            results.update(json.load(f).get("results", {}))
    except (OSError, ValueError):
        pass


# --------------------------------------------------------------------------
# web UI
# --------------------------------------------------------------------------

app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width">
<title>Vision calibration</title><style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}
img{width:100%;max-width:640px;border:1px solid #444}
.row{display:flex;gap:16px;flex-wrap:wrap}
button{font-size:16px;padding:10px 14px;margin:4px;border:0;border-radius:6px;cursor:pointer}
.min{background:#456}.max{background:#465}.red{background:#a33}.green{background:#3a4}
button:disabled{opacity:.4}
pre{background:#222;padding:10px;border-radius:6px;white-space:pre-wrap}
.big{font-size:22px;font-variant-numeric:tabular-nums}
</style></head><body>
<h2>Pillar vision calibration</h2>
<div class=row><div><img src="/stream"></div>
<div style="min-width:280px">
<div class=big id=live>…</div>
<p>
<button class=min   onclick="cap('min')">Capture MIN area</button><br>
<button class=max   onclick="cap('max')">Capture MAX area</button><br>
<button class=red   onclick="cap('red')">Capture RED offset</button><br>
<button class=green onclick="cap('green')">Capture GREEN offset</button>
</p>
<div id=msg></div>
<h3>Results</h3><pre id=res></pre>
<h3>Paste into ObstacleRound.cpp</h3><pre id=cpp></pre>
</div></div>
<script>
async function cap(k){await fetch('/api/capture/'+k,{method:'POST'});}
function f(x){return x===null||x===undefined?'--':x}
async function tick(){
 try{
  const r=await (await fetch('/api/live')).json();
  const v=r.live;
  document.getElementById('live').innerHTML=
   `colour <b>${v.name}</b><br>err <b>${v.err}</b> px<br>area <b>${v.area}</b><br>dist <b>${f(v.dist_mm)}</b> mm`;
  const c=r.capture;
  document.querySelectorAll('button').forEach(b=>b.disabled=c.busy);
  document.getElementById('msg').textContent=c.busy?`capturing ${c.kind}… ${c.n}/${r.frames}`:c.msg;
  let t='';
  for(const [k,x] of Object.entries(r.results))
   t+=`${k.padEnd(6)} ${x.field}=${x.median}  (${x.lo}..${x.hi}, n=${x.n})  ${x.colour}  dist=${f(x.dist_mm)}mm\\n`;
  document.getElementById('res').textContent=t||'nothing yet';
  document.getElementById('cpp').textContent=r.cpp;
 }catch(e){}
 setTimeout(tick,200);
}
tick();
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/live")
def api_live():
    v = live()
    with res_lock:
        return jsonify({
            "live": {"color": v["color"], "name": NAMES[v["color"]], "err": v["err"],
                     "area": v["area"], "dist_mm": v["dist_mm"]},
            "capture": dict(capture_state),
            "results": results,
            "frames": CAPTURE_FRAMES,
            "cpp": constants_text(),
        })


@app.route("/api/capture/<kind>", methods=["POST"])
def api_capture(kind):
    if kind not in KINDS:
        return jsonify({"ok": False, "error": "unknown kind"}), 404
    with res_lock:
        if capture_state["busy"]:
            return jsonify({"ok": False, "error": "busy"}), 409
        capture_state.update(busy=True, kind=kind, n=0, msg="")
    threading.Thread(target=do_capture, args=(kind,), daemon=True).start()
    return jsonify({"ok": True})


def mjpeg():
    while True:
        with ob.lock:
            frame = None if ob.latest_frame is None else ob.latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        v = live()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cx = camera.FRAME_W // 2
        cv2.line(bgr, (cx, 0), (cx, camera.FRAME_H), (255, 255, 255), 1)
        # current targets as guide lines
        for k, col in (("red", (0, 0, 255)), ("green", (0, 255, 0))):
            if k in results:
                x = cx + results[k]["median"]
                cv2.line(bgr, (x, 0), (x, camera.FRAME_H), col, 1)
        if v["box"] is not None and v["color"] != ob.COL_NONE:
            col = ob.BOX_BGR[v["color"]]
            x0, y0, x1, y1 = v["box"]
            cv2.rectangle(bgr, (x0, y0), (x1, y1), col, 2)
            px = (x0 + x1) // 2
            cv2.line(bgr, (px, y0), (px, y1), col, 1)
            d = "" if v["dist_mm"] is None else f" d:{v['dist_mm']}mm"
            cv2.putText(bgr, f"err:{v['err']} area:{v['area']}{d}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + jpg.tobytes() + b"\r\n")
        time.sleep(0.05)


@app.route("/stream")
def stream():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------

def main():
    global lidar
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lidar", action="store_true", help="camera only, no distance readout")
    args = ap.parse_args()

    load()
    ob.viewers = 1                      # make VisionThread keep frames for our stream
    vis = ob.VisionThread()
    vis.start()

    if not args.no_lidar:
        try:
            from worldstate import SharedState
            from sensors.lidar import LidarThread
            lidar = LidarThread(SharedState())
            lidar.start()
        except Exception as e:
            print(f"[cal] lidar unavailable ({e}) — distances will show '--'")
            lidar = None

    print(f"[cal] open http://<pi>:{PORT}   results -> {OUT_PATH}")
    if results:
        print_constants()
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False, use_reloader=False)
    finally:
        vis.stop()
        vis.join(timeout=2.0)
        if lidar is not None:
            lidar.stop()
            lidar.join(timeout=2.0)
        print_constants()


if __name__ == "__main__":
    main()
