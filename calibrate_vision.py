#!/usr/bin/env python3
"""
calibrate_vision.py - click the mat and the pillars, get Lab thresholds.

Run it INSTEAD of obstacleRound.py (one process owns the camera):

    python3 calibrate_vision.py          # camera + lidar (lidar only for AREA_K)
    python3 calibrate_vision.py --no-lidar

then open  http://<pi>:5000

WHY LAB AND NOT HSV
  HSV separates the pillars on hue but gates on SATURATION. A matte pillar under
  dim indoor light falls under the S threshold and disappears - and green
  disappears first, because red has two hue bands and survives longer. That is
  the usual cause of "green stops being detected".

  In OpenCV's 8-bit Lab, a and b are centred on 128: a > 128 is red, a < 128 is
  green, and the white mat sits near (128, 128) whatever the light does to L.
  The pillars separate on ONE channel and brightness never removes the colour.

HOW TO USE IT
  1. Put the car on the mat with a red and a green pillar in view.
  2. Pick a class (Mat / Red / Green) and CLICK that thing in the image.
     Click 8-15 times per class, spread across the frame - near and far, in the
     bright patch and in the shadow, on the lit face and the shaded face. Every
     click samples a small patch, so a few clicks in the right places beat
     dozens in one spot.
  3. Press FIT. Ranges are fitted from your samples and pushed clear of the mat
     samples, then applied live - the detector boxes update immediately.
  4. Check the Detector panel: it runs the REAL find_pillars() on the live
     frame and shows what it accepts, and for anything it rejects, which test
     said no (A flat, S ragged, F not on mat, C low contrast).
  5. That's it: FIT already wrote the result to vision_cal.json (atomic, synced
     to the SD card), so it survives the Pi being switched off, and
     obstacleRound.py loads it on every start. SAVE writes it again by hand.
     Calibrate ONCE per venue / lighting; nothing else overwrites this file -
     not tuning.json, not "Revert Pi" on the race page.

AREA_K
  The firmware falls back to a distance estimate from blob area when no LiDAR
  return agrees with the camera ray: distance ~ AREA_K / sqrt(area). Put a
  pillar at a measured distance, type that distance in, click the pillar, and
  the page computes AREA_K for you.

What this replaces: the old version captured VIS_AREA_START / VIS_AREA_FULL /
VIS_RED_TARGET_PX / VIS_GREEN_TARGET_PX for a pixel-offset PD law. That law is
gone - the firmware steers to a lane POSITION now, and those four constants do
not exist in it any more. AREA_K is the one number from the old flow that the
firmware still uses.
"""

import argparse
import math
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

import obstacleRound as ob
import params as prm

PORT = 5000
PATCH = 5                 # click samples a PATCH x PATCH box, in detection pixels
MAX_SAMPLES = 400

CLASSES = ("mat", "red", "green")

# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

lock = threading.Lock()
samples = {c: [] for c in CLASSES}      # each: {"x","y","L","a","b","H","S","V"}
msg = {"text": "click the mat and the pillars, then press FIT"}
view = {"mode": "camera"}
area_k = {"dist_mm": 600.0, "result": None}
lidar = None


def fit_params():
    """Live copy of the fitting knobs (exposed on the page)."""
    return dict(margin=FIT["margin"], sep=FIT["sep"], pct=FIT["pct"])


FIT = {"margin": 10, "sep": 6, "pct": 5}


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def sample_at(cls, x, y):
    """Sample the detection-resolution frame at (x, y). Returns the sample."""
    with ob.lock:
        frame = None if ob.latest_frame is None else ob.latest_frame.copy()
    if frame is None:
        return None
    small = cv2.resize(frame, ob.PROC_SIZE, interpolation=cv2.INTER_AREA)
    hsv, lab = ob.colour_spaces(small)
    h, w = small.shape[:2]
    x = max(0, min(w - 1, int(x)))
    y = max(0, min(h - 1, int(y)))
    r = PATCH // 2
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    # median, not mean: one stray pixel on a pillar edge should not move it
    L, a, b = np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    H, S, V = np.median(hsv[y0:y1, x0:x1].reshape(-1, 3), axis=0)
    s = {"x": x, "y": y, "L": float(L), "a": float(a), "b": float(b),
         "H": float(H), "S": float(S), "V": float(V)}
    with lock:
        if len(samples[cls]) < MAX_SAMPLES:
            samples[cls].append(s)
    return s


def stats(cls):
    with lock:
        rows = list(samples[cls])
    if not rows:
        return None
    arr = np.array([[r["L"], r["a"], r["b"]] for r in rows])
    return {"n": len(rows),
            "mean": arr.mean(axis=0).round(1).tolist(),
            "min": arr.min(axis=0).round(0).tolist(),
            "max": arr.max(axis=0).round(0).tolist()}


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
#
# ONE channel classifies, the others only exclude.
#
# 'a' is what separates the three classes: red sits well above 128, green well
# below, the mat on it. So each pillar's 'a' boundary is placed BETWEEN its own
# samples and the mat's - halfway from one distribution's edge to the other,
# using robust percentiles rather than min/max so one bad click cannot move it.
# If the two distributions actually touch, the fit REFUSES rather than
# returning thresholds that cannot work; that means the camera genuinely cannot
# tell them apart and the answer is exposure or white balance, not numbers.
#
# L and b are fitted loosely on purpose. A pillar is seen near and far, lit and
# shaded, so its lightness moves far more than its colour, and a tight L range
# fitted under one lighting condition is exactly what makes a calibration stop
# working when the room changes. They only drop near-black and wildly wrong hues.

def fit():
    with lock:
        have = {c: len(samples[c]) for c in CLASSES}
    missing = [c for c in CLASSES if have[c] < 3]
    if missing:
        return False, f"need at least 3 samples of: {', '.join(missing)}"

    def arr(c):
        with lock:
            return np.array([[r["L"], r["a"], r["b"]] for r in samples[c]])

    m, sep, pct = FIT["margin"], FIT["sep"], FIT["pct"]
    mat, red, grn = arr("mat"), arr("red"), arr("green")

    lo_p = lambda a, i: float(np.percentile(a[:, i], pct))
    hi_p = lambda a, i: float(np.percentile(a[:, i], 100 - pct))

    out, warn = {}, []
    for name, a in (("RED", red), ("GREEN", grn)):
        # 'a' is the channel that does the work, and its boundary is placed
        # BETWEEN the two distributions - halfway from this class's edge to the
        # mat's. Clamping to "mat edge minus sep" instead, as an earlier version
        # did, can land INSIDE this class's own samples and threshold the pillar
        # straight back out.
        if name == "RED":
            mine, theirs = lo_p(a, 1), hi_p(mat, 1)          # my low vs mat's high
            if mine <= theirs + sep:
                return False, ("red and the mat have overlapping 'a' values - the "
                               "camera cannot tell them apart. Re-sample, or fix "
                               "exposure / white balance")
            a_lo, a_hi = (mine + theirs) / 2.0, 255.0
        else:
            mine, theirs = hi_p(a, 1), lo_p(mat, 1)          # my high vs mat's low
            if mine >= theirs - sep:
                return False, ("green and the mat have overlapping 'a' values - the "
                               "camera cannot tell them apart. Re-sample, or fix "
                               "exposure / white balance")
            a_lo, a_hi = 0.0, (mine + theirs) / 2.0

        # L and b do NOT classify - a does. A pillar is seen near and far, lit
        # and shaded, so its lightness moves far more than its colour, and
        # fitting L tightly to one lighting condition is what makes a
        # calibration stop working when a cloud passes. Both are kept loose:
        # L only drops near-black, b only drops the wildly wrong hue.
        L_lo = max(10.0, lo_p(a, 0) - max(40.0, 4 * m))
        b_lo = max(0.0, lo_p(a, 2) - 3 * m)
        b_hi = min(255.0, hi_p(a, 2) + 3 * m)
        out[name] = dict(L=(L_lo, 255.0), a=(a_lo, a_hi), b=(b_lo, b_hi))

    # Mat: bright and near-neutral. Same reasoning - generous headroom on
    # brightness, because the mat dims with the room and the floor test failing
    # rejects every pillar at once (reason F).
    # Extra-generous, and deliberately more so than the pillars': the mat is not
    # evenly lit (it is brighter under the lights and darker at the far end), and
    # when this threshold is missed the floor test rejects EVERY pillar at once
    # with reason F - one number taking the whole detector down. Being loose here
    # costs little, because the near-neutral a/b test is what actually
    # identifies the mat.
    floor_L = max(0.0, lo_p(mat, 0) - max(60.0, 6 * m))
    ab_tol = float(np.ceil(max(10.0, max(np.abs(mat[:, 1] - 128).max(),
                                         np.abs(mat[:, 2] - 128).max()) + m)))

    # chroma gap between the pillars and the mat under them
    def chroma(a):
        return np.hypot(a[:, 1] - 128, a[:, 2] - 128)
    gap = min(chroma(red).mean(), chroma(grn).mean()) - chroma(mat).mean()
    chroma_min = float(max(5.0, round(gap * 0.5)))

    vals = {
        "RED_L_LO": out["RED"]["L"][0],   "RED_L_HI": out["RED"]["L"][1],
        "RED_A_LO": out["RED"]["a"][0],   "RED_A_HI": out["RED"]["a"][1],
        "RED_B_LO": out["RED"]["b"][0],   "RED_B_HI": out["RED"]["b"][1],
        "GREEN_L_LO": out["GREEN"]["L"][0], "GREEN_L_HI": out["GREEN"]["L"][1],
        "GREEN_A_LO": out["GREEN"]["a"][0], "GREEN_A_HI": out["GREEN"]["a"][1],
        "GREEN_B_LO": out["GREEN"]["b"][0], "GREEN_B_HI": out["GREEN"]["b"][1],
        "FLOOR_L_MIN": floor_L,
        "FLOOR_AB_TOL": ab_tol,
        "LAB_CHROMA_MIN": chroma_min,
        "USE_LAB": True,
    }
    ob.PI.set_many(vals)
    ob.sync_globals()
    try:
        prm.save_calibration(ob.PI)
        saved = "fitted, applied live and saved to vision_cal.json"
    except OSError as e:
        saved = f"fitted and applied live, but NOT saved: {e}"
    text = saved + (" - " + "; ".join(sorted(set(warn))) if warn else "")
    if gap < 12:
        text += ". WARNING: pillars are barely more colourful than the mat - " \
                "check exposure and white balance"
    return True, text


# --------------------------------------------------------------------------
# what the real detector makes of the current frame
# --------------------------------------------------------------------------

REASON = {"H": "too short (box height)", "A": "too flat (aspect)", "S": "ragged (solidity)",
          "F": "not standing on the mat", "C": "too little colour vs the mat"}


def detector_report():
    with ob.lock:
        frame = None if ob.latest_frame is None else ob.latest_frame.copy()
    if frame is None:
        return {"ok": False, "blobs": []}
    small = cv2.resize(frame, ob.PROC_SIZE, interpolation=cv2.INTER_AREA)
    hsv, lab = ob.colour_spaces(small)
    accepted, cands, _ = ob.find_pillars(hsv, lab, ob.PILLAR_FILTER)
    blobs = []
    for cnt, a, code, why in sorted(cands, key=lambda c: -c[1])[:8]:
        x, y, w, h = cv2.boundingRect(cnt)
        blobs.append({"colour": ob.NAMES[code], "area": int(a),
                      "box": [x, y, w, h],
                      "verdict": "ACCEPTED" if why is None else REASON.get(why, why)})
    return {"ok": True, "space": "Lab" if ob.USE_LAB else "HSV",
            "min_area": ob.MIN_AREA_PROC,
            # what the firmware would actually receive this frame
            "primary": None if not accepted else
                       {"colour": ob.NAMES[accepted[0][2]], "area": int(accepted[0][1])},
            "blobs": blobs}


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------

def render(mode):
    with ob.lock:
        frame = None if ob.latest_frame is None else ob.latest_frame.copy()
    if frame is None:
        return None
    small = cv2.resize(frame, ob.PROC_SIZE, interpolation=cv2.INTER_AREA)
    hsv, lab = ob.colour_spaces(small)

    if mode in ("red", "green"):
        m = ob.colour_mask(hsv, lab, mode.upper())
        out = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    elif mode == "mat":
        out = cv2.cvtColor(ob.floor_mask(hsv, lab, ob.PILLAR_FILTER), cv2.COLOR_GRAY2BGR)
    elif mode == "a":
        # the channel that separates the classes, as a blue-to-red ramp
        out = cv2.applyColorMap(lab[:, :, 1], cv2.COLORMAP_COOL)
    elif mode == "chroma":
        ch = np.clip(ob.chroma(lab) * 2, 0, 255).astype(np.uint8)
        out = cv2.applyColorMap(ch, cv2.COLORMAP_VIRIDIS)
    else:
        out = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        accepted, cands, _ = ob.find_pillars(hsv, lab, ob.PILLAR_FILTER)
        rank = {id(t[0]): i for i, t in enumerate(accepted[:2])}
        for cnt, a, code, why in cands:
            x, y, w, h = cv2.boundingRect(cnt)
            col = (140, 140, 140) if why else ob.BOX_BGR[code]
            cv2.rectangle(out, (x, y), (x + w, y + h), col, 1 if why else 2)
            tag = why if why else ("1st" if rank.get(id(cnt)) == 0 else
                                   "2nd" if rank.get(id(cnt)) == 1 else "")
            if tag:
                cv2.putText(out, tag, (x, max(9, y - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

    # sample markers, always
    for cls, col in (("mat", (255, 255, 255)), ("red", (60, 60, 255)),
                     ("green", (60, 220, 60))):
        with lock:
            pts = [(s["x"], s["y"]) for s in samples[cls]]
        for (x, y) in pts:
            cv2.drawMarker(out, (x, y), col, cv2.MARKER_CROSS, 7, 1)
    return cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# web
# --------------------------------------------------------------------------

app = Flask(__name__)

PAGE = r"""<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Lab calibration</title><style>
:root{--bg:#111;--panel:#1b1b1b;--line:#333;--fg:#eee;--dim:#999;--ok:#6d6;--bad:#f66;--acc:#2a7}
*{box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:12px}
h2{margin:0 0 10px}h3{margin:14px 0 6px;font-size:14px;color:var(--acc);
  text-transform:uppercase;letter-spacing:.08em}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
img{border:1px solid var(--line);background:#000;cursor:crosshair;max-width:100%;
  width:640px;image-rendering:pixelated}
button{font-size:14px;padding:8px 14px;margin:3px 4px 3px 0;border:0;border-radius:6px;
  cursor:pointer;color:#fff;background:#444}
button.on{outline:2px solid #fff}
#cmat{background:#777}#cred{background:#a33}#cgreen{background:#2a6}
#fit{background:var(--acc);font-size:16px;padding:10px 20px}
#save{background:#36c;font-size:16px;padding:10px 20px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:10px 12px;min-width:320px;flex:1 1 340px}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}
td,th{padding:2px 8px 2px 0;text-align:left}th{color:var(--dim);font-weight:400}
.dim{color:var(--dim)}.ok{color:var(--ok)}.bad{color:var(--bad)}
.hint{font-size:12px;color:var(--dim);margin:4px 0 8px;line-height:1.5}
input[type=number]{width:70px;background:#0d0d0d;color:var(--fg);border:1px solid var(--line);
  border-radius:4px;padding:3px 5px;font-size:13px}
#msg{font-size:14px;margin:8px 0;min-height:20px}
</style></head><body>

<h2>Lab calibration</h2>
<div class=row>

<div>
  <img id=cam src="/view?mode=camera">
  <div class=hint>Click the image to sample. Crosses mark your samples:
    white = mat, red = red pillar, green = green pillar.</div>
  <div>
    <b class=dim>sampling:</b>
    <button id=cmat class=on onclick="pick('mat')">Mat</button>
    <button id=cred onclick="pick('red')">Red pillar</button>
    <button id=cgreen onclick="pick('green')">Green pillar</button>
    <button onclick="undo()">Undo</button>
    <button onclick="clr()">Clear class</button>
  </div>
  <div>
    <b class=dim>view:</b>
    <button onclick="setview('camera')">camera</button>
    <button onclick="setview('red')">red mask</button>
    <button onclick="setview('green')">green mask</button>
    <button onclick="setview('mat')">mat mask</button>
    <button onclick="setview('a')">a channel</button>
    <button onclick="setview('chroma')">chroma</button>
  </div>
  <div style="margin-top:10px">
    <button id=fit onclick="dofit()">FIT</button>
    <button id=save onclick="save()">SAVE to vision_cal.json</button>
  </div>
  <div id=msg class=dim></div>
</div>

<div class=panel>
  <h3>Samples</h3>
  <table id=stats></table>
  <div class=hint>8-15 clicks per class, spread across the frame: near and far,
    lit and shaded, both faces of each pillar.</div>

  <h3>Fit</h3>
  <span class=dim>margin</span> <input type=number id=fmargin value=10 min=0 max=40
    onchange="setfit()">
  <span class=dim>mat gap</span> <input type=number id=fsep value=6 min=0 max=40
    onchange="setfit()">
  <span class=dim>percentile</span> <input type=number id=fpct value=5 min=0 max=25
    onchange="setfit()">
  <div class=hint>margin widens each range; mat gap forces the pillar ranges clear
    of the mat on the <b>a</b> channel; percentile trims outliers.</div>

  <h3>Detector <span id=space class=dim></span></h3>
  <table id=det></table>

  <h3>Fitted values</h3>
  <table id=vals></table>

  <h3>AREA_K</h3>
  <div class=hint>Pillar at a measured distance, type it, then click the pillar
    with the <b>Red</b> or <b>Green</b> class selected.</div>
  <span class=dim>true distance</span>
  <input type=number id=dist value=600 min=100 max=3000 onchange="setdist()"> mm
  <div id=areak class=dim></div>
</div>
</div>

<script>
const $=id=>document.getElementById(id);
let cls='mat';
function pick(c){cls=c;for(const k of ['mat','red','green'])
  $('c'+k).classList.toggle('on',k===c);}
function setview(m){view=m;$('cam').src='/view?mode='+m+'&t='+Date.now();}
let view='camera';
$('cam').onclick=async e=>{
  const r=e.target.getBoundingClientRect();
  const x=(e.clientX-r.left)/r.width*320, y=(e.clientY-r.top)/r.height*240;
  const s=await (await fetch('/api/sample',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({cls,x,y})})).json();
  if(s.ok){$('msg').innerHTML=`<span class=dim>${cls} @ ${s.s.x},${s.s.y}</span>  `+
    `L ${s.s.L.toFixed(0)}  a ${s.s.a.toFixed(0)}  b ${s.s.b.toFixed(0)}`+
    `  <span class=dim>(HSV ${s.s.H.toFixed(0)},${s.s.S.toFixed(0)},${s.s.V.toFixed(0)})</span>`;
   if(s.area_k)$('areak').innerHTML=`area ${s.area_k.area} at ${s.area_k.dist} mm `+
     `-> <b>AREA_K = ${s.area_k.k}</b> (applied)`;}
  else $('msg').innerHTML='<span class=bad>'+s.error+'</span>';
  refresh();};
async function undo(){await fetch('/api/undo',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({cls})});refresh();}
async function clr(){await fetch('/api/clear',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({cls})});refresh();}
async function setfit(){await fetch('/api/fitparams',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({
   margin:+$('fmargin').value,sep:+$('fsep').value,pct:+$('fpct').value})});}
async function setdist(){await fetch('/api/dist',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify({d:+$('dist').value})});}
async function dofit(){const r=await (await fetch('/api/fit',{method:'POST'})).json();
  $('msg').innerHTML=r.ok?('<span class=ok>'+r.msg+'</span>'):('<span class=bad>'+r.msg+'</span>');
  refresh();}
async function save(){const r=await (await fetch('/api/save',{method:'POST'})).json();
  $('msg').innerHTML=r.ok?('<span class=ok>'+r.msg+'</span>'):('<span class=bad>'+r.msg+'</span>');}
function tbl(t,head,rows){t.innerHTML='<tr>'+head.map(h=>`<th>${h}</th>`).join('')+'</tr>'+
  rows.map(r=>'<tr>'+r.map(c=>`<td>${c}</td>`).join('')+'</tr>').join('');}
async function refresh(){
 try{
  const r=await (await fetch('/api/state')).json();
  tbl($('stats'),['class','n','mean L a b','min','max'],
    r.classes.map(c=>[c.name,c.n,c.mean||'--',c.min||'--',c.max||'--']));
  $('space').textContent=r.det.ok?('- classifying in '+r.det.space+
    ', min area '+r.det.min_area+
    ' | sent: '+(r.det.primary?r.det.primary.colour:'none')+' (largest blob)'):'- no frame yet';
  tbl($('det'),['colour','area','verdict'],
    (r.det.blobs||[]).map(b=>[b.colour,b.area,
      b.verdict==='ACCEPTED'?'<span class=ok>ACCEPTED</span>':
      '<span class=bad>'+b.verdict+'</span>']));
  tbl($('vals'),['name','value'],r.vals.map(v=>[v[0],v[1]]));
 }catch(e){}
}
setInterval(refresh,700); refresh();
</script></body></html>"""


@app.route("/favicon.ico")
def favicon():
    return ("", 204)       # keeps the browser console clean


@app.route("/")
def index():
    return PAGE


@app.route("/view")
def view_stream():
    mode = request.args.get("mode", "camera")

    def gen():
        while True:
            img = render(mode)
            if img is None:
                time.sleep(0.05)
                continue
            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg.tobytes() + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/sample", methods=["POST"])
def api_sample():
    body = request.get_json(force=True, silent=True) or {}
    cls = body.get("cls")
    if cls not in CLASSES:
        return jsonify({"ok": False, "error": "unknown class"}), 400
    s = sample_at(cls, body.get("x", 0), body.get("y", 0))
    if s is None:
        return jsonify({"ok": False, "error": "no camera frame yet"}), 503
    out = {"ok": True, "s": s}
    if cls in ("red", "green"):
        k = area_k_from_click(s)
        if k:
            out["area_k"] = k
    return jsonify(out)


def area_k_from_click(s):
    """If the click landed inside an accepted blob, turn its area into AREA_K."""
    rep = detector_report()
    for b in rep.get("blobs", []):
        x, y, w, h = b["box"]
        if x <= s["x"] < x + w and y <= s["y"] < y + h and b["area"] > 0:
            k = area_k["dist_mm"] * math.sqrt(b["area"])
            ob.PI.set("AREA_K", k)
            ob.sync_globals()
            try:
                prm.save_calibration(ob.PI)
            except OSError:
                pass
            area_k["result"] = k
            return {"area": b["area"], "dist": int(area_k["dist_mm"]), "k": int(k)}
    return None


@app.route("/api/undo", methods=["POST"])
def api_undo():
    cls = (request.get_json(force=True, silent=True) or {}).get("cls")
    with lock:
        if cls in samples and samples[cls]:
            samples[cls].pop()
    return jsonify({"ok": True})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    cls = (request.get_json(force=True, silent=True) or {}).get("cls")
    with lock:
        if cls in samples:
            samples[cls].clear()
    return jsonify({"ok": True})


@app.route("/api/fitparams", methods=["POST"])
def api_fitparams():
    b = request.get_json(force=True, silent=True) or {}
    for k in ("margin", "sep", "pct"):
        if k in b:
            try:
                FIT[k] = max(0, min(80, float(b[k])))
            except (TypeError, ValueError):
                pass
    return jsonify({"ok": True, "fit": FIT})


@app.route("/api/dist", methods=["POST"])
def api_dist():
    b = request.get_json(force=True, silent=True) or {}
    try:
        area_k["dist_mm"] = max(50.0, min(4000.0, float(b.get("d", 600))))
    except (TypeError, ValueError):
        pass
    return jsonify({"ok": True})


@app.route("/api/fit", methods=["POST"])
def api_fit():
    ok, text = fit()
    return jsonify({"ok": ok, "msg": text})


LAB_NAMES = ["USE_LAB"] + sorted(n for n in prm.CAL_KEYS if n != "USE_LAB")


@app.route("/api/state")
def api_state():
    return jsonify({
        "classes": [dict(name=c, **(stats(c) or {"n": 0})) for c in CLASSES],
        "det": detector_report(),
        "vals": [[n, round(float(ob.PI[n]), 1) if not isinstance(ob.PI[n], bool)
                  else ("on" if ob.PI[n] else "off")] for n in LAB_NAMES],
        "fit": FIT,
    })


@app.route("/api/save", methods=["POST"])
def api_save():
    """Writes vision_cal.json only - tuning.json (the rest of the Pi values and
    the STM32 table) is never touched by this tool."""
    try:
        text = prm.save_calibration(ob.PI)
    except OSError as e:
        return jsonify({"ok": False, "msg": f"could not save: {e}"}), 500
    return jsonify({"ok": True, "msg": text + (" - USE_LAB is on" if ob.PI["USE_LAB"] else "")})


# --------------------------------------------------------------------------

def main():
    global lidar
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lidar", action="store_true",
                    help="camera only (the lidar is not needed for Lab fitting)")
    args = ap.parse_args()

    note = prm.load(ob.PI, ob.STM)
    ob.sync_globals()
    print(note)

    ob.viewers = 1                      # make VisionThread keep frames for our views
    vis = ob.VisionThread()
    vis.start()

    if not args.no_lidar:
        try:
            from worldstate import SharedState
            from sensors.lidar import LidarThread
            lidar = LidarThread(SharedState())
            lidar.start()
        except Exception as e:
            print(f"[cal] lidar unavailable ({e}) - not needed for Lab fitting")
            lidar = None

    print(f"[cal] open http://<pi>:{PORT}   saves to {prm.CAL_PATH}")
    try:
        app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False, use_reloader=False)
    finally:
        vis.stop()
        vis.join(timeout=2.0)
        if lidar is not None:
            lidar.stop()
            lidar.join(timeout=2.0)
        print("[cal] stopped")


if __name__ == "__main__":
    main()
