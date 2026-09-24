#!/usr/bin/env python3
"""
tools/yolo/yolo.py - the PC side of the fast YOLO retrain loop.

    python tools/yolo/yolo.py harvest          pull + prep + upload, in one go
    python tools/yolo/yolo.py pull             new recording sessions from the Pi
    python tools/yolo/yolo.py prep             pick, pre-label and zip the new frames
    python tools/yolo/yolo.py upload [zip]     send a batch to Roboflow (API key)
    python tools/yolo/yolo.py deploy [zip]     check a trained model, copy it to the
                                               Pi and make it the active one
    python tools/yolo/yolo.py activate NAME    switch the Pi to models/NAME
    python tools/yolo/yolo.py models           models on the Pi, and which is active
    python tools/yolo/yolo.py status           what is pulled / labelled so far

Everything is driven by tools/yolo/config.json (Pi address, Roboflow project).
The whole protocol - what to record, how to label, the Colab notebook - is in
tools/yolo/PROTOCOL.md.

Needs on the PC: Python 3.9+, numpy, opencv-python, onnxruntime
(pip install -r tools/yolo/requirements.txt), and the OpenSSH client that ships
with Windows 10/11 (ssh must work: `ssh suntzu@<pi>`). `roboflow` is optional,
only for `upload`.
"""

import argparse
import glob
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

DATA = os.path.join(REPO, "datasets")          # git-ignored
RAW = os.path.join(DATA, "raw")                # one folder per Pi session
UPLOAD = os.path.join(DATA, "upload")          # one folder + zip per batch
USED = os.path.join(DATA, "used.txt")          # frames already looked at by prep

# Class ids in every label file this tool writes. Alphabetical, because that is
# the order Roboflow gives classes in its exports - so ids never shuffle between
# a batch and the dataset it lands in. The detector on the car matches classes
# by NAME, so the order does not matter there.
CLASSES = ["GREEN PILLAR", "PARKING LOT", "RED PILLAR"]

DEFAULT_CFG = {
    "pi_host": "192.168.137.45",
    "pi_user": "suntzu",
    "pi_repo": "/home/suntzu/BlueprintPi",
    "roboflow_workspace": "",
    "roboflow_project": "",
    "max_images": 400,          # per batch; ~10-15 min of review in Roboflow
    "empty_fraction": 0.10,     # share of the batch kept with nothing in it
    "prelabel_conf": 0.35,      # model boxes below this are not pre-drawn
    "dup_bits": 6,              # dHash distance: closer than this = duplicate
    "blur_min": 25.0,           # Laplacian variance below this = motion-blurred
    "dark_max": 25.0,           # mean grey below this = too dark
}


def load_cfg():
    cfg = dict(DEFAULT_CFG)
    p = os.path.join(HERE, "config.json")
    if os.path.exists(p):
        with open(p) as f:
            cfg.update(json.load(f))
    return cfg


def say(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# ssh helpers (Windows OpenSSH or any ssh on PATH)
# --------------------------------------------------------------------------

def ssh_cmd(cfg, remote):
    return ["ssh", "-o", "ConnectTimeout=8", f"{cfg['pi_user']}@{cfg['pi_host']}", remote]


def ssh_out(cfg, remote):
    r = subprocess.run(ssh_cmd(cfg, remote), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ssh failed ({r.returncode}): {r.stderr.strip() or remote}\n"
                         f"Check: ssh {cfg['pi_user']}@{cfg['pi_host']} works, and the Pi "
                         f"address in tools/yolo/config.json.")
    return r.stdout


def _safe_extract(tf, dest):
    try:
        tf.extractall(dest, filter="data")        # Python 3.12+
    except TypeError:
        tf.extractall(dest)


# --------------------------------------------------------------------------
# pull
# --------------------------------------------------------------------------

def cmd_pull(cfg, args):
    os.makedirs(RAW, exist_ok=True)
    take = f"{cfg['pi_repo']}/take"
    listing = ssh_out(cfg, f"ls -1 {shlex.quote(take)} 2>/dev/null || true").split()
    sessions = [s for s in listing if not s.startswith(".")]
    new = [s for s in sessions if not os.path.isdir(os.path.join(RAW, s))]
    if not sessions:
        say("Nothing recorded on the Pi yet (take/ is empty). Turn RECORD_RUNS on, or use "
            "the Record button, then drive or push the car around.")
        return []
    if not new:
        say(f"All {len(sessions)} sessions on the Pi are already pulled.")
        return []
    say(f"Pulling {len(new)} new session(s): {', '.join(new)}")
    t0 = time.monotonic()
    # one tar stream over one ssh connection: far faster than scp -r for
    # thousands of small JPEGs
    remote = f"tar -C {shlex.quote(take)} -cf - " + " ".join(shlex.quote(s) for s in new)
    p = subprocess.Popen(ssh_cmd(cfg, remote), stdout=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        _safe_extract(tf, RAW)
    if p.wait() != 0:
        raise SystemExit("ssh/tar failed while pulling - partial sessions may be in "
                         + RAW + "; delete them and pull again")
    n = sum(len(glob.glob(os.path.join(RAW, s, "*.jpg"))) for s in new)
    say(f"  {n} frames in {time.monotonic() - t0:.0f} s -> {RAW}")
    return new


# --------------------------------------------------------------------------
# prep: pick the frames worth labelling, pre-label them, zip for Roboflow
# --------------------------------------------------------------------------

def dhash(gray):
    """64-bit difference hash of a grey image."""
    import cv2
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def parking_proposal(bgr, min_frac=0.0015):
    """(x0, y0, x1, y1) around every magenta blob, or None. WRO parking-lot
    walls are the only magenta on the field, so their union is a first guess
    at the whole lot; the review in Roboflow fixes the edges.
    Bright only (V >= 110): this camera renders black walls and dark doors as a
    dark purple (V ~60) that is magenta by hue. And low in the frame: a blob
    that ends above the top 35% is room, not mat."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (135, 110, 110), (165, 255, 255))  # magenta, clear of red
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    h, w = m.shape
    boxes = [stats[i] for i in range(1, n)
             if stats[i][4] >= min_frac * w * h and stats[i][1] + stats[i][3] > 0.35 * h]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
    return (float(x0), float(y0), float(x1), float(y1))


def class_of(name):
    """Model class name -> index in CLASSES, or None."""
    n = name.upper()
    if "RED" in n:
        return CLASSES.index("RED PILLAR")
    if "GREEN" in n:
        return CLASSES.index("GREEN PILLAR")
    if "PARK" in n or "MAGENTA" in n:
        return CLASSES.index("PARKING LOT")
    return None


def local_active_model():
    """The newest model folder in models/ that deploy put there, else pillars26."""
    from sensors.yolo_detector import DEFAULT_MODEL_DIR
    act = os.path.join(REPO, "models", "ACTIVE")
    if os.path.exists(act):
        d = os.path.join(REPO, "models", open(act).read().strip())
        if os.path.isdir(d):
            return d
    return DEFAULT_MODEL_DIR


def load_detector(model_dir=None):
    from sensors.yolo_detector import YoloDetector
    d = model_dir or local_active_model()
    try:
        det = YoloDetector(d, backend="onnx", threads=4)
    except Exception:
        det = YoloDetector(d, backend="auto", threads=4)
    return det


def contact_sheet(items, path, cols=6, tile=(320, 240)):
    """A preview grid of the batch with the pre-labels drawn, to eyeball it."""
    import cv2
    colours = {0: (0, 200, 0), 1: (255, 0, 255), 2: (0, 0, 255)}
    tiles = []
    for img_path, labels in items[:cols * 8]:
        im = cv2.imread(img_path)
        h, w = im.shape[:2]
        for c, cx, cy, bw, bh in labels:
            x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(im, (x0, y0), (x1, y1), colours[c], 3)
        tiles.append(cv2.resize(im, tile))
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 80])


def cmd_prep(cfg, args):
    import cv2
    used = set()
    if os.path.exists(USED):
        used = set(open(USED).read().split())
    frames = sorted(p for p in glob.glob(os.path.join(RAW, "*", "*.jpg"))
                    if os.path.relpath(p, RAW).replace("\\", "/") not in used)
    if not frames:
        say("No new frames to prepare (pull first, or everything is already in a batch).")
        return None
    say(f"{len(frames)} new frames. Filtering blur / dark / duplicates ...")

    det = load_detector(args.model)
    say(f"  pre-labelling with {os.path.basename(det.model_dir)} ({det.backend}): "
        f"{', '.join(det.names.values())}")

    kept, recent = [], {}
    n_blur = n_dark = n_dup = 0
    t0 = time.monotonic()
    for i, p in enumerate(frames):
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
        if small.mean() < cfg["dark_max"]:
            n_dark += 1
            continue
        if cv2.Laplacian(small, cv2.CV_64F).var() < cfg["blur_min"]:
            n_blur += 1
            continue
        h = dhash(small)
        sess = os.path.basename(os.path.dirname(p))
        hist = recent.setdefault(sess, [])
        if any(hamming(h, o) < cfg["dup_bits"] for o in hist[-60:]):
            n_dup += 1
            continue
        hist.append(h)

        labels = []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = bgr.shape[:2]
        have_park = False
        for d in det.detect(rgb, conf=cfg["prelabel_conf"]):
            c = class_of(d.name)
            if c is None:
                continue
            have_park |= (CLASSES[c] == "PARKING LOT")
            labels.append((c, (d.x0 + d.x1) / 2 / W, (d.y0 + d.y1) / 2 / H, d.w / W, d.h / H))
        if not have_park:
            pb = parking_proposal(bgr)
            if pb is not None:
                x0, y0, x1, y1 = pb
                labels.append((CLASSES.index("PARKING LOT"), (x0 + x1) / 2 / W,
                               (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
        kept.append((p, sess, labels))
        if (i + 1) % 200 == 0:
            say(f"  {i + 1}/{len(frames)} ...")

    with_obj = [k for k in kept if k[2]]
    empty = [k for k in kept if not k[2]]
    n_empty = min(len(empty), int(round(cfg["empty_fraction"] * max(1, len(with_obj))
                                        / max(1e-6, 1 - cfg["empty_fraction"]))))
    pick = with_obj + [empty[int(j)] for j in np.linspace(0, len(empty) - 1, n_empty)] \
        if n_empty else list(with_obj)
    pick.sort(key=lambda k: k[0])
    cap = args.max or cfg["max_images"]
    if len(pick) > cap:                        # even spread over time, every session
        pick = [pick[int(j)] for j in np.linspace(0, len(pick) - 1, cap)]
    say(f"  dropped {n_dark} dark, {n_blur} blurred, {n_dup} near-duplicates; "
        f"{len(with_obj)} with something in view, {len(empty)} empty -> batch of {len(pick)} "
        f"({time.monotonic() - t0:.0f} s)")
    if not pick:
        return None

    batch = args.name or time.strftime("batch_%Y%m%d_%H%M%S")
    out = os.path.join(UPLOAD, batch)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "images"))
    os.makedirs(os.path.join(out, "labels"))
    counts = [0] * len(CLASSES)
    sheet = []
    for p, sess, labels in pick:
        base = os.path.basename(p)
        dst = os.path.join(out, "images", base)
        shutil.copy2(p, dst)
        with open(os.path.join(out, "labels", base[:-4] + ".txt"), "w") as f:
            for c, cx, cy, bw, bh in labels:
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                counts[c] += 1
        sheet.append((dst, labels))
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"# {batch}: {len(pick)} frames pre-labelled by "
                f"{os.path.basename(det.model_dir)} + magenta proposal\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {json.dumps(CLASSES)}\n")
    contact_sheet(sheet, os.path.join(UPLOAD, batch + "_preview.jpg"))
    zpath = os.path.join(UPLOAD, batch + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:     # JPEGs don't compress
        for root, _, files in os.walk(out):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, out))
    with open(USED, "a") as f:             # every frame looked at, picked or not:
        for p in frames:                   #   the next prep only sees new ones
            f.write(os.path.relpath(p, RAW).replace("\\", "/") + "\n")
    say(f"  pre-labels: " + ", ".join(f"{n} {c}" for c, n in zip(CLASSES, counts)))
    say(f"Batch ready: {zpath}\n  preview: {os.path.join(UPLOAD, batch + '_preview.jpg')}")
    return zpath


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------

def cmd_upload(cfg, args, zpath=None):
    zpath = zpath or args.zip or newest(os.path.join(UPLOAD, "*.zip"))
    if not zpath:
        raise SystemExit("No batch zip found - run prep first.")
    ws, proj = cfg["roboflow_workspace"], cfg["roboflow_project"]
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    manual = (f"Upload it by hand: app.roboflow.com -> your project -> Upload Data -> drop\n"
              f"  {zpath}\n  (it carries data.yaml, so the labels come in as boxes).")
    if not (ws and proj and key):
        say("Roboflow API not set up (roboflow_workspace / roboflow_project in "
            "tools/yolo/config.json and the ROBOFLOW_API_KEY environment variable).\n" + manual)
        return
    try:
        import roboflow
    except ImportError:
        say("pip install roboflow  for automatic upload.\n" + manual)
        return
    batch = os.path.splitext(os.path.basename(zpath))[0]
    say(f"Uploading {batch} to {ws}/{proj} ...")
    t0 = time.monotonic()
    rf = roboflow.Roboflow(api_key=key)
    rf.workspace(ws).upload_dataset(zpath, proj, batch_name=batch)
    say(f"  done in {time.monotonic() - t0:.0f} s. In Roboflow: Annotate -> {batch} -> "
        f"review every image, then Generate a new version.")


def newest(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# --------------------------------------------------------------------------
# deploy
# --------------------------------------------------------------------------

def _find_model_root(folder):
    for root, dirs, files in os.walk(folder):
        if "best.onnx" in files or "best_ncnn_model" in dirs:
            return root
    return None


def _quick_eval(det, images):
    import cv2
    per = {n: 0 for n in det.names.values()}
    ms = []
    for p in images:
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        for d in det.detect(rgb, conf=0.5):
            per[d.name] = per.get(d.name, 0) + 1
        ms.append(det.infer_ms)
    return per, (sum(ms) / len(ms) if ms else 0.0)


def cmd_deploy(cfg, args):
    zpath = args.zip
    if not zpath:
        dl = os.path.join(os.path.expanduser("~"), "Downloads")
        zpath = newest(os.path.join(dl, "*.zip"))
        if not zpath:
            raise SystemExit("Give the model zip from Colab: yolo.py deploy <file.zip>")
        say(f"Newest zip in Downloads: {zpath}")
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    root = _find_model_root(tmp)
    if root is None:
        raise SystemExit(f"{zpath} has no best.onnx / best_ncnn_model - is it the Colab export?")
    name = args.name or os.path.basename(root.rstrip("/\\"))
    if name in ("", ".", "weights") or name.startswith("tmp"):
        name = os.path.splitext(os.path.basename(zpath))[0]
    dest = os.path.join(REPO, "models", name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(root, dest)
    say(f"Model {name} -> {dest}")

    # 1. does it load, with sign classes the car understands?
    new = load_detector(dest)
    names = list(new.names.values())
    say(f"  classes: {names}")
    if not any("RED" in n.upper() for n in names) or not any("GREEN" in n.upper() for n in names):
        raise SystemExit("  REFUSED: no RED / GREEN class - the car would steer by nothing.")

    # 2. side by side with the model the car runs now, on the same frames
    imgs = sorted(glob.glob(os.path.join(REPO, "tests", "data", "yolo", "*.jpg")))
    dirs = [d for d in glob.glob(os.path.join(UPLOAD, "batch_*")) if os.path.isdir(d)]
    last = max(dirs, key=os.path.getmtime) if dirs else None
    if last:
        imgs += sorted(glob.glob(os.path.join(last, "images", "*.jpg")))[:40]
    if imgs:
        old = load_detector(local_active_model() if not args.against else args.against)
        o, oms = _quick_eval(old, imgs)
        n, nms = _quick_eval(new, imgs)
        say(f"  detections on {len(imgs)} frames (conf 0.5):")
        for k in sorted(set(o) | set(n)):
            say(f"    {k:14s} old {o.get(k, 0):4d}   new {n.get(k, 0):4d}")
        say(f"    PC inference   old {oms:5.1f} ms   new {nms:5.1f} ms")

    if args.no_pi:
        say("  (not copied to the Pi: --no-pi)")
        return
    # 3. to the Pi, then switch
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        tf.add(dest, arcname=name)
    models = f"{cfg['pi_repo']}/models"
    remote = (f"mkdir -p {shlex.quote(models)} && rm -rf {shlex.quote(models + '/' + name)} && "
              f"tar -C {shlex.quote(models)} -xf - && "
              f"echo {shlex.quote(name)} > {shlex.quote(models + '/ACTIVE')} && "
              f"cat {shlex.quote(models + '/ACTIVE')}")
    r = subprocess.run(ssh_cmd(cfg, remote), input=buf.getvalue(), capture_output=True)
    if r.returncode != 0:
        raise SystemExit("copy to the Pi failed: " + r.stderr.decode(errors="replace"))
    with open(os.path.join(REPO, "models", "ACTIVE"), "w") as f:
        f.write(name + "\n")
    say(f"  on the Pi and ACTIVE = {r.stdout.decode().strip()}. obstacleRound.py reloads it "
        f"within 2 s (page: detector 'YOLO ncnn {name}'). Roll back: yolo.py activate pillars26")


def cmd_activate(cfg, args):
    models = f"{cfg['pi_repo']}/models"
    out = ssh_out(cfg, f"test -d {shlex.quote(models + '/' + args.name)} && "
                       f"echo {shlex.quote(args.name)} > {shlex.quote(models + '/ACTIVE')} && "
                       f"echo ok || echo missing")
    if "missing" in out:
        raise SystemExit(f"models/{args.name} is not on the Pi (yolo.py models lists them)")
    with open(os.path.join(REPO, "models", "ACTIVE"), "w") as f:
        f.write(args.name + "\n")
    say(f"Pi ACTIVE = {args.name}; reloads within 2 s.")


def cmd_models(cfg, args):
    models = f"{cfg['pi_repo']}/models"
    out = ssh_out(cfg, f"cd {shlex.quote(models)} && for d in */; do echo \"${{d%/}}\"; done; "
                       f"echo '--'; cat ACTIVE 2>/dev/null || echo pillars26")
    names, active = out.split("--")
    active = active.strip()
    for n in names.split():
        say(("* " if n == active else "  ") + n)


def cmd_status(cfg, args):
    sess = sorted(glob.glob(os.path.join(RAW, "*")))
    n = sum(len(glob.glob(os.path.join(s, "*.jpg"))) for s in sess)
    used = len(open(USED).read().split()) if os.path.exists(USED) else 0
    batches = sorted(glob.glob(os.path.join(UPLOAD, "*.zip")))
    say(f"pulled: {len(sess)} sessions, {n} frames; already prepped: {used}; "
        f"batches: {len(batches)}" + (f" (last {os.path.basename(batches[-1])})" if batches else ""))


def cmd_harvest(cfg, args):
    cmd_pull(cfg, args)
    z = cmd_prep(cfg, args)
    if z:
        cmd_upload(cfg, args, z)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("harvest", "prep"):
        s = sub.add_parser(name)
        s.add_argument("--max", type=int, default=0, help="batch size cap (config max_images)")
        s.add_argument("--name", default="", help="batch name (default batch_<date>_<time>)")
        s.add_argument("--model", default=None, help="model folder to pre-label with")
        s.add_argument("--zip", default=None)
    sub.add_parser("pull")
    s = sub.add_parser("upload"); s.add_argument("zip", nargs="?")
    s = sub.add_parser("deploy")
    s.add_argument("zip", nargs="?", help="model zip from Colab (default: newest in Downloads)")
    s.add_argument("--name", default="")
    s.add_argument("--no-pi", action="store_true", help="check and copy into models/ only")
    s.add_argument("--against", default=None, help="model folder to compare with")
    s = sub.add_parser("activate"); s.add_argument("name")
    sub.add_parser("models")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    {"pull": cmd_pull, "prep": cmd_prep, "upload": cmd_upload, "deploy": cmd_deploy,
     "activate": cmd_activate, "models": cmd_models, "status": cmd_status,
     "harvest": cmd_harvest}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
