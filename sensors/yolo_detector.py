"""
yolo_detector.py - the trained pillar detector (YOLO26n, models/pillars26/).

One class, three interchangeable backends for the same trained weights:

    ncnn      models/pillars26/best_ncnn_model/     usually fastest on a Pi 5
    openvino  models/pillars26/best_openvino_model/
    onnx      models/pillars26/best.onnx            (onnxruntime)

"auto" tries them in that order and keeps the first one that loads, so the Pi
only needs ONE of  `pip install ncnn`,  `pip install openvino`  or
`pip install onnxruntime`. No torch and no ultralytics at run time.

THE MODEL (what the export actually contains - checked on all three files)
    input   1 x 3 x 416 x 416, RGB, 0..1, letterboxed (grey 114 padding)
    output  1 x 6 x 3549: per anchor cx, cy, w, h (input pixels) then one
            sigmoid score per class. Exported with end2end=False, so NMS is
            done here.
    classes read from the metadata.yaml next to the weights (or the ONNX
            metadata): 0 GREEN PILLAR, 1 RED PILLAR for pillars26.

detect() takes the camera's RGB frame at any size (640 x 480 on the car) and
returns boxes in THAT frame's pixels, best first. It never raises on a bad
frame; construction raises if no backend can load the model.
"""

import ast
import os
import time

import numpy as np

try:
    import cv2
except ImportError:                     # resize falls back to numpy (tests only)
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(HERE), "models", "pillars26")

BACKENDS = ("ncnn", "openvino", "onnx")
PAD_VALUE = 114


class Detection:
    __slots__ = ("cls", "name", "conf", "x0", "y0", "x1", "y1")

    def __init__(self, cls, name, conf, x0, y0, x1, y1):
        self.cls, self.name, self.conf = cls, name, conf
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def box(self):
        return (int(round(self.x0)), int(round(self.y0)),
                int(round(self.x1)), int(round(self.y1)))

    def __repr__(self):
        return (f"Detection({self.name} {self.conf:.2f} "
                f"[{self.x0:.0f},{self.y0:.0f},{self.x1:.0f},{self.y1:.0f}])")


# --------------------------------------------------------------------------
# model metadata
# --------------------------------------------------------------------------

def _read_names_yaml(path):
    """names + imgsz from an Ultralytics metadata.yaml, without needing PyYAML."""
    names, imgsz, section = {}, None, None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if not line.startswith(" "):
                section = line.split(":")[0].strip()
                continue
            item = line.strip()
            if section == "names" and ":" in item:
                k, v = item.split(":", 1)
                try:
                    names[int(k)] = v.strip().strip("'\"")
                except ValueError:
                    pass
            elif section == "imgsz" and item.startswith("-"):
                imgsz = int(item[1:].strip()) if imgsz is None else imgsz
    return names, imgsz


# --------------------------------------------------------------------------
# pre / post processing (identical for every backend)
# --------------------------------------------------------------------------

def letterbox(rgb, size):
    """RGB HxWx3 uint8 -> (1x3xSxS float32 0..1, scale, pad_x, pad_y).
    Same geometry as Ultralytics' LetterBox(center=True): keep aspect, pad
    both sides equally with grey 114."""
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        if cv2 is not None:
            img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        else:                                           # nearest, tests only
            ys = (np.arange(nh) / r).astype(int).clip(0, h - 1)
            xs = (np.arange(nw) / r).astype(int).clip(0, w - 1)
            img = rgb[ys][:, xs]
    else:
        img = rgb
    px, py = (size - nw) / 2.0, (size - nh) / 2.0
    top, left = int(round(py - 0.1)), int(round(px - 0.1))
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, left, top


def nms(boxes, scores, iou):
    """Plain greedy NMS. boxes Nx4 xyxy. Returns kept indices, best first."""
    order = scores.argsort()[::-1]
    keep = []
    x0, y0, x1, y1 = boxes.T
    area = (x1 - x0).clip(0) * (y1 - y0).clip(0)
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest]); iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest]); iy1 = np.minimum(y1[i], y1[rest])
        inter = (ix1 - ix0).clip(0) * (iy1 - iy0).clip(0)
        ovl = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[ovl <= iou]
    return keep


def decode(out, conf, iou, scale, pad_x, pad_y, src_w, src_h, max_det=20):
    """Raw (4+nc) x N head output -> [(cls, conf, x0, y0, x1, y1)] in source
    pixels, best first. Class-aware NMS (a red box never suppresses a green)."""
    out = np.asarray(out, dtype=np.float32)
    out = out.reshape(out.shape[-2], out.shape[-1])     # (4 + classes) x anchors
    cls_scores = out[4:]
    cls = cls_scores.argmax(0)
    sc = cls_scores.max(0)
    m = sc >= conf
    if not m.any():
        return []
    cx, cy, w, h = out[0, m], out[1, m], out[2, m], out[3, m]
    cls, sc = cls[m], sc[m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    off = cls[:, None].astype(np.float32) * 4096.0      # class-aware NMS trick
    keep = nms(boxes + off, sc, iou)[:max_det]
    res = []
    for i in keep:
        bx0 = (boxes[i, 0] - pad_x) / scale
        by0 = (boxes[i, 1] - pad_y) / scale
        bx1 = (boxes[i, 2] - pad_x) / scale
        by1 = (boxes[i, 3] - pad_y) / scale
        res.append((int(cls[i]), float(sc[i]),
                    float(np.clip(bx0, 0, src_w)), float(np.clip(by0, 0, src_h)),
                    float(np.clip(bx1, 0, src_w)), float(np.clip(by1, 0, src_h))))
    return res


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

class _Ncnn:
    def __init__(self, d, threads):
        import ncnn
        p = os.path.join(d, "best_ncnn_model", "model.ncnn.param")
        b = os.path.join(d, "best_ncnn_model", "model.ncnn.bin")
        if not (os.path.exists(p) and os.path.exists(b)):
            raise FileNotFoundError(p)
        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = int(threads)
        if self.net.load_param(p) != 0 or self.net.load_model(b) != 0:
            raise RuntimeError("ncnn could not load the model")
        self.meta = os.path.join(d, "best_ncnn_model", "metadata.yaml")

    def __call__(self, x):
        ex = self.net.create_extractor()
        ex.input("in0", self.ncnn.Mat(x[0]))
        ret, out = ex.extract("out0")
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed ({ret})")
        return np.array(out)


class _OpenVino:
    def __init__(self, d, threads):
        import openvino as ov
        xml = os.path.join(d, "best_openvino_model", "best.xml")
        if not os.path.exists(xml):
            raise FileNotFoundError(xml)
        core = ov.Core()
        self.model = core.compile_model(core.read_model(xml), "CPU",
                                        {"INFERENCE_NUM_THREADS": int(threads)})
        self.req = self.model.create_infer_request()
        self.out = self.model.output(0)
        self.meta = os.path.join(d, "best_openvino_model", "metadata.yaml")

    def __call__(self, x):
        return self.req.infer({0: x})[self.out]


class _Onnx:
    def __init__(self, d, threads):
        import onnxruntime as ort
        path = os.path.join(d, "best.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = ast.literal_eval(meta["names"]) if "names" in meta else None
        self.imgsz = ast.literal_eval(meta["imgsz"])[0] if "imgsz" in meta else None
        self.meta = None

    def __call__(self, x):
        return self.sess.run(None, {self.inp: x})[0]


_LOADERS = {"ncnn": _Ncnn, "openvino": _OpenVino, "onnx": _Onnx}


class YoloDetector:
    """
    det = YoloDetector()                 # auto backend, models/pillars26
    for d in det.detect(rgb_frame, conf=0.5):
        d.name, d.conf, d.box()          # box in rgb_frame pixels
    det.backend, det.infer_ms, det.names
    """

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, backend="auto", threads=3, imgsz=None):
        order = BACKENDS if backend in (None, "", "auto") else (backend,)
        errors = []
        self.run = None
        for name in order:
            try:
                self.run = _LOADERS[name](model_dir, threads)
                self.backend = name
                break
            except Exception as e:                      # missing package or file
                errors.append(f"{name}: {type(e).__name__}: {e}")
        if self.run is None:
            raise RuntimeError("no YOLO backend could load " + model_dir + " - "
                               + "; ".join(errors))
        self.load_errors = errors
        names, size = {}, None
        if getattr(self.run, "meta", None) and os.path.exists(self.run.meta):
            names, size = _read_names_yaml(self.run.meta)
        if not names and getattr(self.run, "names", None):
            names = dict(self.run.names)
        size = size or getattr(self.run, "imgsz", None)
        self.names = names or {0: "GREEN PILLAR", 1: "RED PILLAR"}
        self.imgsz = int(imgsz or size or 416)
        self.model_dir = model_dir
        self.threads = threads
        self.infer_ms = 0.0

    def detect(self, rgb, conf=0.5, iou=0.5, max_det=20):
        t0 = time.monotonic()
        x, r, px, py = letterbox(rgb, self.imgsz)
        out = self.run(x)
        raw = decode(out, conf, iou, r, px, py, rgb.shape[1], rgb.shape[0], max_det)
        self.infer_ms = (time.monotonic() - t0) * 1000.0
        return [Detection(c, self.names.get(c, str(c)), s, x0, y0, x1, y1)
                for (c, s, x0, y0, x1, y1) in raw]


if __name__ == "__main__":
    # python3 sensors/yolo_detector.py [image ...]  - quick check / benchmark on the Pi
    import sys
    imgs = sys.argv[1:]
    for be in BACKENDS:
        try:
            det = YoloDetector(backend=be)
        except Exception as e:
            print(f"{be:9s} not available: {e}")
            continue
        frames = []
        for p in imgs or [None]:
            if p is None:
                frames.append(np.full((480, 640, 3), 128, np.uint8))
            else:
                frames.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        det.detect(frames[0])                           # warm-up
        times, found = [], []
        for _ in range(max(1, 20 // len(frames))):
            for f in frames:
                found = det.detect(f)
                times.append(det.infer_ms)
        print(f"{be:9s} {np.median(times):6.1f} ms median over {len(times)} frames "
              f"({det.threads} threads)  last: {found}")
