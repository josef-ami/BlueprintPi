"""
tests/test_yolo_detector.py - the YOLO pillar detector (sensors/yolo_detector.py).

The geometry tests need only numpy. The model tests run once per backend that
is installed (ncnn / openvino / onnxruntime) and are skipped for the others,
so this passes on a laptop with none of them and checks all of them on the Pi.

    python3 -m pytest tests/test_yolo_detector.py -q

The frames in tests/data/yolo/ are real camera frames from the car (take/
recordings), not part of the training set's validation split bookkeeping -
they are a smoke test that the export, the preprocessing and the decoding
still agree, not an accuracy measurement.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sensors import yolo_detector as yd  # noqa: E402

DATA = os.path.join(ROOT, "tests", "data", "yolo")
TWO_SIGNS = "run_20260923_033127_00000.jpg"     # a red and a green pillar
EMPTY = "run_20260923_033204_00055.jpg"          # mat and walls, no pillar


def _load(name):
    cv2 = pytest.importorskip("cv2")
    bgr = cv2.imread(os.path.join(DATA, name))
    assert bgr is not None, name
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- geometry --

def test_letterbox_640x480_to_416():
    x, r, px, py = yd.letterbox(np.zeros((480, 640, 3), np.uint8), 416)
    assert x.shape == (1, 3, 416, 416) and x.dtype == np.float32
    assert r == pytest.approx(0.65)
    assert (px, py) == (0, 52)                    # 416x312 image, 52 px grey above/below
    assert x[0, :, 0, 0] == pytest.approx(114 / 255.0)
    assert x[0, :, 200, 200] == pytest.approx(0.0)


def test_decode_maps_back_to_camera_pixels():
    # one RED (class 1) box, centre (208, 208) 40x80 in model input pixels
    out = np.zeros((1, 6, 10), np.float32)
    out[0, :, 3] = [208, 208, 40, 80, 0.05, 0.9]
    dets = yd.decode(out, 0.5, 0.5, 0.65, 0, 52, 640, 480)
    assert len(dets) == 1
    cls, conf, x0, y0, x1, y1 = dets[0]
    assert cls == 1 and conf == pytest.approx(0.9)
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == (pytest.approx(320), pytest.approx(240))
    assert (x1 - x0, y1 - y0) == (pytest.approx(40 / 0.65), pytest.approx(80 / 0.65))


def test_nms_is_per_class():
    out = np.zeros((1, 6, 3), np.float32)
    out[0, :, 0] = [100, 100, 40, 80, 0.9, 0.0]    # green
    out[0, :, 1] = [102, 101, 40, 80, 0.8, 0.0]    # same green, lower score -> dropped
    out[0, :, 2] = [101, 100, 40, 80, 0.0, 0.7]    # red on top of it -> kept
    dets = yd.decode(out, 0.5, 0.5, 1.0, 0, 0, 416, 416)
    assert sorted(d[0] for d in dets) == [0, 1]


def test_below_threshold_is_empty():
    out = np.zeros((1, 6, 5), np.float32)
    assert yd.decode(out, 0.5, 0.5, 1.0, 0, 0, 416, 416) == []


# ------------------------------------------------------------------ model --

def _detector(backend):
    try:
        return yd.YoloDetector(backend=backend, threads=2)
    except Exception as e:
        pytest.skip(f"{backend} not available: {e}")


@pytest.mark.parametrize("backend", yd.BACKENDS)
def test_model_finds_both_colours(backend):
    det = _detector(backend)
    assert det.names == {0: "GREEN PILLAR", 1: "RED PILLAR"}
    found = det.detect(_load(TWO_SIGNS), conf=0.5)
    names = sorted(d.name for d in found)
    assert names == ["GREEN PILLAR", "RED PILLAR"], found
    red = next(d for d in found if d.name == "RED PILLAR")
    assert 380 < (red.x0 + red.x1) / 2 < 470 and red.h > 60     # the near red, right of centre


@pytest.mark.parametrize("backend", yd.BACKENDS)
def test_model_empty_frame(backend):
    det = _detector(backend)
    assert det.detect(_load(EMPTY), conf=0.5) == []


def test_backends_agree():
    dets = []
    for be in yd.BACKENDS:
        try:
            dets.append(yd.YoloDetector(backend=be, threads=2))
        except Exception:
            pass
    if len(dets) < 2:
        pytest.skip("needs two backends installed")
    for name in sorted(os.listdir(DATA)):
        rgb = _load(name)
        ref = [(d.cls, d.box()) for d in dets[0].detect(rgb, conf=0.5)]
        for other in dets[1:]:
            got = [(d.cls, d.box()) for d in other.detect(rgb, conf=0.5)]
            assert len(got) == len(ref), (name, other.backend)
            for (c1, b1), (c2, b2) in zip(ref, got):
                assert c1 == c2 and max(abs(a - b) for a, b in zip(b1, b2)) <= 2, name
