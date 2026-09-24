"""
tests/test_yolo_tools.py - the PC side of the retrain loop (tools/yolo/yolo.py).

No Pi, no network: prep runs on frames copied from tests/data/yolo into a
temporary raw/ folder, with the model in models/pillars26.

    python3 -m pytest tests/test_yolo_tools.py -q
"""

import glob
import os
import sys
import zipfile

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "yolo"))
import yolo  # noqa: E402

FRAMES = sorted(glob.glob(os.path.join(ROOT, "tests", "data", "yolo", "*.jpg")))


def test_classes_are_alphabetical_like_roboflow():
    assert yolo.CLASSES == sorted(yolo.CLASSES)
    assert yolo.class_of("RED PILLAR") == yolo.CLASSES.index("RED PILLAR")
    assert yolo.class_of("green pillar") == yolo.CLASSES.index("GREEN PILLAR")
    assert yolo.class_of("PARKING LOT") == yolo.CLASSES.index("PARKING LOT")
    assert yolo.class_of("person") is None


def test_dhash_sees_duplicates_not_different_frames():
    a = cv2.cvtColor(cv2.imread(FRAMES[0]), cv2.COLOR_BGR2GRAY)
    noisy = np.clip(a.astype(int) + 3, 0, 255).astype(np.uint8)
    b = cv2.cvtColor(cv2.imread(FRAMES[5]), cv2.COLOR_BGR2GRAY)
    assert yolo.hamming(yolo.dhash(a), yolo.dhash(noisy)) < 6
    assert yolo.hamming(yolo.dhash(a), yolo.dhash(b)) >= 6


def test_parking_proposal_spans_both_walls_and_ignores_dark_purple():
    im = cv2.imread(FRAMES[3])
    assert yolo.parking_proposal(im) is None                      # no magenta in the room shots
    two = im.copy()
    cv2.rectangle(two, (100, 260), (160, 300), (160, 40, 170), -1)
    cv2.rectangle(two, (300, 262), (360, 300), (170, 45, 175), -1)
    x0, y0, x1, y1 = yolo.parking_proposal(two)
    assert x0 <= 100 and x1 >= 360                                # one box over the whole lot
    dark = im.copy()
    cv2.rectangle(dark, (100, 260), (300, 300), (60, 20, 60), -1)  # V ~60 purple = wall shadow
    assert yolo.parking_proposal(dark) is None


def test_prep_builds_a_roboflow_ready_zip(tmp_path, monkeypatch):
    raw, up = tmp_path / "raw", tmp_path / "upload"
    sess = raw / "run_20260924_100000"
    sess.mkdir(parents=True)
    for i, p in enumerate(FRAMES):
        im = cv2.imread(p)
        cv2.imwrite(str(sess / f"a_{i:03d}.jpg"), im)
        cv2.imwrite(str(sess / f"b_{i:03d}.jpg"), np.clip(im.astype(int) + 2, 0, 255).astype(np.uint8))
    monkeypatch.setattr(yolo, "RAW", str(raw))
    monkeypatch.setattr(yolo, "UPLOAD", str(up))
    monkeypatch.setattr(yolo, "USED", str(tmp_path / "used.txt"))

    class A:
        max, name, model, zip = 0, "batch_test", None, None
    try:
        z = yolo.cmd_prep(yolo.load_cfg(), A())
    except RuntimeError as e:                                     # no onnxruntime / ncnn here
        pytest.skip(str(e))
    names = zipfile.ZipFile(z).namelist()
    imgs = [n for n in names if n.startswith("images/")]
    assert "data.yaml" in names
    assert len(imgs) <= len(FRAMES)                               # the +2 copies are duplicates
    for n in imgs:
        assert "labels/" + os.path.basename(n)[:-4] + ".txt" in names
    lab = zipfile.ZipFile(z).read("labels/" + os.path.basename(imgs[0])[:-4] + ".txt").decode()
    for line in lab.split("\n"):
        if line:
            c, *xywh = line.split()
            assert 0 <= int(c) < len(yolo.CLASSES) and all(0 <= float(v) <= 1 for v in xywh)
    # a second prep finds nothing new
    assert yolo.cmd_prep(yolo.load_cfg(), A()) is None
