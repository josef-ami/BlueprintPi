#!/usr/bin/env python3
"""
Step 2 of fisheye calibration: compute K and D, write them to config.json.

Reads the chessboard images from calibration/captures/, detects corners,
runs cv2.fisheye.calibrate, reports the reprojection error, and (on success)
saves the intrinsics into config.json under a "camera_intrinsics" block:

    "camera_intrinsics": {
        "model": "fisheye",
        "image_size": [640, 480],
        "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
        "D": [k1, k2, k3, k4],
        "rms_reproj_error": <float>
    }

This can run off the Pi (any machine with opencv) since it only needs the
saved images. Run once per lens; K/D are physical and don't change per run.

    python3 calibration/calibrate_fisheye.py

Board geometry — EDIT THESE to match your printed board:
"""

import glob
import json
import os
import sys

import cv2
import numpy as np

BOARD = (9, 6)          # inner corners (must match capture)
SQUARE_MM = 25.0        # side length of one printed square, in mm

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_DIR = os.path.join(HERE, "captures")
CONFIG = os.path.join(os.path.dirname(HERE), "config.json")


def main():
    images = sorted(glob.glob(os.path.join(CAP_DIR, "*.png")))
    if len(images) < 5:
        print(f"Only {len(images)} images in {CAP_DIR}. Need >=15 for a good "
              f"result; refusing below 5.")
        sys.exit(1)

    # one board's 3D points, in board coordinates (z=0 plane), scaled to mm.
    # fisheye.calibrate wants object points shaped (1, N, 3).
    objp = np.zeros((1, BOARD[0] * BOARD[1], 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_MM

    objpoints, imgpoints = [], []
    image_size = None
    subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    used = 0
    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]      # (w, h)
        elif gray.shape[::-1] != image_size:
            print(f"  skip {os.path.basename(path)}: size mismatch "
                  f"{gray.shape[::-1]} vs {image_size}")
            continue
        found, corners = cv2.findChessboardCorners(
            gray, BOARD,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"  skip {os.path.basename(path)}: no corners")
            continue
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix)
        objpoints.append(objp)
        imgpoints.append(corners.reshape(1, -1, 2))
        used += 1
        print(f"  ok   {os.path.basename(path)}")

    if used < 5:
        print(f"Only {used} usable images. Recapture with clearer boards.")
        sys.exit(1)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    # Flag constants live under cv2.fisheye.* in some OpenCV builds and only
    # under cv2.* in others (and occasionally neither name is generated).
    # Resolve each from wherever it exists, falling back to its stable integer
    # value so this works regardless of build.
    def _flag(name, value):
        return getattr(cv2.fisheye, name, getattr(cv2, name, value))

    flags = (_flag("CALIB_RECOMPUTE_EXTRINSIC", 2)
             + _flag("CALIB_FIX_SKEW", 8)
             + _flag("CALIB_CHECK_COND", 4))
    try:
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            objpoints, imgpoints, image_size, K, D,
            None, None, flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    except cv2.error as e:
        # CALIB_CHECK_COND throws on a bad image; tell the user which to drop
        print(f"\ncalibrate failed: {e}\n"
              f"One or more images are ill-conditioned (board too edge-on or "
              f"blurry). Remove the worst captures and retry, or drop "
              f"CALIB_CHECK_COND.")
        sys.exit(1)

    print(f"\nUsed {used} images. RMS reprojection error: {rms:.4f} px")
    if rms > 1.0:
        print("WARNING: error > 1px. Coverage or corner quality is weak; "
              "recapturing with more edge/corner shots will help.")

    intr = {
        "model": "fisheye",
        "image_size": [int(image_size[0]), int(image_size[1])],
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "rms_reproj_error": float(rms),
    }

    with open(CONFIG) as f:
        cfg = json.load(f)
    cfg["camera_intrinsics"] = intr
    tmp = CONFIG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG)

    print("\nSaved camera_intrinsics to config.json:")
    print(json.dumps(intr, indent=2))
    print("\nNow px_to_bearing will undistort blob centres through K/D. "
          "Restart main.py / dashboard.py to load it.")


if __name__ == "__main__":
    main()
