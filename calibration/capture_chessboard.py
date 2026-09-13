#!/usr/bin/env python3
"""
Step 1 of fisheye calibration: capture chessboard images.

Run on the Pi with the SAME camera and SAME resolution you run the robot at
(calibration is resolution-specific). Point the camera at a printed
chessboard and press ENTER to save each shot; aim for 15-30 shots covering:
  - board near each corner and edge of the frame (where fisheye distortion
    lives — this is what teaches k3/k4)
  - board tilted at various angles, not just flat-on
  - board at a few distances

Images are saved to calibration/captures/. Stop main.py / dashboard.py first
(one process holds the camera).

    python3 calibration/capture_chessboard.py
"""

import os
import sys
import time

# allow importing the robot's camera helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import sensors.camera as camera

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
# inner-corner count of your printed board (squares - 1 in each direction)
BOARD = (9, 6)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cam = camera.open_camera()          # uses config resolution / swap_rb
    print(f"Camera open at {camera.FRAME_W}x{camera.FRAME_H}. "
          f"Board inner corners = {BOARD}.")
    print("ENTER = save if a board is detected · q + ENTER = quit\n")

    saved = 0
    swap = camera.load_config().get("swap_rb", False)
    try:
        while True:
            frame = camera.grab_rgb(cam, swap)                 # RGB
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, BOARD,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            status = "BOARD FOUND" if found else "no board"
            cmd = input(f"[{saved} saved] {status} — action: ").strip().lower()
            if cmd == "q":
                break
            if found:
                path = os.path.join(OUT_DIR, f"cal_{saved:02d}.png")
                # save the RAW frame (BGR for cv2.imwrite) so calibrate re-detects
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                saved += 1
                print(f"  saved {path}")
            else:
                print("  not saved (no board detected) — reposition and retry")
    finally:
        try:
            cam.stop()
        except Exception:
            pass
        print(f"\nDone. {saved} images in {OUT_DIR}")
        if saved < 15:
            print("WARNING: fewer than 15 images — calibration may be poor. "
                  "Aim for 15-30 with good edge/corner coverage.")


if __name__ == "__main__":
    main()
