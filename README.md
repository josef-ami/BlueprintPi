# BluePrint Pi
## All that lives in the Pi5 of Team Blueprint's bot for WRO 2026 APAC

### Setup Procedure
    sudo apt install -y python3-libcamera python3-kms++ python3-picamera2
    python3 -m venv --system-site-packages env
    source env/bin/activate
    pip install -r requirements.txt


### Dashboard Daemon
    sudo chmod+x dashboard.py
    sudo cp robodash.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now robodash.service

    Open http://<pi>:8080 - two tabs: Calibration, and Obstacle run
    (runs obstacle_lap.py's loop in the dashboard; see docs/OBSTACLE_LAP.md section 10).
    Red/green only counts as a pillar when it stands on the white mat
    (floor filter, tuned on the Calibration tab; docs/OBSTACLE_LAP.md section 4.4a).

### Pillar detector (YOLO)
    obstacleRound.py finds the red / green pillars with a YOLO26n model trained
    on this car's own camera frames: models/pillars26/ (416 px, 2 classes).
    It runs on ncnn or onnxruntime (both in requirements.txt), no torch needed.
    Quick check / speed test on the Pi:
        python3 sensors/yolo_detector.py tests/data/yolo/*.jpg
        python3 -m pytest tests/test_yolo_detector.py -q
    Tune tab, "YOLO detector": USE_YOLO (off = the old Lab/HSV masking),
    backend, threads, confidence. If the model cannot load, masking is used.
    Retrain (pillars + parking lot, ~40 min at a practice round):
    tools/yolo/PROTOCOL.md. Record on the Pi (Tune tab, Recording), then on the PC
        python tools/yolo/yolo.py harvest      # pull, filter, pre-label, upload
    label in Roboflow, run tools/yolo/train_colab.ipynb in Colab, then
        python tools/yolo/yolo.py deploy       # check, copy to the Pi, switch live
    The Pi loads the model named in models/ACTIVE (default pillars26).
