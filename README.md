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
