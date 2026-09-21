"""
perception/telemetry.py - the Pi side of the STM32 link, for odometry.

THE STM32 ALREADY SENDS THIS. firmware/ObstacleRound.cpp::sendTelem() puts a
21-byte TELEM frame on the wire every 20 ms; nothing needs adding there. This
module decodes it and feeds Navigator.on_telemetry(), which is what turns the
pose from "re-match every scan and hope" into "dead-reckon, then apply the scan
as a gated correction" - the fix for erratic tracking while moving.

FRAME (little-endian, from the firmware)
    0    0x55            sync
    1    0xAA            sync
    2    seq      u8
    3    status   u8     bit1 = IMU_OK
    4    heading  i16    deci-degrees, IMU frame, + = left
    6    yawRate  i16    deci-deg/s
    8    odo      i32    encoder ticks, signed, forward counts up, NEVER zeroed
    12   servo    i16    deci-degrees
    14   floorC   u8
    15   tries    u8
    16   bootId   u32
    20   xor8 over bytes 2..19

WHY bootId MATTERS
    The encoder is never zeroed, so a tick delta is only meaningful within one
    boot. If the STM32 resets, odo restarts and bootId changes; taking the
    difference across that boundary would inject a huge phantom movement into
    the pose. So a bootId change re-baselines instead of dead-reckoning.
"""

from __future__ import annotations

import struct
import threading
import time

SYNC = b"\x55\xaa"
TELEM_LEN = 21
PAYLOAD_LEN = 18
_FMT = "<BBhhihBBI"          # seq status heading yaw odo servo floorC tries bootId
S_IMU_OK = 0x02


def xor8(data) -> int:
    v = 0
    for b in data:
        v ^= b
    return v


def decode(frame: bytes):
    """One 21-byte frame -> dict, or None if the checksum fails."""
    if len(frame) != TELEM_LEN or frame[0:2] != SYNC:
        return None
    payload = frame[2:2 + PAYLOAD_LEN]
    if xor8(payload) != frame[TELEM_LEN - 1]:
        return None
    seq, status, heading, yaw, odo, servo, floorc, tries, boot = \
        struct.unpack(_FMT, payload)
    return {"seq": seq, "status": status,
            "heading_deg": heading / 10.0,
            "yaw_rate_dps": yaw / 10.0,
            "odo_ticks": odo,
            "servo_deg": servo / 10.0,
            "floor": floorc, "recover_tries": tries,
            "boot_id": boot,
            "imu_ok": bool(status & S_IMU_OK)}


class TelemetryReader(threading.Thread):
    """Reads TELEM frames and pushes them into a Navigator.

    Resynchronises on the 0x55 0xAA pair, so a log line or a partial frame
    only ever costs one frame rather than desynchronising the stream.
    """

    daemon = True

    def __init__(self, nav, port="/dev/ttyACM0", baud=115200, log=print):
        super().__init__(name="TelemetryReader")
        self.nav = nav
        self.port = port
        self.baud = baud
        self.log = log
        self._stop = threading.Event()
        self.frames = 0
        self.bad = 0
        self.last = None
        self._boot = None

    def run(self):
        try:
            import serial
        except ImportError:
            self.log("[telem] pyserial not installed - no odometry")
            return
        try:
            ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except Exception as e:                      # noqa: BLE001
            self.log(f"[telem] cannot open {self.port}: {e} - no odometry")
            return
        self.log(f"[telem] reading TELEM from {self.port} @ {self.baud}")
        buf = bytearray()
        while not self._stop.is_set():
            try:
                chunk = ser.read(64)
            except Exception as e:                  # noqa: BLE001
                self.log(f"[telem] read failed: {e}")
                break
            if chunk:
                buf.extend(chunk)
            # pull out every complete frame we can find
            while True:
                i = buf.find(SYNC)
                if i < 0:
                    del buf[:max(0, len(buf) - 1)]      # keep a possible 0x55
                    break
                if len(buf) - i < TELEM_LEN:
                    del buf[:i]                         # wait for the rest
                    break
                frame = bytes(buf[i:i + TELEM_LEN])
                del buf[:i + TELEM_LEN]
                t = decode(frame)
                if t is None:
                    self.bad += 1
                    continue
                self.frames += 1
                self.last = t
                self._feed(t)
        try:
            ser.close()
        except Exception:                           # noqa: BLE001
            pass

    def _feed(self, t):
        if self._boot is not None and t["boot_id"] != self._boot:
            # STM32 rebooted: the encoder restarted, so re-baseline instead of
            # dead-reckoning across the discontinuity
            self.nav._last_ticks = None
            self.log("[telem] STM32 rebooted - odometry re-baselined")
        self._boot = t["boot_id"]
        if t["imu_ok"]:
            self.nav.on_telemetry(t["heading_deg"], t["odo_ticks"])

    def status(self):
        return {"frames": self.frames, "bad": self.bad,
                "imu_ok": bool(self.last and self.last["imu_ok"]),
                "heading_deg": self.last["heading_deg"] if self.last else None,
                "odo_ticks": self.last["odo_ticks"] if self.last else None}

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------- camera
def camera_dets(shared, max_age=0.5):
    """SharedState camera slot -> [(colour, bearing_deg)] for PillarMap.label.

    The camera only has to answer "what colour is the thing on this bearing";
    the LiDAR already supplied the position. On this car the two sensors are
    mounted on top of each other, so the bearings share an origin exactly.
    """
    from nav.pillarmap import RED, GREEN
    try:
        cam, _lidar = shared.snapshot()
    except Exception:                               # noqa: BLE001
        return []
    if cam is None or not getattr(cam, "ok", False):
        return []
    if time.time() - getattr(cam, "timestamp", 0.0) > max_age:
        return []
    out = []
    for p in getattr(cam, "pillars", []) or []:
        col = str(getattr(p, "colour", "")).upper()
        b = getattr(p, "bearing_deg", None)
        if b is None or col not in ("RED", "GREEN"):
            continue
        out.append((RED if col == "RED" else GREEN, float(b)))
    return out
