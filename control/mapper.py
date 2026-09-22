"""
mapper.py — ActionIntent -> DriveCommand.

This is the only place on the Pi that knows the car's physical limits. The FSM
asks for things in the language of the world ("hold 0 degrees at 400 mm/s");
the mapper makes those physically legal and hands them to link.py.

What it does NOT know: servo trim, servo microseconds, PWM duty, the direction
the servo horn faces. All of that is owned by the STM32 (hardware_config.h),
so a mechanical change never touches Pi code.

What it enforces:
  - steering clamped to the mechanical lock
  - speed clamped to the drivetrain's real top speed
  - steering slew limit, so a state change can't ask the servo to slam across
    its travel in one tick (wheel scrub, current spike, yaw jerk)
  - a stop is always honoured immediately, never slewed

Geometry constants come from SPECSHEET.md section 3 (CAD verified 2026-09-15).
"""

import math

from .intent import ActionIntent, SteerMode

# ------------------------------------------------ vehicle geometry --------
# SPECSHEET.md section 3, "Geometry — AS BUILT (2026-07-26)", CAD verified.

WHEELBASE_MM = 136.14        # front axle to rear axle
TRACK_MM = 105.0             # wheel centre to centre
STEER_LOCK_DEG = 35.0        # measured at the knuckles. FINAL.
MIN_TURN_RADIUS_MM = WHEELBASE_MM / math.tan(math.radians(STEER_LOCK_DEG))
#                  = 194.4 mm

TOP_SPEED_MMPS = 700.0       # 266 rpm rear wheel x pi x 50 mm, section 4

# Slew limit on the steering command. The servo is a JX PS-1171MG digital, so
# it is not stuck at the MG996R's 50 Hz ceiling, but the tyres still scrub if
# you step the angle. At a 30 Hz control tick, 6 deg/tick = 180 deg/s.
MAX_STEER_SLEW_DEG_PER_TICK = 6.0

# Below this the motor stalls rather than creeps; the PID will wind up trying.
# Ask for 0 or ask for at least this.
MIN_MOVING_SPEED_MMPS = 60.0


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ------------------------------------------------ geometry helpers --------
# These exist so FSM code can reason in real units instead of magic angles.

def radius_for_steer(angle_deg: float) -> float:
    """
    Turn radius (mm) the car will follow at this road-wheel angle.
    Returns inf for straight ahead. Sign is dropped — magnitude only.
    """
    a = abs(angle_deg)
    if a < 0.01:
        return float("inf")
    return WHEELBASE_MM / math.tan(math.radians(a))


def steer_for_radius(radius_mm: float) -> float:
    """
    Road-wheel angle (magnitude, deg) needed to follow this radius.
    Clamped to the mechanical lock: ask for a tighter circle than the car can
    physically turn and you get full lock, not a fantasy.
    """
    if radius_mm == 0 or math.isinf(radius_mm):
        return 0.0
    a = math.degrees(math.atan(WHEELBASE_MM / abs(radius_mm)))
    return min(a, STEER_LOCK_DEG)


def steer_for_lateral_shift(shift_mm: float, over_distance_mm: float) -> float:
    """
    Approximate road-wheel angle to move sideways by shift_mm while travelling
    over_distance_mm forward, as one circular arc.

    This is the workhorse for pillar avoidance: "I need to be 250 mm further
    left by the time I reach the pillar 800 mm ahead" -> an angle.

    Small-angle arc approximation: shift ~= d^2 / (2R)  ->  R = d^2 / (2*shift)
    Accurate enough while shift << distance, which is the regime you are in.
    Sign follows shift (+ = left).
    """
    if abs(shift_mm) < 0.1 or over_distance_mm <= 0:
        return 0.0
    radius = (over_distance_mm ** 2) / (2.0 * abs(shift_mm))
    return math.copysign(steer_for_radius(radius), shift_mm)


# --------------------------------------------------------- the mapper ----

class CommandMapper:
    """
    Stateful because of the slew limit — it remembers the last angle it sent.
    One instance, created once, called once per control tick.
    """

    def __init__(self,
                 steer_lock_deg: float = STEER_LOCK_DEG,
                 top_speed_mmps: float = TOP_SPEED_MMPS,
                 max_slew: float = MAX_STEER_SLEW_DEG_PER_TICK):
        self.steer_lock_deg = steer_lock_deg
        self.top_speed_mmps = top_speed_mmps
        self.max_slew = max_slew
        self._last_steer_deg = 0.0
        # set by map(); read by logging/telemetry so you can see what was sent
        self.last_speed_mmps = 0.0
        self.clamped_steer = False
        self.clamped_speed = False

    def reset(self):
        """Call when entering a state that should start from centred steering."""
        self._last_steer_deg = 0.0

    def map(self, intent: ActionIntent):
        """
        ActionIntent -> (steer_deg, speed_mmps), both physically legal.

        Returns the pair link.send_intent() wants. The intent itself carries
        the mode and the target heading, which pass through untouched.
        """
        # A stop is absolute: no slew, no minimum-speed logic, immediate.
        if intent.steer_mode is SteerMode.STOP:
            self._last_steer_deg = 0.0
            self.last_speed_mmps = 0.0
            self.clamped_steer = self.clamped_speed = False
            return 0.0, 0.0

        # ---- steering ----
        if intent.steer_mode is SteerMode.HEADING_HOLD:
            # The STM32 computes the angle from its own IMU; we send 0 as a
            # defined value rather than stale garbage, and we let the slew
            # state decay towards centre so a later DIRECT starts sanely.
            steer = 0.0
            self.clamped_steer = False
        else:
            want = intent.steer_angle_deg
            steer = clamp(want, -self.steer_lock_deg, self.steer_lock_deg)
            self.clamped_steer = (abs(want - steer) > 0.01)

        # slew limit, applied in both modes so _last_steer_deg stays truthful
        delta = steer - self._last_steer_deg
        if delta > self.max_slew:
            steer = self._last_steer_deg + self.max_slew
        elif delta < -self.max_slew:
            steer = self._last_steer_deg - self.max_slew
        self._last_steer_deg = steer

        # ---- speed ----
        want_v = intent.speed_mmps
        speed = clamp(want_v, -self.top_speed_mmps, self.top_speed_mmps)
        self.clamped_speed = (abs(want_v - speed) > 0.01)
        # don't ask for a speed the drivetrain can only stall at
        if 0.0 < abs(speed) < MIN_MOVING_SPEED_MMPS:
            speed = math.copysign(MIN_MOVING_SPEED_MMPS, speed)
        self.last_speed_mmps = speed

        return steer, speed
