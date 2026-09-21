"""
mapper.py - ActionIntent -> a frame the STM32 will accept.

This is the only place on the Pi that knows the car's physical limits. The
FSM asks for things in the language of the world ("hold 12 degrees left of
the lane at PWM 60"); the mapper makes those legal and hands them to link.py.

What it does NOT know, and must not: servo trim, servo microseconds, PWM
duty, which way the servo horn faces, the heading PID gains. All of that is
owned by the STM32 (its parameter table), so a mechanical change never
touches Pi code. The mapper's clamps are a second line of defence on top of
the firmware's own - a bug here should produce a slow wrong turn, not a
servo against its stop.

Geometry constants come from SPECSHEET.md section 3 (CAD verified
2026-09-15), except the turn radius - see the note on that below.
"""

import math

from worldstate import clamp
from .intent import ActionIntent, SteerMode

# ------------------------------------------------ vehicle geometry --------

WHEELBASE_MM = 136.14        # front axle to rear axle
TRACK_MM = 105.0             # wheel centre to centre
STEER_LOCK_DEG = 35.0        # measured at the knuckles. FINAL.

# Two turn radii, and they disagree on purpose.
#
#   KINEMATIC_RADIUS_MM   what the bicycle model predicts from the wheelbase
#                         and the lock: 194 mm. It assumes the tyres do not
#                         slip, which at full lock on a mat they do.
#   MEASURED_RADIUS_MM    what the car actually does: 27 cm turning left,
#                         25 cm turning right, measured on the mat.
#
# The planner's reach estimate uses the MEASURED figure (params.py's
# TURN_RADIUS_MM, default 270, the worse of the two sides) because a reach
# estimate that believes the car turns tighter than it does will commit to a
# pass it cannot make. The kinematic value is kept because it is the right
# one for steer_for_radius() below, which asks the inverse question.
KINEMATIC_RADIUS_MM = WHEELBASE_MM / math.tan(math.radians(STEER_LOCK_DEG))
MEASURED_RADIUS_MM = 270.0

# The motor stalls rather than creeps below this; the firmware's PWM band.
SPEED_MIN_MOVING = 30
SPEED_MAX = 255

ARC_LOCK_MIN, ARC_LOCK_MAX = 0.20, 1.00


# ------------------------------------------------ geometry helpers --------

def steer_for_radius(radius_mm):
    """Road-wheel angle that produces this turn radius, + = left."""
    if radius_mm == 0:
        return 0.0
    sign = 1.0 if radius_mm > 0 else -1.0
    r = abs(radius_mm)
    if r < KINEMATIC_RADIUS_MM:
        return sign * STEER_LOCK_DEG
    return sign * math.degrees(math.atan(WHEELBASE_MM / r))


def radius_for_steer(steer_deg):
    """The inverse. inf when the wheels are straight."""
    if abs(steer_deg) < 1e-6:
        return float("inf")
    sign = 1.0 if steer_deg > 0 else -1.0
    return sign * WHEELBASE_MM / math.tan(math.radians(abs(steer_deg)))


def steer_for_lateral_shift(shift_mm, over_mm):
    """Steering that moves the car `shift_mm` sideways over `over_mm` of
    travel, in the small-angle limit. Used for sanity checks, not for
    driving - the planner aims at a point instead, which behaves better when
    the shift is large."""
    if over_mm <= 0:
        return 0.0
    return math.degrees(math.atan(2.0 * WHEELBASE_MM * shift_mm / (over_mm ** 2)))


# ------------------------------------------------------- the mapper -------

class CommandMapper:
    """Stateless apart from what it reports. Every intent that leaves here is
    something the car can physically do."""

    def __init__(self):
        self.clamped = ""          # what the last map() had to change, if any

    def map(self, intent: ActionIntent, speed_max=SPEED_MAX):
        """-> a new ActionIntent, clamped. The original is left alone."""
        notes = []

        if intent.mode == SteerMode.STOP:
            self.clamped = ""
            return ActionIntent.stop(intent.reason or "stop")

        speed = int(clamp(intent.speed_pwm, 0, speed_max))
        if 0 < speed < SPEED_MIN_MOVING:
            # Asking for a creep the motor cannot deliver just winds up the
            # PID. Ask for nothing, or ask for enough to move.
            notes.append(f"speed {speed} below stall, raised to {SPEED_MIN_MOVING}")
            speed = SPEED_MIN_MOVING

        steer = clamp(intent.steer_deg, -STEER_LOCK_DEG, STEER_LOCK_DEG)
        if abs(steer - intent.steer_deg) > 1e-6:
            notes.append(f"steer {intent.steer_deg:+.1f} clamped to {steer:+.1f}")

        lock = clamp(intent.arc_lock, ARC_LOCK_MIN, ARC_LOCK_MAX)
        if abs(lock - intent.arc_lock) > 1e-6:
            notes.append(f"arc lock {intent.arc_lock:.2f} clamped to {lock:.2f}")

        heading = ((intent.target_heading_deg + 180.0) % 360.0) - 180.0

        self.clamped = "; ".join(notes)
        return ActionIntent(mode=intent.mode, target_heading_deg=heading,
                            steer_deg=steer, speed_pwm=speed,
                            reverse=intent.reverse, arc_lock=lock,
                            reason=intent.reason)
