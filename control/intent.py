"""
intent.py - the contract between the FSM and the command mapper.

The FSM never touches servos, PWM or serial. It emits an ActionIntent: a
hardware-abstract statement of what the car should be doing right now. The
mapper makes it physically legal; link.py puts it on the wire; the STM32
turns it into motion.

Why this layer exists: when a rule changes, you edit FSM logic that speaks in
headings and lane positions. Nothing below this line moves.

Conventions - identical to worldstate.py, enforced everywhere:
    distances  mm
    angles     degrees
    headings   absolute, IMU frame, POSITIVE = LEFT
    speed      motor PWM, 0-255, magnitude only (direction is `reverse`)
"""

from dataclasses import dataclass
from enum import IntEnum


class SteerMode(IntEnum):
    """How the STM32 should decide the steering angle this tick.

    These are the wire values in the DRIVE frame's flags byte; link.py and
    ObstacleRound.cpp both depend on them.

    STOP          motor off, steering centred, PID reset, whatever else the
                  frame says.

    HEADING_HOLD  the Pi names an absolute heading; the STM32 closes its
                  heading PID on its own IMU at full rate. This is the normal
                  driving mode - the planner's lane-position yaw command has
                  already been folded into the heading.

    DIRECT        the Pi has computed a road-wheel angle; the STM32 applies it
                  through its own trim and limits with no PID. Used by the
                  3-point corner's locked-over legs, where the point is to be
                  at full lock rather than at a heading.

    ARC           the eased 90 deg arc toward `target_heading_deg`, capped at
                  `arc_lock` of each side's travel. The STM32 runs it at IMU
                  rate and reports ARC_DONE when it is within TURN_STOP_DEG;
                  the Pi enforces the odometry backstop itself.
    """
    STOP = 0
    HEADING_HOLD = 1
    DIRECT = 2
    ARC = 3


@dataclass
class ActionIntent:
    """One tick of "what the car should do". Produced by the FSM, consumed by
    the mapper. Contains no servo angles, no PWM duty, no byte layout."""

    mode: SteerMode = SteerMode.STOP
    target_heading_deg: float = 0.0   # absolute, IMU frame, + = left
    steer_deg: float = 0.0            # DIRECT only, + = left
    speed_pwm: int = 0                # magnitude, 0-255
    reverse: bool = False             # drive backwards at speed_pwm
    arc_lock: float = 0.70            # ARC only, fraction of full travel
    reason: str = ""                  # one phrase, for the log and the page

    @classmethod
    def stop(cls, reason="stop"):
        return cls(mode=SteerMode.STOP, speed_pwm=0, reason=reason)

    @classmethod
    def hold(cls, heading, speed, reason=""):
        return cls(mode=SteerMode.HEADING_HOLD, target_heading_deg=heading,
                   speed_pwm=speed, reason=reason)

    @classmethod
    def arc(cls, heading, speed, lock, reason=""):
        return cls(mode=SteerMode.ARC, target_heading_deg=heading,
                   speed_pwm=speed, arc_lock=lock, reason=reason)

    @classmethod
    def direct(cls, steer, speed, reverse=False, reason=""):
        return cls(mode=SteerMode.DIRECT, steer_deg=steer, speed_pwm=speed,
                   reverse=reverse, reason=reason)

    def moving(self):
        return self.mode != SteerMode.STOP and self.speed_pwm > 0
