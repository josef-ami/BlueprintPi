"""
intent.py — the contract between the FSM and the command mapper.

The FSM never touches servos, PWM or serial. It emits an ActionIntent: a
hardware-abstract statement of what the car should be doing right now. The
mapper turns that into a DriveCommand; the STM32 turns that into motion.

Why this layer exists: when a rule changes (or the surprise rule lands), you
edit FSM logic that speaks in headings and mm/s. Nothing below this line moves.

Conventions — identical to worldstate.py, enforced everywhere:
    distances  mm
    angles     degrees
    bearings   0 = robot forward, POSITIVE = LEFT
    speeds     mm/s, POSITIVE = FORWARD
"""

from dataclasses import dataclass
from enum import Enum


class SteerMode(Enum):
    """
    How the STM32 should decide the steering angle this tick.

    DIRECT       the Pi has computed a road-wheel angle; STM32 just applies it
                 (through its own trim + limits). Use for arcs, parking
                 shuffles, avoidance swerves — anything geometric.

    HEADING_HOLD the Pi names a heading; the STM32 closes a P-loop on its own
                 IMU at full rate to hold it. Use for straights and any time
                 you want the heading loop faster than the 50 Hz link.

    STOP         motor off, steering centred, regardless of the other fields.
    """
    DIRECT = 0
    HEADING_HOLD = 1
    STOP = 2


@dataclass
class ActionIntent:
    """
    One tick of "what the car should do". Produced by the FSM, consumed by the
    mapper. Contains no servo angles, no PWM, no byte layout.

    steer_mode        which of the two steering fields below is authoritative
    steer_angle_deg   road-wheel angle, + = left. Used when DIRECT.
                      Mapper clamps to the mechanical lock; you may ask for
                      more and it will be limited rather than rejected.
    target_heading_deg  absolute heading to hold, + = left of the heading the
                      IMU was zeroed at. Used when HEADING_HOLD.
    speed_mmps        target ground speed, + = forward, - = reverse.
                      Closed-loop on the STM32 encoder when closed_loop=True.
    closed_loop       False during bring-up: speed_mmps is passed through as a
                      raw duty fraction of top speed instead of PID'd. Lets you
                      run the whole stack before the PID is tuned.
    reason            free-text, telemetry/logging only. Never affects motion.
                      Put the state name and why you chose this here — it is
                      what you will read when debugging a run.
    """
    steer_mode: SteerMode = SteerMode.STOP
    steer_angle_deg: float = 0.0
    target_heading_deg: float = 0.0
    speed_mmps: float = 0.0
    closed_loop: bool = True
    reason: str = ""

    # ---- constructors for the three things you actually want to say --------

    @staticmethod
    def stop(reason: str = "") -> "ActionIntent":
        return ActionIntent(steer_mode=SteerMode.STOP, speed_mmps=0.0,
                            reason=reason)

    @staticmethod
    def hold_heading(heading_deg: float, speed_mmps: float,
                     closed_loop: bool = True, reason: str = "") -> "ActionIntent":
        """Drive at speed while the STM32 holds this absolute heading."""
        return ActionIntent(steer_mode=SteerMode.HEADING_HOLD,
                            target_heading_deg=heading_deg,
                            speed_mmps=speed_mmps,
                            closed_loop=closed_loop, reason=reason)

    @staticmethod
    def steer(angle_deg: float, speed_mmps: float,
              closed_loop: bool = True, reason: str = "") -> "ActionIntent":
        """Drive at speed with this road-wheel angle. + = left."""
        return ActionIntent(steer_mode=SteerMode.DIRECT,
                            steer_angle_deg=angle_deg,
                            speed_mmps=speed_mmps,
                            closed_loop=closed_loop, reason=reason)
