"""
control - the FSM, the lane planner, command mapping and the STM32 link.

    intent.py    ActionIntent: what the car should do, hardware-abstract
    planner.py   lane position, pillar tracks, the pass planner, levelling
    fsm.py       the obstacle-round state machine - the whole run
    mapper.py    ActionIntent -> physically legal steering and speed
    link.py      binary DRIVE/TELEM framing to and from the STM32

Data flows one way:

    perception + telemetry -> Ctx -> FSM -> ActionIntent -> mapper -> link
                                 ^                                      |
                                 +--------------- Telemetry ------------+

The FSM and the planner used to live in ObstacleRound.cpp. They run here now;
the STM32 keeps only the loops a 50 Hz link cannot close - the heading PID at
IMU rate, the arc's inner loop, and the wall-panic reflex. See
docs/PI_STM32_PROTOCOL.md for where the line falls and why.
"""

from worldstate import clamp, wrap180

from .intent import ActionIntent, SteerMode
from .planner import LanePlanner, PlanInput, Track, MAX_TRACKS
from .fsm import FSM, Ctx, State
from .mapper import (CommandMapper, steer_for_radius, steer_for_lateral_shift,
                     radius_for_steer, WHEELBASE_MM, STEER_LOCK_DEG,
                     KINEMATIC_RADIUS_MM, MEASURED_RADIUS_MM)
from .link import Link, Telemetry, pack_drive, unpack_telem

__all__ = [
    "wrap180", "clamp",
    "ActionIntent", "SteerMode",
    "LanePlanner", "PlanInput", "Track", "MAX_TRACKS",
    "FSM", "Ctx", "State",
    "CommandMapper", "steer_for_radius", "steer_for_lateral_shift",
    "radius_for_steer", "WHEELBASE_MM", "STEER_LOCK_DEG",
    "KINEMATIC_RADIUS_MM", "MEASURED_RADIUS_MM",
    "Link", "Telemetry", "pack_drive", "unpack_telem",
]
