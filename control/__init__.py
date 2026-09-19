"""
control — interpretation, FSM, command mapping and the STM32 link.

    intent.py   ActionIntent: what the car should do, hardware-abstract
    fsm.py      the obstacle-round state machine (skeleton; see FSM_GUIDE.md)
    mapper.py   ActionIntent -> physically legal steering/speed
    link.py     binary framing to/from the STM32 over USB CDC

Data flows one way:
    worldstate + telemetry -> Ctx -> FSM -> ActionIntent -> mapper -> link -> STM32
"""

from .intent import ActionIntent, SteerMode
from .fsm import FSM, Ctx, RunState, State, wrap180
from .mapper import (CommandMapper, steer_for_radius, steer_for_lateral_shift,
                     radius_for_steer, WHEELBASE_MM, STEER_LOCK_DEG,
                     MIN_TURN_RADIUS_MM, TOP_SPEED_MMPS)
from .link import Link, Telemetry, FLOOR_NONE, FLOOR_ORANGE, FLOOR_BLUE

__all__ = [
    "ActionIntent", "SteerMode",
    "FSM", "Ctx", "RunState", "State", "wrap180",
    "CommandMapper", "steer_for_radius", "steer_for_lateral_shift",
    "radius_for_steer", "WHEELBASE_MM", "STEER_LOCK_DEG",
    "MIN_TURN_RADIUS_MM", "TOP_SPEED_MMPS",
    "Link", "Telemetry", "FLOOR_NONE", "FLOOR_ORANGE", "FLOOR_BLUE",
]
