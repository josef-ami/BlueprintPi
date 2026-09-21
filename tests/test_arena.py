"""Arena semantic map: seat grid + parking geometry."""

import math

import pytest

from nav import arena


def test_seat_count_and_unique_ids():
    assert len(arena.SEATS) == 24
    ids = [s.id for s in arena.SEATS]
    assert len(set(ids)) == 24


def test_seats_lie_inside_the_corridor():
    # every seat is between the inner and outer walls, never in a wall
    for s in arena.SEATS:
        r = max(abs(s.x), abs(s.y))          # chebyshev radius from centre
        assert arena.INNER / 2 < r < arena.OUTER / 2


def test_nearest_seat_snaps_and_rejects():
    s0 = arena.SEATS[0]
    seat, d = arena.nearest_seat(s0.x + 5, s0.y - 5)
    assert seat is not None and seat.id == s0.id and d < 10
    seat, d = arena.nearest_seat(0.0, 0.0)   # centre of the inner block
    assert seat is None and d > arena.SNAP_TOL_MM


def test_parking_bay_length_is_1_5_car():
    bay = arena.parking_bay("S", 0.0, 180.0)
    assert bay.length == pytest.approx(270.0)
    # rect sits against the south outer wall, 200 mm deep
    x0, y0, x1, y1 = bay.rect
    assert y0 == pytest.approx(-arena.OUTER_HALF)
    assert (y1 - y0) == pytest.approx(200.0)


def test_parking_blocks_are_four_segments_each():
    bay = arena.parking_bay("E", 100.0, 175.0)
    assert bay.left_block.shape == (4, 4)
    assert bay.right_block.shape == (4, 4)
