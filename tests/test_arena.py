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


def test_seats_share_a_station_across_the_corridor():
    """RULEBOOK p.14: the signs in a section come from a set of 36 cards, and a
    card never puts two signs on one line - the car would have to pass one on
    the right and the other on the left at the same instant."""
    by_station = arena.SEATS_BY_STATION
    # every station holds exactly the two lanes of one line
    assert all(len(v) == 2 for v in by_station.values())
    assert len(by_station) == 12            # 4 sides x 3 along-stations
    s = arena.SEATS[0]
    riv = arena.rivals(s)
    assert len(riv) == 1
    assert riv[0].station == s.station and riv[0].id != s.id
    assert riv[0].along == s.along and riv[0].lat != s.lat
