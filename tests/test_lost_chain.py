"""
The lost-pillar chain, run against the REAL firmware compiled for the host.

firmware/sim/build_lost.sh compiles firmware/ObstacleRound.cpp unmodified
against the stub headers and exposes updateLost() / lostYaw(); these tests
drive them through ctypes. So what is checked is the code that gets flashed,
not a Python restatement of it.

The chain came from WRO2025_FE_ANTi (src/obstacle.py, lost_color/pass_color).
Its job here is narrower than there: it only steers when no wall could be
fitted, because everywhere else this firmware has a lane frame and odometry
and does not need to guess. These tests cover the chain itself; where it is
allowed to act is a property of updatePlanner() and is stated in the comment
above the chain.

Skipped when there is no g++ (e.g. a Pi without build-essential).
"""

import ctypes
import os
import shutil
import subprocess
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(os.path.dirname(HERE), "firmware", "sim")

needs_gcc = pytest.mark.skipif(shutil.which("g++") is None,
                               reason="no g++ to build the firmware with")

_BUILT = None
_N = 0


def _build():
    global _BUILT
    if _BUILT is None:
        out = os.path.join(tempfile.mkdtemp(prefix="lostsim-"), "liblost.so")
        r = subprocess.run(["sh", os.path.join(SIM, "build_lost.sh"), out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise AssertionError("firmware did not compile:\n" + r.stderr)
        assert "warning" not in r.stderr, \
            "firmware compiles with warnings:\n" + r.stderr
        _BUILT = out
    return _BUILT


class Chain:
    """A freshly loaded copy of the firmware, so no test inherits another's
    globals - dlopen refcounts by path, so each one gets its own file."""

    def __init__(self, search=2, commit=6, abandon=40, sdeg=15.0, cdeg=35.0):
        global _N
        src = _build()
        _N += 1
        dst = os.path.join(os.path.dirname(src), f"liblost-{_N}.so")
        shutil.copyfile(src, dst)
        self.lib = ctypes.CDLL(dst)
        self.lib.lost_yaw.restype = ctypes.c_float
        self.lib.lost_tune.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]
        self.RED = self.lib.k_vis_red()
        self.GREEN = self.lib.k_vis_green()
        self.NONE = self.lib.k_vis_none()
        self.P_NONE = self.lib.k_lost_none()
        self.P_SEARCH = self.lib.k_lost_search()
        self.P_COMMIT = self.lib.k_lost_commit()
        self.search, self.commit, self.abandon = search, commit, abandon
        self.sdeg, self.cdeg = sdeg, cdeg
        self.lib.lost_tune(search, commit, abandon, sdeg, cdeg)
        self.lib.lost_reset()

    def see(self, colour, frames=1):
        for _ in range(frames):
            self.lib.lost_feed(colour)
        return self

    def blind(self, frames=1):
        return self.see(self.NONE, frames)

    @property
    def phase(self):
        return self.lib.lost_phase()

    @property
    def yaw(self):
        return round(self.lib.lost_yaw(), 4)


# ------------------------------------------------------------- the phases

@needs_gcc
def test_nothing_happens_before_a_pillar_has_ever_been_seen():
    c = Chain()
    c.blind(50)
    assert c.phase == c.P_NONE
    assert c.yaw == 0.0


@needs_gcc
def test_a_pillar_in_sight_keeps_the_chain_idle():
    c = Chain()
    c.see(c.RED, 10)
    assert c.phase == c.P_NONE
    assert c.yaw == 0.0


@needs_gcc
def test_search_starts_after_LOST_SEARCH_FRAMES_blind_frames():
    c = Chain()
    c.see(c.RED).blind(c.search)
    assert c.phase == c.P_NONE          # not yet
    c.blind(1)
    assert c.phase == c.P_SEARCH


@needs_gcc
def test_commit_starts_after_LOST_COMMIT_FRAMES():
    c = Chain()
    c.see(c.RED).blind(c.commit)
    assert c.phase == c.P_SEARCH
    c.blind(1)
    assert c.phase == c.P_COMMIT


@needs_gcc
def test_the_chain_is_forgotten_after_LOST_ABANDON_FRAMES():
    c = Chain()
    c.see(c.RED).blind(c.abandon + 1)
    assert c.phase == c.P_NONE
    assert c.yaw == 0.0


@needs_gcc
def test_seeing_a_pillar_again_clears_the_chain_at_once():
    c = Chain()
    c.see(c.RED).blind(c.commit + 1)
    assert c.phase == c.P_COMMIT
    c.see(c.GREEN)
    assert c.phase == c.P_NONE
    assert c.yaw == 0.0


# -------------------------------------------------------------- the sides

@needs_gcc
def test_red_searches_left_then_commits_right():
    """Red is passed on its right, so the car ends up right of it. The search
    that precedes that turns the other way, to bring it back into frame."""
    c = Chain()
    c.see(c.RED).blind(c.search + 1)
    assert c.yaw == c.sdeg
    c.blind(c.commit - c.search)
    assert c.yaw == -c.cdeg


@needs_gcc
def test_green_is_the_mirror_of_red():
    c = Chain()
    c.see(c.GREEN).blind(c.search + 1)
    assert c.yaw == -c.sdeg
    c.blind(c.commit - c.search)
    assert c.yaw == c.cdeg


@needs_gcc
def test_the_side_follows_the_last_colour_seen_not_the_first():
    c = Chain()
    c.see(c.RED, 3).see(c.GREEN, 3).blind(c.commit + 1)
    assert c.yaw == c.cdeg              # green's commit side


# ------------------------------------------------------------- the tuning

@needs_gcc
def test_zero_search_frames_goes_straight_into_searching():
    c = Chain(search=0)
    c.see(c.RED).blind(1)
    assert c.phase == c.P_SEARCH


@needs_gcc
def test_the_angles_are_tunable_at_runtime():
    c = Chain(sdeg=8.0, cdeg=50.0)
    c.see(c.RED).blind(c.search + 1)
    assert c.yaw == 8.0
    c.blind(c.commit - c.search)
    assert c.yaw == -50.0
