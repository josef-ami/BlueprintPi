"""
Shared test fixtures.

The Pi-only libraries (picamera2, libcamera, rplidarc1) are not installed on a
dev box and are not needed to test any of the logic, so they are stubbed here
before anything imports them. Nothing under test calls into them - that is the
point of keeping the geometry and the state machine free of hardware.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for name in ("picamera2", "libcamera", "rplidarc1"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules["libcamera"].Transform = object
sys.modules["rplidarc1"].RPLidar = object
sys.modules["picamera2"].Picamera2 = object

try:
    import pytest                                          # noqa: E402
except ImportError:
    # No pytest on this machine. tests/_minipytest.py covers the handful of
    # features this suite uses, so `python3 tests/run.py` still works.
    import _minipytest as pytest
    sys.modules["pytest"] = pytest

import params as prm                                       # noqa: E402


@pytest.fixture
def p():
    """A PiParams at its defaults - which are the firmware's defaults for
    every parameter that came from it."""
    return prm.PiParams()
