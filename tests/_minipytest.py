"""
A very small stand-in for the parts of pytest this suite uses.

The Pi has pytest; some machines do not, and a test suite you cannot run is
worth nothing. conftest.py imports the real pytest when it is installed and
falls back to this otherwise, so `python3 tests/run.py` always works and
`pytest` still does the right thing where it exists.

Supported: approx, raises, fixture (function-scoped, by name), mark.skipif,
mark.parametrize. Nothing else - if a test needs more, install pytest.
"""

import math


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _close(self, a, b):
        if a is None or b is None:
            return a is b
        rel = self.rel if self.rel is not None else 1e-6
        tol = self.abs if self.abs is not None else 0.0
        return math.isclose(a, b, rel_tol=rel, abs_tol=max(tol, 1e-12))

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            if not isinstance(other, (list, tuple)) \
                    or len(other) != len(self.expected):
                return False
            return all(self._close(float(x), float(y))
                       for x, y in zip(other, self.expected))
        try:
            return self._close(float(other), float(self.expected))
        except (TypeError, ValueError):
            return other == self.expected

    def __req__(self, other):
        return self.__eq__(other)

    def __repr__(self):
        return f"approx({self.expected!r})"


def approx(expected, rel=None, abs=None):
    return _Approx(expected, rel=rel, abs=abs)


class _Raises:
    def __init__(self, exc):
        self.exc = exc
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        self.value = v
        return issubclass(t, self.exc)


def raises(exc):
    return _Raises(exc)


_FIXTURES = {}


def fixture(func=None, **kw):
    def wrap(f):
        _FIXTURES[f.__name__] = f
        f._is_fixture = True
        return f
    return wrap(func) if func is not None else wrap


class _Mark:
    @staticmethod
    def skipif(cond, reason=""):
        def deco(f):
            f._skipif = (bool(cond), reason)
            return f
        return deco

    @staticmethod
    def parametrize(names, values):
        def deco(f):
            f._parametrize = (names, values)
            return f
        return deco


mark = _Mark()


class Skipped(Exception):
    pass


def skip(reason=""):
    raise Skipped(reason)


def fail(msg=""):
    raise AssertionError(msg)


def fixtures():
    return _FIXTURES
