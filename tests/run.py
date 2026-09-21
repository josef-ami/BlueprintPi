#!/usr/bin/env python3
"""
Run the suite with or without pytest.

    python3 tests/run.py            everything
    python3 tests/run.py planner    just the files whose name matches

On a machine with pytest, `pytest -q` does the same thing and reports better.
This exists so the suite is runnable anywhere, including a fresh Pi before
anyone has installed the dev tools.
"""

import importlib.util
import inspect
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import conftest                                            # noqa: E402,F401
import pytest                                              # noqa: E402


def load(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_fixture(name):
    fx = getattr(pytest, "fixtures", lambda: {})().get(name)
    if fx is None:
        fx = getattr(conftest, name, None)
        if fx is None:
            raise KeyError(f"no fixture named {name!r}")
    return fx()


def run_one(fn):
    kwargs = {}
    for pname, param in inspect.signature(fn).parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue                      # it has a default; leave it alone
        kwargs[pname] = build_fixture(pname)
    fn(**kwargs)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(f for f in os.listdir(HERE)
                   if f.startswith("test_") and f.endswith(".py")
                   and pattern in f)
    passed = failed = skipped = 0
    failures = []

    for f in files:
        mod = load(os.path.join(HERE, f))
        names = [n for n in dir(mod) if n.startswith("test_")]
        for n in sorted(names):
            fn = getattr(mod, n)
            if not callable(fn):
                continue
            cond, reason = getattr(fn, "_skipif", (False, ""))
            if cond:
                skipped += 1
                continue
            params = getattr(fn, "_parametrize", None)
            cases = [None]
            if params:
                pnames, values = params
                pnames = [x.strip() for x in pnames.split(",")]
                cases = values
            for case in cases:
                try:
                    if params:
                        vals = case if isinstance(case, (list, tuple)) else (case,)
                        extra = dict(zip(pnames, vals))
                        kwargs = {}
                        for pname in inspect.signature(fn).parameters:
                            kwargs[pname] = (extra[pname] if pname in extra
                                             else build_fixture(pname))
                        fn(**kwargs)
                    else:
                        run_one(fn)
                    passed += 1
                except Exception as e:
                    if type(e).__name__ == "Skipped":
                        skipped += 1
                        continue
                    failed += 1
                    failures.append((f, n, traceback.format_exc()))

    for f, n, tb in failures:
        print(f"\n=== FAIL {f}::{n} ===")
        print(tb.rstrip())
    tail = f"{passed} passed, {failed} failed"
    if skipped:
        tail += f", {skipped} skipped"
    print(("\n" if failures else "") + tail)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
