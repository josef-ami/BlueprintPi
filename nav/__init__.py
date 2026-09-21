"""
nav - map-based localisation, sign mapping and path planning for the
Blueprint obstacle round.

    from nav import NavService
    svc = NavService(lidar)          # a sensors.lidar.LidarThread
    svc.start()
    ...
    fields = svc.wire_fields()       # (navOk, crossMm, hdgErrDd, curvUm, ver)

Everything in here works in the MAT frame, which is fixed by the rulebook
and known before the round starts. Nothing is learned; the only unknowns are
where the car is and where the traffic signs are.

The heavy solve runs on its own thread. It must never sit in the 50 Hz
sensor-feed loop - a replan takes ~200 ms here and would stall the link.
"""

from .service import NavService          # noqa: F401
from .geom import OUTER, INNER, CORRIDOR, MID, PILLAR   # noqa: F401
