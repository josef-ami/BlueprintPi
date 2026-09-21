"""
lab_fit.py - click the mat and the pillars, get Lab thresholds.

The fitting half of what used to be calibrate_vision.py, with the web page
removed: the dashboard's Calibrate tab drives it now, so there is no second
process fighting for the camera.

WHY LAB AND NOT HSV
  HSV separates the pillars on hue but gates on SATURATION. A matte pillar
  under dim indoor light falls under the S threshold and disappears - and
  green disappears first, because red has two hue bands and survives longer.
  That is the usual cause of "green stops being detected".

  In OpenCV's 8-bit Lab, a and b are centred on 128: a > 128 is red, a < 128
  is green, and the white mat sits near (128, 128) whatever the light does to
  L. The pillars separate on ONE channel and brightness never removes the
  colour.

HOW THE FIT WORKS
  ONE channel classifies, the others only exclude.

  'a' is what separates the three classes: red sits well above 128, green well
  below, the mat on it. So each pillar's 'a' boundary is placed BETWEEN its own
  samples and the mat's - halfway from one distribution's edge to the other,
  using robust percentiles rather than min/max so one bad click cannot move
  it. If the two distributions actually touch, the fit REFUSES rather than
  returning thresholds that cannot work; that means the camera genuinely
  cannot tell them apart and the answer is exposure or white balance, not
  numbers.

  L and b are fitted loosely on purpose. A pillar is seen near and far, lit
  and shaded, so its lightness moves far more than its colour, and a tight L
  range fitted under one lighting condition is exactly what makes a
  calibration stop working when the room changes. They only drop near-black
  and wildly wrong hues.
"""

import math
import threading

import numpy as np

CLASSES = ("mat", "red", "green")
PATCH = 5                 # a click samples a PATCH x PATCH box, detection px
MAX_SAMPLES = 400

# Everything the fit writes. The Calibrate tab shows these, in this order.
FITTED_NAMES = ["USE_LAB", "RED_L_LO", "RED_L_HI", "RED_A_LO", "RED_A_HI",
                "RED_B_LO", "RED_B_HI", "GREEN_L_LO", "GREEN_L_HI",
                "GREEN_A_LO", "GREEN_A_HI", "GREEN_B_LO", "GREEN_B_HI",
                "FLOOR_L_MIN", "FLOOR_AB_TOL", "LAB_CHROMA_MIN", "AREA_K"]


class LabCalibration:
    """Click samples and the fit over them. Thread-safe; the web handlers and
    the vision thread both touch it."""

    def __init__(self):
        self._lock = threading.Lock()
        self.samples = {c: [] for c in CLASSES}
        self.fit_params = {"margin": 10.0, "sep": 6.0, "pct": 5.0}
        self.area_k = {"dist_mm": 600.0, "result": None}

    # ---------------------------------------------------------- sampling

    def sample(self, cls, x, y, hsv, lab):
        """Sample one detection-resolution frame at (x, y). Returns the
        sample dict, or None if the class is unknown."""
        if cls not in CLASSES:
            return None
        h, w = lab.shape[:2]
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        r = PATCH // 2
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        # median, not mean: one stray pixel on a pillar edge should not move it
        L, a, b = np.median(lab[y0:y1, x0:x1].reshape(-1, 3), axis=0)
        H, S, V = np.median(hsv[y0:y1, x0:x1].reshape(-1, 3), axis=0)
        s = {"x": x, "y": y, "L": float(L), "a": float(a), "b": float(b),
             "H": float(H), "S": float(S), "V": float(V)}
        with self._lock:
            if len(self.samples[cls]) < MAX_SAMPLES:
                self.samples[cls].append(s)
        return s

    def undo(self, cls):
        with self._lock:
            if cls in self.samples and self.samples[cls]:
                self.samples[cls].pop()

    def clear(self, cls):
        with self._lock:
            if cls in self.samples:
                self.samples[cls].clear()

    def markers(self):
        """[(cls, x, y)] for the overlay."""
        with self._lock:
            return [(c, s["x"], s["y"])
                    for c in CLASSES for s in self.samples[c]]

    def stats(self):
        out = []
        for c in CLASSES:
            with self._lock:
                rows = list(self.samples[c])
            if not rows:
                out.append({"name": c, "n": 0})
                continue
            arr = np.array([[r["L"], r["a"], r["b"]] for r in rows])
            out.append({"name": c, "n": len(rows),
                        "mean": arr.mean(axis=0).round(1).tolist(),
                        "min": arr.min(axis=0).round(0).tolist(),
                        "max": arr.max(axis=0).round(0).tolist()})
        return out

    def set_fit_params(self, **kw):
        for k in ("margin", "sep", "pct"):
            if k in kw and kw[k] is not None:
                try:
                    self.fit_params[k] = max(0.0, min(80.0, float(kw[k])))
                except (TypeError, ValueError):
                    pass
        return dict(self.fit_params)

    def set_distance(self, mm):
        try:
            self.area_k["dist_mm"] = max(50.0, min(4000.0, float(mm)))
        except (TypeError, ValueError):
            pass
        return self.area_k["dist_mm"]

    def area_k_from_area(self, area):
        """A pillar of known area at a known distance gives AREA_K, the
        fallback the pillar locator uses when no LiDAR return agrees with the
        camera ray: distance ~ AREA_K / sqrt(area)."""
        if not area or area <= 0:
            return None
        k = self.area_k["dist_mm"] * math.sqrt(area)
        self.area_k["result"] = k
        return k

    # -------------------------------------------------------------- fit

    def fit(self):
        """(ok, message, values). `values` is a dict ready for PiParams."""
        with self._lock:
            have = {c: len(self.samples[c]) for c in CLASSES}
        missing = [c for c in CLASSES if have[c] < 3]
        if missing:
            return False, f"need at least 3 samples of: {', '.join(missing)}", {}

        def arr(c):
            with self._lock:
                return np.array([[r["L"], r["a"], r["b"]] for r in self.samples[c]])

        m = self.fit_params["margin"]
        sep = self.fit_params["sep"]
        pct = self.fit_params["pct"]
        mat, red, grn = arr("mat"), arr("red"), arr("green")

        def lo_p(a, i):
            return float(np.percentile(a[:, i], pct))

        def hi_p(a, i):
            return float(np.percentile(a[:, i], 100 - pct))

        out = {}
        for name, a in (("RED", red), ("GREEN", grn)):
            # The 'a' boundary goes BETWEEN the two distributions - halfway
            # from this class's edge to the mat's. Clamping to "mat edge minus
            # sep" instead, as an earlier version did, can land INSIDE this
            # class's own samples and threshold the pillar straight back out.
            if name == "RED":
                mine, theirs = lo_p(a, 1), hi_p(mat, 1)      # my low vs mat's high
                if mine <= theirs + sep:
                    return (False, "red and the mat have overlapping 'a' values - "
                            "the camera cannot tell them apart. Re-sample, or fix "
                            "exposure / white balance", {})
                a_lo, a_hi = (mine + theirs) / 2.0, 255.0
            else:
                mine, theirs = hi_p(a, 1), lo_p(mat, 1)      # my high vs mat's low
                if mine >= theirs - sep:
                    return (False, "green and the mat have overlapping 'a' values - "
                            "the camera cannot tell them apart. Re-sample, or fix "
                            "exposure / white balance", {})
                a_lo, a_hi = 0.0, (mine + theirs) / 2.0

            L_lo = max(10.0, lo_p(a, 0) - max(40.0, 4 * m))
            b_lo = max(0.0, lo_p(a, 2) - 3 * m)
            b_hi = min(255.0, hi_p(a, 2) + 3 * m)
            out[name] = {"L": (L_lo, 255.0), "a": (a_lo, a_hi), "b": (b_lo, b_hi)}

        # The mat gets even more headroom than the pillars. It is not evenly
        # lit - brighter under the lights, darker at the far end - and when
        # this threshold is missed the floor test rejects EVERY pillar at once
        # with reason F: one number taking the whole detector down. Being loose
        # costs little, because the near-neutral a/b test is what actually
        # identifies the mat.
        floor_L = max(0.0, lo_p(mat, 0) - max(60.0, 6 * m))
        ab_tol = float(np.ceil(max(10.0, max(np.abs(mat[:, 1] - 128).max(),
                                             np.abs(mat[:, 2] - 128).max()) + m)))

        def chroma(a):
            return np.hypot(a[:, 1] - 128, a[:, 2] - 128)

        gap = min(chroma(red).mean(), chroma(grn).mean()) - chroma(mat).mean()
        chroma_min = float(max(5.0, round(gap * 0.5)))

        vals = {
            "RED_L_LO": out["RED"]["L"][0],     "RED_L_HI": out["RED"]["L"][1],
            "RED_A_LO": out["RED"]["a"][0],     "RED_A_HI": out["RED"]["a"][1],
            "RED_B_LO": out["RED"]["b"][0],     "RED_B_HI": out["RED"]["b"][1],
            "GREEN_L_LO": out["GREEN"]["L"][0], "GREEN_L_HI": out["GREEN"]["L"][1],
            "GREEN_A_LO": out["GREEN"]["a"][0], "GREEN_A_HI": out["GREEN"]["a"][1],
            "GREEN_B_LO": out["GREEN"]["b"][0], "GREEN_B_HI": out["GREEN"]["b"][1],
            "FLOOR_L_MIN": floor_L,
            "FLOOR_AB_TOL": ab_tol,
            "LAB_CHROMA_MIN": chroma_min,
            "USE_LAB": True,
        }
        msg = "fitted and applied live"
        if gap < 12:
            msg += (". WARNING: the pillars are barely more colourful than the "
                    "mat - check exposure and white balance")
        return True, msg, vals
