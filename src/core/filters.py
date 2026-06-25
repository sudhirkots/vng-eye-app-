"""Jitter filters for pupil tracking (see docs/TRACKING_PHILOSOPHY.md).

Goal: remove measurement noise (shimmer) WITHOUT smearing real fast eye movements
(saccades, nystagmus beats). The 1€ filter is adaptive — it smooths hard when the
eye is slow/still and backs off automatically as velocity rises, so quick movements
pass through with little lag. A 3-frame median in front removes single-frame spikes.

All filters RESET on a lost/blink frame (input None) so they never bridge a gap.
The raw signal is always kept separately; these only produce the *filtered* signal.
"""
import math


class Median3:
    """Causal 3-sample median (removes single-frame spikes)."""

    def __init__(self):
        self.buf = []

    def reset(self):
        self.buf = []

    def __call__(self, v):
        if v is None:
            self.reset()
            return None
        self.buf.append(v)
        if len(self.buf) > 3:
            self.buf.pop(0)
        return sorted(self.buf)[len(self.buf) // 2]


class OneEuroFilter:
    """1€ filter (Casiez, Roussel, Vogel 2012) for one scalar signal.

    mincutoff: baseline cutoff (Hz) — lower = more smoothing when still.
    beta:      speed coefficient — higher = backs off faster on quick movement
               (preserves saccades/nystagmus). dcutoff: cutoff for the speed estimate.
    """

    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.02, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        if x is None:
            self.reset()
            return None
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        if t is not None and self.t_prev is not None and t > self.t_prev:
            self.freq = 1.0 / (t - self.t_prev)
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# filter presets: mincutoff, beta. Tuned so a still eye is de-shimmered but FAST eye movements
# (saccades / nystagmus beats) pass through — high beta + responsive speed estimate (dcutoff) make
# the 1€ filter back off sharply on any quick movement. (None mode bypasses filtering entirely.)
_PRESETS = {
    "light": (2.5, 0.5),       # very gentle; preserves almost everything
    "adaptive": (1.0, 0.7),    # de-shimmer when still, but let nystagmus beats through
}


class PointFilter:
    """1€ filter applied independently to x and y. Resets on a None (lost/blink) input.
    mode in {none, light, adaptive}. NO median by default — a 3-frame median erased brief
    nystagmus fast phases; enable use_median only for non-eye-movement points if ever needed."""

    def __init__(self, mode="adaptive", freq=30.0, use_median=False):
        self.mode = mode
        mincutoff, beta = _PRESETS.get(mode, _PRESETS["adaptive"])
        self.use_median = use_median
        self.mx, self.my = Median3(), Median3()
        self.fx = OneEuroFilter(freq, mincutoff, beta, dcutoff=2.0)
        self.fy = OneEuroFilter(freq, mincutoff, beta, dcutoff=2.0)

    def reset(self):
        for f in (self.mx, self.my, self.fx, self.fy):
            f.reset()

    def __call__(self, x, y, t=None):
        if x is None or y is None:
            self.reset()
            return None, None
        if self.mode == "none":
            return x, y
        if self.use_median:
            x, y = self.mx(x), self.my(y)
        return self.fx(x, t), self.fy(y, t)
