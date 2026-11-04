"""A full and a truncated cross-correlation of one accelerating centrifuge (2026-10-15).

Two scans under ``config.XCORR_DEMO_ROOT``, same centrifuge (DA = 17.8 mm, GA = 10 mm),
phase stabilization on, 100 repeats per delay, on the same (non-uniform) probe grid:

  20261015355_CFG_Full_Slow.csv        prism out
  20261015419_CFG_Truncated_Slow.csv   prism at 14 mm

LabVIEW format: first column probe stage mm, then the repeats. The probe grid is denser
where the fringe is faster, so the step is not constant.

The detector sees the two arms projected onto one polarization,

    S(t) = b + I1(t) + I2(t) E(t) + 2 v sqrt(I1 I2 E) cos phi(t),

with I1 the delay arm, I2 the shaper arm, E = 1 for the full scan and an erfc edge for
the truncated one. After the cut only I1 is left, which is the smooth tail of the
truncated scan. I1 and I2 are exp(quartic) envelopes, phi is a cubic, and both scans
share every parameter except the phase offset and the edge. The fringe sits at 2 f_usCFG
(``xcorr_fit.FRINGE_PER_USCFG``), so f_usCFG(t) = phi'(t) / (2 pi) / 2.

The fit is seeded by a grid over the fringe's centre frequency and chirp on the full
scan, with the phase offset solved linearly at each grid point.

No plotting here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain.xcorr_fit import (
    C_MM_PER_PS, FRINGE_PER_USCFG)

FULL = config.XCORR_DEMO_ROOT / "20261015355_CFG_Full_Slow.csv"
TRUNC = config.XCORR_DEMO_ROOT / "20261015419_CFG_Truncated_Slow.csv"

#: Fit origin on the probe axis (ps from the first point). Only conditions the polynomials.
T_ORIGIN_PS = 300.0
#: Scale of the envelope polynomials, ps.
S_PS = 150.0
#: Common residual weight, V. The repeat spread (~0.2 mV) is much smaller than the
#: scan-to-scan drift, so the two scans are weighted equally rather than by it.
SIGMA_V = 1.5e-3

# p = b, A1[5], A2[5], v, c0, c1, c2, c3, c0_trunc, t_cut, w
N_PAR = 19


def load(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(probe delay ps from the first point, mean V, standard error V), double pass."""
    a = np.loadtxt(path)
    x, Y = a[:, 0], a[:, 1:]
    return (2.0 * (x - x[0]) / C_MM_PER_PS, Y.mean(1),
            Y.std(1, ddof=1) / np.sqrt(Y.shape[1]))


def _env(a, u):
    s = u / S_PS
    return np.exp(a[0] + a[1] * s + a[2] * s ** 2 + a[3] * s ** 3 + a[4] * s ** 4)


def edge(p, u):
    """Intensity transmission of the shaper arm across the cut."""
    return 0.5 * erfc((u - p[17]) / p[18])


def arms(p, u, truncated: bool):
    """(I1, I2 E) on the fit axis u."""
    I1, I2 = _env(p[1:6], u), _env(p[6:11], u)
    return I1, I2 * (edge(p, u) if truncated else 1.0)


def model(p, u, truncated: bool):
    I1, I2 = arms(p, u, truncated)
    c0 = p[16] if truncated else p[12]
    phi = c0 + p[13] * u + p[14] * u ** 2 + p[15] * u ** 3
    return p[0] + I1 + I2 + 2.0 * p[11] * np.sqrt(I1 * I2) * np.cos(phi)


@dataclass
class DemoFit:
    u: np.ndarray            # fit axis, ps
    full: tuple              # (y, e) V
    trunc: tuple
    p: np.ndarray
    rms: float               # rms residual over both scans, V

    def f_uscfg(self, u):
        """f_usCFG(u), GHz."""
        u = np.asarray(u, float)
        p = self.p
        return ((p[13] + 2 * p[14] * u + 3 * p[15] * u ** 2)
                / (2 * np.pi) * 1e3 / FRINGE_PER_USCFG)

    @property
    def t_cut(self) -> float:
        return float(self.p[17])

    @property
    def edge_1090(self) -> float:
        """10-90 % width of the fitted intensity edge, ps."""
        return float(abs(self.p[18]) * 1.8124)


def _seed(u, y):
    """Envelope from a smooth fit, then the (f, chirp) grid with c0 solved linearly."""
    s = u / S_PS
    base = float(np.percentile(y, 2))
    a = np.polyfit(s, np.log(np.clip(y - base, 1e-4, None)), 4)[::-1]
    env = _env(a, u) / 2.0
    best = None
    for f in np.linspace(5.0, 20.0, 61):              # f_usCFG at u = 0, GHz
        for ch in np.linspace(0.0, 0.08, 33):          # GHz/ps
            w1 = 2 * np.pi * FRINGE_PER_USCFG * f / 1e3
            w2 = 2 * np.pi * FRINGE_PER_USCFG * ch / 1e3 / 2
            ph = w1 * u + w2 * u ** 2
            M = np.column_stack([env * np.cos(ph), env * np.sin(ph)])
            r = y - base - 2 * env
            coef, *_ = np.linalg.lstsq(M, r, rcond=None)
            cost = float(np.sum((M @ coef - r) ** 2))
            if best is None or cost < best[0]:
                best = (cost, w1, w2, float(np.arctan2(-coef[1], coef[0])))
    _, w1, w2, c0 = best
    a2 = a.copy(); a2[0] -= np.log(2.0)
    return np.concatenate([[base], a2, a2, [0.9, c0, w1, w2, 0.0, c0], [50.0, 5.0]])


def fit() -> DemoFit:
    t, yF, eF = load(FULL)
    t2, yT, eT = load(TRUNC)
    if not np.allclose(t, t2):
        raise ValueError("the two scans are not on the same probe grid")
    u = t - T_ORIGIN_PS
    p0 = _seed(u, yF)

    def resid(p):
        return np.concatenate([model(p, u, False) - yF, model(p, u, True) - yT]) / SIGMA_V

    r = least_squares(resid, p0, max_nfev=50000, x_scale="jac")
    p = r.x
    if p[13] < 0:                       # cos is even: canonicalise on f > 0
        p[[12, 13, 14, 15, 16]] *= -1
    rms = float(np.sqrt(2 * r.cost / (2 * len(u)))) * SIGMA_V
    return DemoFit(u, (yF, eF), (yT, eT), p, rms)


if __name__ == "__main__":
    F = fit()
    print(f"rms {F.rms * 1e3:.2f} mV, v = {F.p[11]:.2f}")
    print(f"cut at u = {F.t_cut:.1f} ps, 10-90 {F.edge_1090:.1f} ps, "
          f"f(cut) = {F.f_uscfg(F.t_cut):.2f} GHz")
    for uu in (-153, 0, 153):
        print(f"f({uu}) = {F.f_uscfg(uu):.2f} GHz")
