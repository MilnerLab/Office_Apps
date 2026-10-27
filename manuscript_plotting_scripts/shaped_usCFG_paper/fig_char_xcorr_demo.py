"""Cross-correlation of one accelerating centrifuge, full and truncated.

The full centrifuge (grey, dashed, filled circles) and the same centrifuge with the
truncation prism at 14 mm (red, solid, open circles), on one axis, both normalised to the full scan's maximum,
against the probe delay from the full scan's fitted envelope centre (t = 0, as for every
cross-correlation in the paper). `draw` is also the main panel of fig_char_xcorr.

Data: Z:\Droplets\20261015\XCORR\Accelerating (20261015355 full, 20261015419
truncated), loaded by domain/xcorr_demo.py.
"""
import matplotlib.pyplot as plt

import sys

import numpy as np
from scipy.optimize import least_squares

from base_core.plotting.enums import PlotColor
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import xcorr_demo

NAME = "fig_char_xcorr_demo"


def envelope_centre(t, y) -> float:
    """t = 0 of the probe axis: the centre mu of the full scan's Gaussian envelope in
    xcorr_fit's model, base + A exp(-(t - mu)^2 / 2 sigma^2) (1 + v cos phi), phi cubic.

    xcorr_fit.full_fit seeds phi from an STFT. This scan is sampled only every 25 ps at the
    start, and with that seed the envelope locks onto the early bright lobe (mu 169 ps,
    sigma 91 ps). Here phi is seeded from the joint two-arm fit (xcorr_demo.fit), which
    follows the fringes, and the envelope from a grid of starts; the lowest cost wins. The
    session's other untruncated scans (331, 337) give 256.7 and 250.5 ps; this one 247.9."""
    T0, c = xcorr_demo.T_ORIGIN_PS, xcorr_demo.fit().p[12:16]

    def model(q):
        base, A, mu, sig, v, c0, c1, c2, c3 = q
        u = t - T0
        return base + A * np.exp(-(t - mu) ** 2 / (2 * sig ** 2)) * (1 + v * np.cos(c0 + c1 * u + c2 * u ** 2 + c3 * u ** 3))

    lo = [-1, 0, 0, 30, 0, -np.inf, -np.inf, -np.inf, -np.inf]
    hi = [1, 1, 700, 400, 1, np.inf, np.inf, np.inf, np.inf]
    best = None
    for mu in np.arange(150, 360, 15):
        for sig in (100, 130, 160, 200):
            for c0 in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                q0 = [np.percentile(y, 5), y.max() / 2, mu, sig, 0.7, c0, *c[1:]]
                r = least_squares(lambda q: model(q) - y, q0, x_scale="jac", max_nfev=4000, bounds=(lo, hi))
                if best is None or r.cost < best.cost:
                    best = r
    return float(best.x[2])


def draw(ax, flip: bool = False) -> float:
    """Both traces on ``ax`` against the probe delay from the full scan's envelope centre.
    ``flip`` draws the full trace red and solid instead (Ian's suggestion). Returns t0, ps."""
    t, yF, eF = xcorr_demo.load(xcorr_demo.FULL)
    tT, yT, eT = xcorr_demo.load(xcorr_demo.TRUNC)
    k = 1.0 / float(yF.max())
    t0 = envelope_centre(t, yF)
    # full: black short-dashed line, filled circles; truncated: red solid line, open circles
    styles = ((PlotColor.BLACK, (0, (2, 1.5)), PlotColor.BLACK), (PlotColor.RED, "-", "white"))
    if flip:   # full: solid line, filled circles; truncated: short-dashed line, open circles
        styles = ((PlotColor.RED, "-", PlotColor.RED), (PlotColor.GRAY, (0, (2, 1.5)), "white"))
    mk = "o"
    for tt, y, (col, ls, face) in ((t, yF, styles[0]), (tT, yT, styles[1])):
        ax.plot(tt - t0, y * k, ls=ls, lw=0.75, marker=mk, ms=2.8, mew=0.6,
                mfc=face, mec=col, color=col)
    ax.set_xlim(t.min() - t0, t.max() - t0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("probe delay (ps)")
    ax.set_ylabel("XCORR signal (arb. u.)")
    return t0


def main(flip: bool = False) -> None:
    """``flip`` (``--flip``) saves the colour-swapped version as fig_char_xcorr_demo_swapped."""
    fig, ax = plt.subplots(layout="constrained")
    t0 = draw(ax, flip=flip)
    F = xcorr_demo.fit()
    print(f"t = 0 at {t0:.1f} ps from the first point; cut at t = {F.t_cut + xcorr_demo.T_ORIGIN_PS - t0:.1f} ps")
    config.save_figure(fig, NAME + ("_swapped" if flip else ""))


if __name__ == "__main__":
    main(flip="--flip" in sys.argv[1:])
