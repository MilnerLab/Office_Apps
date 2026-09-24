"""One spectrometer fit.

The L = -43.12 mm, dt = 0 spectrometer trace mapped to time, its two pinball envelopes
(dotted) and the final cubic-phase fit between them (solid).

Data: _temp/shaped_usCFG_paper/fit_demo.npz, from the 2026-08-25 grating scan
(Z:\\Droplets\\shaped_usCFG_paper\\20260825\\XCORR_scan_L_20260825_200235.h5).
Run run_analysis first (python -m manuscript_plotting_scripts.shaped_usCFG_paper.run_analysis).
"""
import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor
from manuscript_plotting_scripts.shaped_usCFG_paper import config

NAME = "fig_char_fitdemo"


def main() -> None:
    d = np.load(config.TEMP_DIR / "fit_demo.npz")
    ts, sp, s_up, s_lo, cs = d["ts"], d["s"], d["sUd"], d["sLd"], d["cs"]
    k = 1.0 / float(s_up.max())
    fit = (s_up + s_lo) / 2 + (s_up - s_lo) / 2 * np.cos(
        cs[0] + cs[1] * ts + cs[2] * ts**2 + cs[3] * ts**3)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(ts, sp * k, ls="", marker=".", ms=1.9, color=PlotColor.RED)
    ax.plot(ts, s_up * k, ls=":", marker="", color=PlotColor.GRAY)
    ax.plot(ts, s_lo * k, ls=":", marker="", color=PlotColor.GRAY)
    ax.plot(ts, fit * k, ls="-", marker="", color=PlotColor.BLUE)
    ax.set_xlabel(r"$t$ (ps)")
    ax.set_ylabel("intensity (arb. u.)")
    ax.set_ylim(0, 1.08)
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
