"""Extracted centrifuge frequency across both scans.

f_usCFG(t) for every sweep of the two calibration scans, cross-correlation (solid) and
spectrometer (dashed): (a) the grating scan, (b) the delay scan. Line styles, each scan's
fixed coordinate and the order of the curves are stated in the caption, so the panels
carry no legend.

Data: _temp/shaped_usCFG_paper/joint_fit.json, from the two XCORR scans under
Z:\\Droplets\\shaped_usCFG_paper (20260825\\XCORR_scan_L_20260825_200235.h5 and
20260831\\XCORR_scan_d_20260831_131421.h5).
Run run_analysis first (python -m manuscript_plotting_scripts.shaped_usCFG_paper.run_analysis).
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor, PlotColorMap
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain.xcorr_fit import (
    FRINGE_PER_USCFG, KEEP_SIGMA)

NAME = "fig_char_freqtime"
KF = 1e3 / (2 * np.pi) / FRINGE_PER_USCFG


def f_curve(row: dict, u: np.ndarray, spectrometer: bool = False) -> np.ndarray:
    s = "_s" if spectrometer else ""
    return row["f0" + s] + row["chirp" + s] * 1e-3 * u + 3 * row["c3" + s] * KF * u**2


def main() -> None:
    rows = json.loads((config.TEMP_DIR / "joint_fit.json").read_text())["rows"]
    scans = ([r for r in rows if r["tag"] == "scan_L"], [r for r in rows if r["tag"] == "scan_d"])
    cmap = plt.get_cmap(PlotColorMap.DEFAULT.value)

    fig, ax = plt.subplots(2, 1, sharex=True, layout="constrained")
    for k, sub in enumerate(scans):
        for col, row in zip(cmap(np.linspace(0, .86, len(sub))), sub):
            u = np.linspace(-KEEP_SIGMA * row["sigma"], KEEP_SIGMA * row["sigma"], 400)
            ax[k].plot(u, f_curve(row, u), ls="-", marker="", color=col)
            ax[k].plot(u, f_curve(row, u, True), ls="--", marker="", color=col, alpha=.9)
        ax[k].axhline(0, color=PlotColor.BLACK, marker="")
        ax[k].axvline(0, color=PlotColor.BLACK, marker="")
        ax[k].set_ylabel(r"$f_{\rm CFG}$ (GHz)")
        ax[k].text(.025, .93, rf"\textbf{{({'ab'[k]})}}", transform=ax[k].transAxes, va="top")
    ax[1].set_xlabel(r"$t$ (ps)")
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
