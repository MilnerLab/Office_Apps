"""Cancelling the TOD sweep of the constant-frequency centrifuge. Partly a placeholder.

(a) the acceleration a0/2pi against the arm delay at L = 0, from the delay scan (filled
    circles, linear fit drawn about its zero crossing). TOD alone makes it grow with the
    delay. The second line, at the compensating L for each delay, is still to be measured
    and is only marked.
(b), (c) empty frames for the short-time Fourier transforms of a high-frequency centrifuge
    (f_0 = F0_DEMO_GHZ) at L = 0 and at the compensating L, still to be measured.

Data: _temp/shaped_usCFG_paper/joint_fit.json and scan_fits.json, from
Z:\\Droplets\\shaped_usCFG_paper\\20260831\\XCORR_scan_d_20260831_131421.h5.
Run run_analysis first (python -m manuscript_plotting_scripts.shaped_usCFG_paper.run_analysis).
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor
from manuscript_plotting_scripts.shaped_usCFG_paper import config

NAME = "fig_char_todcomp"
C_X = PlotColor.BLUE
F0_DEMO_GHZ = 360                               # within the compensable range (~480 GHz)


def placeholder(ax, label: str, text: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.text(.5, .5, text, transform=ax.transAxes, ha="center", va="center",
            color=PlotColor.RED)
    ax.text(.04, .93, rf"\textbf{{{label}}}", transform=ax.transAxes, va="top")


def main() -> None:
    rows = json.loads((config.TEMP_DIR / "joint_fit.json").read_text())["rows"]
    fits = json.loads((config.TEMP_DIR / "scan_fits.json").read_text())
    d = [r for r in rows if r["tag"] == "scan_d"]
    z_dt, p = fits["zero_dt"], fits["chirp"]["x"]

    x = np.array([r["dt"] for r in d]) - z_dt
    y = np.array([r["chirp"] for r in d])
    s = np.array([r["schirp"] for r in d])
    print(f"acceleration vs dt at L = 0: {p['a']:.3f} +/- {p['sa']:.3f} MHz/ps per ps")

    fig = plt.figure(layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))

    a0 = fig.add_subplot(gs[0, :])
    g = np.linspace(min(0, x.min()) * 1.08, max(0, x.max()) * 1.08, 400)
    a0.plot(g, p["a"] * (g - (p["z"] - z_dt)), ls="-", marker="", color=C_X, zorder=2,
            label=r"$L=0$")
    a0.errorbar(x, y, yerr=s, fmt="o", ms=2.4, mew=0.5, color=C_X, ecolor=C_X, zorder=4)
    a0.axhline(0, color=PlotColor.BLACK, marker="")
    a0.axvline(0, color=PlotColor.BLACK, marker="")
    a0.plot([], [], ls="", marker="s", ms=2.4, mfc="none", color=PlotColor.RED,
            label="compensating $L$ (to be measured)")
    a0.legend(loc="lower right")
    a0.set_xlabel(r"$\Delta t$ (ps)")
    a0.set_ylabel(r"$\dot f_{\rm CFG}$ (MHz/ps)")
    a0.text(.02, .93, r"\textbf{(a)}", transform=a0.transAxes, va="top")

    placeholder(fig.add_subplot(gs[1, 0]), "(b)",
                rf"STFT, $f_0={F0_DEMO_GHZ}$ GHz" "\n" r"cfCFG" "\n(to be measured)")
    placeholder(fig.add_subplot(gs[1, 1]), "(c)",
                rf"STFT, $f_0={F0_DEMO_GHZ}$ GHz" "\n" r"compensating $L$" "\n(to be measured)")
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
