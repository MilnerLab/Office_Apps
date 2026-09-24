"""The two calibration scans.

Each sweep reduced to one scalar, cross-correlation (filled circles, solid fit) and
spectrometer (open diamonds, dashed fit), with residuals below: (a) f_0 against the arm
delay, (b) the swept bandwidth tau*a0/2pi against the arm delay, (c) the swept bandwidth
against the grating separation. Each fit is drawn about its own zero crossing.

Data: _temp/shaped_usCFG_paper/joint_fit.json and scan_fits.json, from the two XCORR scans
under Z:\\Droplets\\shaped_usCFG_paper (20260825\\XCORR_scan_L_20260825_200235.h5 and
20260831\\XCORR_scan_d_20260831_131421.h5).
Run run_analysis first (python -m manuscript_plotting_scripts.shaped_usCFG_paper.run_analysis).
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain.xcorr_fit import TAU_PS

NAME = "fig_char_scans"
C_X, C_S = PlotColor.BLUE, PlotColor.RED       # cross-correlation, spectrometer
K = TAU_PS * 1e-3                               # a0/2pi (MHz/ps) -> Delta f (GHz)


def main() -> None:
    rows = json.loads((config.TEMP_DIR / "joint_fit.json").read_text())["rows"]
    fits = json.loads((config.TEMP_DIR / "scan_fits.json").read_text())
    d = [r for r in rows if r["tag"] == "scan_d"]
    sl = [r for r in rows if r["tag"] == "scan_L"]

    def col(sub: list[dict], key: str, scale: float = 1.0) -> np.ndarray:
        return scale * np.array([r[key] for r in sub])

    df0 = fits["df0"]
    z_dt, z_l = fits["zero_dt"], fits["zero_L"]

    def line(g, p):
        return p["a"] * g

    def law(g, p):
        x = p["a"] * g
        return x / (1.0 - x / df0)

    dt = col(d, "dt")
    cases = [
        (dt - z_dt, z_dt, col(d, "f0"), col(d, "sf0"), col(d, "f0_s"), col(d, "sf0_s"),
         fits["f0"]["x"], fits["f0"]["s"], line, r"$\Delta t$ (ps)", r"$f_0$ (GHz)"),
        # swept bandwidth on the delay scan, Delta f = tau a0/2pi, so the chirp fit rescales by tau
        (dt - z_dt, z_dt, col(d, "chirp", K), col(d, "schirp", K),
         col(d, "chirp_s", K), col(d, "schirp_s", K),
         dict(fits["chirp"]["x"], a=K * fits["chirp"]["x"]["a"]),
         dict(fits["chirp"]["s"], a=K * fits["chirp"]["s"]["a"]), line,
         r"$\Delta t$ (ps)", r"$\Delta f_{\rm CFG}$ (GHz)"),
        (col(sl, "L") - z_l, z_l, col(sl, "df"), col(sl, "sdf"),
         col(sl, "chirp_s", K), col(sl, "schirp_s", K),
         fits["df"]["x"], fits["df"]["s"], law, r"$L$ (mm)", r"$\Delta f_{\rm CFG}$ (GHz)"),
    ]

    fig = plt.figure(layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1])
    xcorr = dict(fmt="o", ms=2.4, mew=0.5, color=C_X, ecolor=C_X)
    spec = dict(fmt="D", ms=2.8, mew=0.6, mfc="none", color=C_S, ecolor=C_S)
    for j, (x, x0, y, s, ys, ss, px, ps, fn, xl, yl) in enumerate(cases):
        def own(u, p):                                  # each fit about its own origin
            return fn(u - (p["z"] - x0), p)
        a0 = fig.add_subplot(gs[0, j])
        a1 = fig.add_subplot(gs[1, j], sharex=a0)
        g = np.linspace(min(0, x.min()) * 1.08, max(0, x.max()) * 1.08, 400)
        a0.plot(g, own(g, px), ls="-", marker="", color=C_X, zorder=2)
        a0.plot(g, own(g, ps), ls="--", marker="", color=C_S, zorder=2)
        a0.errorbar(x, y, yerr=s, zorder=4, **xcorr)
        a0.errorbar(x, ys, yerr=ss, zorder=5, **spec)
        a0.axhline(0, color=PlotColor.BLACK, marker="")
        a0.axvline(0, color=PlotColor.BLACK, marker="")
        a0.set_ylabel(yl)
        a0.tick_params(labelbottom=False)
        a0.text(.04, .93, rf"\textbf{{({'abc'[j]})}}", transform=a0.transAxes, va="top")
        a1.errorbar(x, y - own(x, px), yerr=s, **xcorr)
        a1.errorbar(x, ys - own(x, ps), yerr=ss, **spec)
        a1.axhline(0, color=PlotColor.BLACK, marker="")
        a1.set_xlabel(xl)
        a1.set_ylabel("resid.")
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
