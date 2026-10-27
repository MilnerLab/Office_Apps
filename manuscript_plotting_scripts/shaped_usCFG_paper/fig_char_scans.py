"""The two calibration scans.

Each sweep reduced to one scalar, cross-correlation (filled circles, grey dashed fit). Both fits
are straight lines, as in the linear chirp formula of the usCFG paper; the slope of (b) is
printed on run. The spectrometer points moved to the stabilization paper; set
SHOW_SPECTROMETER to draw them again (open diamonds, dashed fit):
(a) f_0 against the arm delay, (b) the acceleration a0/2pi against the grating separation.
Each fit is drawn about its own zero crossing. The acceleration against the arm delay (the
TOD term) moved to the TOD-compensation figure, fig_char_todcomp.

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
C_FIT = "0.5"                                   # grey, for the fit lines
SHOW_SPECTROMETER = False                      # the spectrometer comparison is Mahr2026stab's


def main() -> None:
    rows = json.loads((config.TEMP_DIR / "joint_fit.json").read_text())["rows"]
    fits = json.loads((config.TEMP_DIR / "scan_fits.json").read_text())
    d = [r for r in rows if r["tag"] == "scan_d"]
    sl = [r for r in rows if r["tag"] == "scan_L"]

    def col(sub: list[dict], key: str, scale: float = 1.0) -> np.ndarray:
        return scale * np.array([r[key] for r in sub])

    z_dt = fits["zero_dt"]

    def line(g, p):
        return p["a"] * g

    # Linear fit of the acceleration against the grating separation, uncertainties scaled
    # by sqrt(chi2/dof); its zero is the stage reference, as in the other panel.
    L_raw, y_L, s_L = col(sl, "L"), col(sl, "chirp"), col(sl, "schirp")
    (a_L, b_L), cov = np.polyfit(L_raw, y_L, 1, w=1 / s_L, cov="unscaled")
    chi2 = float(np.sum(((y_L - (a_L * L_raw + b_L)) / s_L) ** 2))
    dof = len(L_raw) - 2
    cov = cov * max(1.0, chi2 / dof)
    z_l = -b_L / a_L
    lin_acc = {"a": a_L, "z": z_l}
    print(f"acceleration vs L linear slope: {a_L:.4f} +/- {np.sqrt(cov[0, 0]):.4f} MHz/ps "
          f"per mm, zero at L = {z_l:.2f} mm, chi2/dof = {chi2 / dof:.1f}, "
          f"max |acceleration| = {np.abs(y_L).max():.1f} MHz/ps")

    dt = col(d, "dt")
    cases = [
        (dt - z_dt, z_dt, col(d, "f0"), col(d, "sf0"), col(d, "f0_s"), col(d, "sf0_s"),
         fits["f0"]["x"], fits["f0"]["s"], line, r"$\Delta t$ (ps)", r"$f_0$ (GHz)"),
        (L_raw - z_l, z_l, y_L, s_L,
         col(sl, "chirp_s"), col(sl, "schirp_s"),
         lin_acc, dict(fits["df"]["s"], a=fits["df"]["s"]["a"] / K), line, r"$L$ (mm)",
         r"$a_0$ (MHz/ps)"),
    ]

    fig = plt.figure(layout="constrained")
    gs = fig.add_gridspec(1, len(cases))
    xcorr = dict(fmt="o", ms=2.4, mew=0.5, color=C_X, ecolor=C_X)
    spec = dict(fmt="D", ms=2.8, mew=0.6, mfc="none", color=C_S, ecolor=C_S)
    for j, (x, x0, y, s, ys, ss, px, ps, fn, xl, yl) in enumerate(cases):
        def own(u, p):                                  # each fit about its own origin
            return fn(u - (p["z"] - x0), p)
        a0 = fig.add_subplot(gs[0, j])
        g = np.linspace(min(0, x.min()) * 1.15, max(0, x.max()) * 1.08, 400)  # (b): to -120
        a0.plot(g, own(g, px), ls="--", marker="", color=C_FIT, zorder=2)
        a0.errorbar(x, y, yerr=s, zorder=4, **xcorr)
        if SHOW_SPECTROMETER:
            a0.plot(g, own(g, ps), ls="--", marker="", color=C_S, zorder=2)
            a0.errorbar(x, ys, yerr=ss, zorder=5, **spec)
        if j == 0:                                      # the origin in a corner of the frame
            a0.set_xlim(left=0)
            a0.set_ylim(bottom=0)
        else:                                           # ticks every 40 mm leave the
            a0.set_xlim(-120, 0)                        # top-left grid cell clear for (b)
            a0.set_xticks(np.arange(-120, 1, 40))
            a0.set_ylim(top=0)
        a0.set_xlabel(xl)
        a0.set_ylabel(yl)
        a0.annotate(rf"\textbf{{({'ab'[j]})}}", xy=(0, 1), xycoords="axes fraction",
                    xytext=(3, -3), textcoords="offset points", ha="left",
                    va="top")                           # same corner, inside one grid cell
    fig.get_layout_engine().set(wspace=0.12)            # room between the two panels
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
