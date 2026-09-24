"""Phase-stabilized centrifuge oscillations against their own cross-correlation.

(a) The averaged <cos^2 theta_2D> trace (nine repeats of one probe delay scan).
(b) The short-time Fourier transform of its oscillating part, with the fringe frequency
fitted to the accompanying cross-correlation (green, short dashes) and the same fringe
frequency as the calibration constants predict it with nothing fitted (cyan, solid).
Both are 2 f_CFG; the one fitted number is the delay-origin offset between the two scans.
(c) The accompanying cross-correlation, on the jet delay axis.
The x range is where the delay grid supports the beat (domain/jet.py).

Data: Z:\\Droplets\\shaped_usCFG_paper\\Jet (Jet Centrifuge Oscillations\\*_ScanFile.dat
and Jet Truncation\\20260904\\XCORR_20260903_jet_accompany_scan.h5), reduced by
domain/jet.py; the predicted beat from _temp/shaped_usCFG_paper/jet_prediction.json.
Run run_analysis first (python -m manuscript_plotting_scripts.shaped_usCFG_paper.run_analysis).
"""
import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor, PlotColorMap
from base_core.quantities.enums import Prefix
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import jet

NAME = "fig_jet_oscillations"
C_XC = PlotColor.GREEN      # (b): the cross-correlation's fitted beat, short dashes
C_PRED = "#22e0ff"          # (b): the calibration's predicted beat, cyan, solid
DPI = 300                   # the spectrogram is resampled at this


def main() -> None:
    k = jet.oscillations()
    ps = lambda xs: np.array([x.value(Prefix.PICO) for x in xs])
    t = ps(k["trace"].delays)
    y = np.array([m.value for m in k["trace"].measured_values])
    tc = ps(k["tc"])
    f = np.array([v.value(Prefix.GIGA) for v in k["f"]])
    S = k["S"]
    ux = ps(k["xcorr"].delays)
    vx = np.array([m.value for m in k["xcorr"].measured_values])
    shift, lo, hi = (k[n].value(Prefix.PICO) for n in ("shift", "lo", "hi"))
    beat, pbeat = k["beat"].numpy, k["pbeat"].numpy

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=config.FIGURE_SIZES_IN[NAME], dpi=DPI,
                           height_ratios=[0.243, 0.327, 0.273], layout="constrained")

    # (a) the jet
    ax[0].plot(t, y, ls="-", marker="o", ms=1.6, mfc="none", mew=0.35, color=PlotColor.BLUE)
    ax[0].axhline(0.5, ls=":", marker="", color=PlotColor.GRAY)
    ax[0].set_ylabel(r"$\langle\cos^2\theta_{\rm 2D}\rangle$")
    ax[0].text(.025, .94, r"\textbf{(a)}", transform=ax[0].transAxes, va="top")

    # (b) spectrogram with the fitted and predicted beats
    ax[1].imshow(S / S.max(), origin="lower", aspect="auto",
                 extent=[tc.min(), tc.max(), f.min(), f.max()],
                 cmap=PlotColorMap.MAGMA.value, vmin=0, vmax=0.5, interpolation="bilinear")
    tb = np.linspace(lo, hi, 400)
    ax[1].plot(tb, np.abs(pbeat(tb - shift)), ls="-", marker="", lw=1.6, color=C_PRED)
    ax[1].plot(tb, np.abs(beat(tb - shift)), ls=(0, (1.5, 1.2)), marker="", lw=1.6,
               color=C_XC, dash_capstyle="butt")     # short dashes on top
    ax[1].grid(False)
    ax[1].set_ylim(0, jet.FMAX_GHZ)
    ax[1].set_ylabel(r"$2f_{\rm CFG}$ (GHz)")
    ax[1].text(.05, .94, r"\textbf{(b)}", transform=ax[1].transAxes, va="top",
               color=PlotColor.WHITE)

    # (c) the accompanying cross-correlation, on the jet axis
    ax[2].plot(ux + shift, vx, ls="-", marker="", lw=0.4, color=PlotColor.BLACK)
    ax[2].set_ylabel("xcorr (V)")
    ax[2].set_xlabel("probe delay (ps)")
    ax[2].set_ylim(0.0, 0.66)
    ax[2].text(.965, .94, r"\textbf{(c)}", transform=ax[2].transAxes, va="top", ha="right")

    ax[0].set_xlim(lo, hi)
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
