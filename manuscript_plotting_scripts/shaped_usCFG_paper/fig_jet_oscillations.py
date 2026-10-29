"""Phase-stabilized centrifuge oscillations against their own cross-correlation.

(a) The averaged <cos^2 theta_2D> trace (nine repeats of one probe delay scan).
(b) The short-time Fourier transform of its oscillating part, with the fringe frequency
fitted to the accompanying cross-correlation (cyan, solid). It is 2 f_CFG; the one number
fitted against the jet is the delay-origin offset between the two scans.
The cross-correlation trace itself is not drawn, only its fitted beat. Spectrogram values
below VMIN of the peak are clipped to black, which hides the noise floor.
The x range is where the delay grid supports the beat (domain/jet.py).

Data: Z:\\Droplets\\shaped_usCFG_paper\\Jet (Jet Centrifuge Oscillations\\*_ScanFile.dat
and Jet Truncation\\20260904\\XCORR_20260903_jet_accompany_scan.h5), reduced by
domain/jet.py.
Run run_analysis first (python -m manuscript_plotting_scripts.shaped_usCFG_paper.run_analysis).
"""
import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor, PlotColorMap
from base_core.quantities.enums import Prefix
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import jet

NAME = "fig_jet_oscillations"
C_XC = "#22e0ff"            # (b): the cross-correlation's fitted beat, cyan, solid
DPI = 300                   # the spectrogram is resampled at this
VMIN, VMAX = 0.12, 0.6     # spectrogram colour range, fraction of its peak


def main() -> None:
    k = jet.oscillations()
    ps = lambda xs: np.array([x.value(Prefix.PICO) for x in xs])
    t = ps(k["trace"].delays)
    y = np.array([m.value for m in k["trace"].measured_values])
    tc = ps(k["tc"])
    f = np.array([v.value(Prefix.GIGA) for v in k["f"]])
    S = k["S"]
    shift, lo, hi = (k[n].value(Prefix.PICO) for n in ("shift", "lo", "hi"))
    beat = k["beat"].numpy

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=config.FIGURE_SIZES_IN[NAME], dpi=DPI,
                           height_ratios=[0.243, 0.327], layout="constrained")
    fig.get_layout_engine().set(h_pad=0, hspace=0)      # panels touch, so the grids join

    # (a) the jet
    ax[0].plot(t, y, ls="-", marker="o", ms=1.6, mfc="none", mew=0.35, color=PlotColor.BLUE)
    ax[0].axhline(0.5, ls=":", marker="", color=PlotColor.GRAY)
    ax[0].set_ylabel(r"$\langle\cos^2\theta_{\rm 2D}\rangle$")
    ax[0].text(.025, .895, r"\textbf{(a)}", transform=ax[0].transAxes, va="top")

    # (b) spectrogram with the cross-correlation's fitted beat
    ax[1].imshow(S / S.max(), origin="lower", aspect="auto",
                 extent=[tc.min(), tc.max(), f.min(), f.max()],
                 cmap=PlotColorMap.MAGMA.value, vmin=VMIN, vmax=VMAX, interpolation="bilinear")
    tb = np.linspace(lo, hi, 400)
    ax[1].plot(tb, np.abs(beat(tb - shift)), ls="-", marker="", lw=1.6, color=C_XC)
    ax[1].set_axisbelow(False)                          # grid over the image, joining (a)'s
    ax[1].set_ylim(0, jet.FMAX_GHZ)
    ax[1].set_ylabel(r"$\langle\cos^2\theta_{\rm 2D}\rangle$ Oscillation" "\n"
                     r"Frequency (GHz)")
    ax[1].set_xlabel("Probe Delay (ps)")
    ax[1].text(.025, .895, r"\textbf{(b)}", transform=ax[1].transAxes, va="top",
               color=PlotColor.WHITE)

    ax[0].set_xlim(lo, hi)
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
