"""Truncation calibration.

(a) The shaper arm's spectrum against the truncation prism position (grey scale), the
linear fit to the cut wavelength (red line), and on the right axis the terminal frequency
f_trunc each cut wavelength implies, ticked over the calibrated range only.
(b) The two truncated cross-correlations, normalised and drawn against the delay from
their fitted cut, prism at 12.86 mm (orange) and 14.50 mm (green), with their erfc fits.
The marks are named in the caption, so the figure carries no legend.

Data: Z:\\Droplets\\shaped_usCFG_paper\\truncation (spectrometer\\<x>.csv, the two
xcorr\\Scan8/Scan9 truncated cross-correlations and
XCORR_20260903_jet_accompany_scan.h5), reduced by domain/truncation.py.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import truncation

NAME = "fig_char_truncation"
BAND_NM = (788.0, 815.0)                 # wavelength range of the image
C_EDGE = {12.86: "tab:orange", 14.50: PlotColor.GREEN}   # caption: orange, green
DPI = 300                                # the spectra image is resampled at this


def main() -> None:
    k = truncation.calibrate()
    l = np.array([v.value(Prefix.NANO) for v in k["l"]])
    files, spec = k["files"], k["spec"]
    lam_of_x = k["lam_of_x"].numpy
    f_of_u, u_cut = k["f_of_u"].numpy, k["u_cut"].numpy
    x_ok = np.array([r.prism.value(Prefix.MILLI) for r, ok in zip(k["rows"], k["ok"]) if ok])

    fig, ax = plt.subplots(2, 1, figsize=config.FIGURE_SIZES_IN[NAME], dpi=DPI,
                           height_ratios=[1.15, 1], layout="constrained")

    # (a) the spectra as an image, wavelength against prism position, with the calibration line
    xs = np.array([float(os.path.basename(f)[:-4]) for f in files])
    keep = (l > BAND_NM[0]) & (l < BAND_NM[1])
    img = np.array([spec(f)[keep] for f in files])
    a = ax[0]
    a.imshow(img.T, aspect="auto", origin="lower", cmap="Greys",
             extent=[xs.min(), xs.max(), l[keep].min(), l[keep].max()],
             vmin=0, vmax=np.percentile(img, 99.5), interpolation="hanning")
    xx = np.linspace(x_ok.min(), x_ok.max(), 2)
    a.plot(xx, lam_of_x(xx), ls="-", marker="", lw=0.8, color=PlotColor.RED)
    a.grid(False)
    a.set_xlabel("prism position (mm)")
    a.set_ylabel("wavelength (nm)")
    a.set_ylim(l[keep].min(), l[keep].max())
    a.text(.025, .96, r"\textbf{(a)}", transform=a.transAxes, va="top")
    # right axis: f_trunc at the cut wavelength, ticked over the calibrated range only
    a2 = a.twinx()
    a2.grid(False)
    a2.set_ylim(a.get_ylim())
    xr = np.linspace(x_ok.min(), x_ok.max(), 400)
    lr, fr = lam_of_x(xr), f_of_u(u_cut(xr))
    o = np.argsort(fr)
    fticks = [f for f in range(25, 400, 25) if fr.min() <= f <= fr.max()]
    a2.set_yticks(np.interp(fticks, fr[o], lr[o]))
    a2.set_yticklabels([str(f) for f in fticks])
    a2.set_ylabel(r"$f_{\rm trunc}$ (GHz)", color=PlotColor.BLUE)
    a2.tick_params(axis="y", colors=PlotColor.BLUE)

    # (b) the two measured edges on a common axis (delay from the fitted cut), with erfc fits
    b = ax[1]
    tq = np.linspace(-25, 25, 2000)
    for E in k["edges"]:
        col = C_EDGE[round(E.prism.value(Prefix.MILLI), 2)]
        F = E.fit
        t = np.array([d.value(Prefix.PICO) for d in E.delays])
        y = np.array([m.value for m in E.measured_values])
        e = np.array([m.error for m in E.measured_values])
        tc, lo, hi = F.tc.value(Prefix.PICO), F.post, F.pre
        b.errorbar(t - tc, (y - lo) / (hi - lo), e / (hi - lo), fmt="o", ms=1.9, mew=0,
                   color=col, ecolor=col, capsize=0)
        b.plot(tq, (F.model.numpy(tq + tc) - lo) / (hi - lo), ls="-", marker="", lw=0.7,
               color=col)
    b.set_xlim(-22, 22)
    b.set_ylim(-0.08, 1.45)
    b.set_xlabel("delay from the cut (ps)")
    b.set_ylabel("cross-correlation (norm.)")
    b.text(.025, .96, r"\textbf{(b)}", transform=b.transAxes, va="top")
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
