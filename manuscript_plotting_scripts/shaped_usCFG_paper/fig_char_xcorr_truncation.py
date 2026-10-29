"""The two truncated cross-correlations.

Panel (b) of fig_char_truncation on its own: the two truncated cross-correlations,
normalised and drawn against the delay from their fitted cut, prism at 12.86 mm 
and 14.50 mm. The erfc fits set the cut but are not drawn. The marks are named in the caption, so the
figure carries no legend.

Data: Z:\Droplets\shaped_usCFG_paper\truncation (xcorr\Scan8/Scan9 truncated
cross-correlations), reduced by domain/truncation.py.
"""
import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import truncation

NAME = "fig_char_xcorr_truncation"
C_EDGE = {12.86: "indigo", 14.50: PlotColor.BLUE}   #it uses these colours for scans with the corresponding prism positions
C_EDGE_L = {12.86: "-", 14.50: ":"}   #it uses these linestyles for scans with the corresponding prism positions


def draw(b, ms: float = 1.9, lines: bool = False) -> None:
    """The two edges on ``b``, each against the delay from its fitted cut. No error bars, as
    in the main panel of fig_char_xcorr, which draws this as its inset with ``lines``."""
    k = truncation.calibrate()
    for E in k["edges"]:
        col = C_EDGE[round(E.prism.value(Prefix.MILLI), 2)]
        lst = C_EDGE_L[round(E.prism.value(Prefix.MILLI), 2)]
        F = E.fit
        t = np.array([d.value(Prefix.PICO) for d in E.delays])
        y = np.array([m.value for m in E.measured_values])
        tc, lo, hi = F.tc.value(Prefix.PICO), F.post, F.pre
        style = dict(ls=lst, lw=0.8, marker="") if lines else dict(ls="", marker="o", ms=ms, mew=0)
        b.plot(t - tc, (y - lo) / (hi - lo), color=col, **style)
    b.set_xlim(-22, 22)
    b.set_ylim(-0.08, 1.45)
    b.set_xlabel("Time (ps)")
    b.set_ylabel("Intensity")


def main() -> None:
    fig, b = plt.subplots(figsize=config.FIGURE_SIZES_IN[NAME], layout="constrained")
    draw(b)
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
