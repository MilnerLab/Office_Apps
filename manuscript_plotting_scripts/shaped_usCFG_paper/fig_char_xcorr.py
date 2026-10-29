"""Figs. 4 and 5 merged into one single-column figure (Ian's review, 2026-10).

The full and truncated cross-correlation of one accelerating centrifuge
(fig_char_xcorr_demo.draw), with the two truncation edges as an inset
(fig_char_xcorr_truncation.draw, black and red lines). The time axis runs past the end of
the traces to leave the upper right empty for the inset, rather than squeezing the traces
vertically. This script only places the two panels.

    python -m manuscript_plotting_scripts.shaped_usCFG_paper.fig_char_xcorr [--flip]

``--flip`` draws the full trace red and solid instead (Ian's colour suggestion) and saves
fig_char_xcorr_swapped.
"""
import sys

import matplotlib.pyplot as plt

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper import fig_char_xcorr_demo as demo
from manuscript_plotting_scripts.shaped_usCFG_paper import fig_char_xcorr_truncation as trunc

NAME = "fig_char_xcorr"
X_LIM = (-300, 400)                   # ps; the traces end at +301 ps, the inset sits beyond
INSET = (0.715, 0.45, 0.255, 0.51)    # axes fraction: x0, y0, width, height


def main(flip: bool = False) -> None:
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZES_IN[NAME], layout="constrained")
    demo.draw(ax, flip=flip)
    ax.set_xlim(*X_LIM)
    b = ax.inset_axes(INSET)
    trunc.draw(b, lines=True)
    b.set_xlim(-20, 20)
    b.set_xticks([-20, 0, 20])
    b.set_yticks([0, 1])
    b.tick_params(labelsize=7, pad=1.5)
    for a in (b.xaxis, b.yaxis):
        a.label.set_fontsize(7)
    b.xaxis.labelpad = 1
    b.yaxis.labelpad = 1
    config.save_figure(fig, NAME + ("_swapped" if flip else ""))


if __name__ == "__main__":
    main(flip="--flip" in sys.argv[1:])
