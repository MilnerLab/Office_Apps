"""Figs. 4 and 5 merged into one double-wide figure (Ian's review, 2026-10).

The full and truncated cross-correlation of one accelerating centrifuge across the full
width (fig_char_xcorr_demo.draw), with the two truncation edges as a narrow inset in its
empty upper right (fig_char_xcorr_truncation.draw). This script only places the two panels.

    python -m manuscript_plotting_scripts.shaped_usCFG_paper.fig_char_xcorr [--flip]

``--flip`` draws the full trace red and solid instead (Ian's colour suggestion) and saves
fig_char_xcorr_swapped, which the paper stacks under the unswapped one.
"""
import sys

import matplotlib.pyplot as plt

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper import fig_char_xcorr_demo as demo
from manuscript_plotting_scripts.shaped_usCFG_paper import fig_char_xcorr_truncation as trunc

NAME = "fig_char_xcorr"
Y_TOP = 1.5                           # headroom above the full trace for the inset
INSET = (0.775, 0.585, 0.215, 0.40)   # axes fraction: x0, y0, width, height


def main(flip: bool = False) -> None:
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZES_IN[NAME], layout="constrained")
    demo.draw(ax, flip=flip)
    ax.set_ylim(0, Y_TOP)
    b = ax.inset_axes(INSET)
    trunc.draw(b, ms=1.3)
    b.set_xlim(-20, 20)
    b.set_xticks([-20, 0, 20])
    b.tick_params(labelsize=7)
    b.xaxis.label.set_fontsize(8)
    b.set_ylabel("")                  # same quantity as the main axis; the caption gives the normalization
    config.save_figure(fig, NAME + ("_swapped" if flip else ""))


if __name__ == "__main__":
    main(flip="--flip" in sys.argv[1:])
