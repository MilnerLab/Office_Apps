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
from matplotlib.patches import Rectangle

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper import fig_char_xcorr_demo as demo
from manuscript_plotting_scripts.shaped_usCFG_paper import fig_char_xcorr_truncation as trunc

NAME = "fig_char_xcorr"
X_LIM = (-270, 400)                 # ps; the traces end at +301 ps, the inset sits beyond
INSET_W, INSET_H = 0.23, 0.33       # axes fraction
INSET_GAP_PT = 6.0                  # inset frame to the main frame's right edge and to its 1.0 grid line
INSET_TOP_GAP_FRACTION = -0.5       # if you want the gap to the top (1.0 line) to be different from the gap to the right 

def inset_bounds(fig, ax) -> tuple[float, float, float, float]:
    """Inset (x0, y0, w, h) in axes fraction, its top right corner INSET_GAP_PT from the
    main frame's right edge and below the intensity = 1 grid line."""
    fig.set_size_inches(*config.FIGURE_SIZES_IN[NAME])
    fig.canvas.draw()                                   # settle the constrained layout first
    bb, gap = ax.get_window_extent(), INSET_GAP_PT * fig.dpi / 72
    right = 1 - gap / bb.width
    top = ax.transAxes.inverted().transform(ax.transData.transform((0, 1.0)))[1] - gap*INSET_TOP_GAP_FRACTION / bb.height
    return right - INSET_W, top - INSET_H, INSET_W, INSET_H


def main(flip: bool = False) -> None:
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZES_IN[NAME], layout="constrained")
    demo.draw(ax, flip=flip)
    ax.set_xlim(*X_LIM)
    b = ax.inset_axes(inset_bounds(fig, ax))
    trunc.draw(b, lines=True)
    b.set_xlim(-20, 20)
    b.set_xticks([-20, 0, 20])
    b.set_yticks([0, 1])
    b.tick_params(labelsize=7, pad=1.5)
    for a in (b.xaxis, b.yaxis):
        a.label.set_fontsize(7)
    b.xaxis.labelpad = 1
    b.yaxis.labelpad = 1
    mask_grid_behind(fig, ax, b)
    config.save_figure(fig, NAME + ("_swapped" if flip else ""))


def mask_grid_behind(fig, ax, b, pad_pt: float = 1.0) -> None:
    """White patch under the inset and its labels: above the main grid (zorder 0.5 with
    axisbelow), below the traces (zorder 2), so it clears the grid but never hides data."""
    fig.set_size_inches(*config.FIGURE_SIZES_IN[NAME])
    fig.canvas.draw()                                   # settle the constrained layout first
    bb = b.get_tightbbox(fig.canvas.get_renderer()).padded(pad_pt * fig.dpi / 72)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(bb.get_points())
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transAxes,
                           facecolor="white", edgecolor="none", zorder=1, clip_on=False))


if __name__ == "__main__":
    main(flip="--flip" in sys.argv[1:])
