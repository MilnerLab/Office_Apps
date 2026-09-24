"""Release at three truncation positions.

<cos^2 theta_2D> against probe delay, centrifuge phase mechanically averaged, read along
the fitted alignment axis, with frame-level error bars: the full centrifuge (black
circles) and the truncation prism at 15.0 mm (green squares), 14.5 mm (red diamonds) and
13.5 mm (blue triangles). The traces are named in the caption, so the figure carries no
legend.

Data: Z:\\Droplets\\shaped_usCFG_paper\\Jet\\Jet Truncation (20260904\\Scan3_CFG,
Scan4_CFG and 20260907\\Scan1_CFG, Scan2_CFG; raw VMI hit lists), reduced by
domain/jet.py.
"""
import matplotlib.pyplot as plt
import numpy as np

from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import jet

NAME = "fig_jet_truncation"
# in jet.TRUNCATION_SCANS order: full centrifuge, prism 15.0, 14.5, 13.5 mm
STYLES = [(PlotColor.BLACK, "o"), (PlotColor.GREEN, "s"), (PlotColor.RED, "D"),
          (PlotColor.BLUE, "^")]


def main() -> None:
    _, traces = jet.truncation_traces()

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZES_IN[NAME], layout="constrained")
    for tr, (col, mk) in zip(traces, STYLES):
        t = np.array([d.value(Prefix.PICO) for d in tr.delays])
        y = np.array([m.value for m in tr.measured_values])
        e = np.array([m.error for m in tr.measured_values])
        ax.errorbar(t, y, yerr=e, fmt="-", marker=mk, ms=2.4, mew=0, color=col, ecolor=col)
    ax.axhline(0.5, ls=":", marker="", color=PlotColor.GRAY)
    ax.set_xlabel("probe delay (ps)")
    ax.set_ylabel(r"$\langle\cos^2\theta_{\rm 2D}\rangle$")
    ax.set_xlim(-520, 1570)
    ax.set_ylim(0.4975, 0.5625)
    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
