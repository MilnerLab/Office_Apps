"""Slow-centrifuge spectrum against optical frequency, with the phase-stabilization fit.

The fit is the same one ``spectrum_analyzer_detached.panel_spec_cfg`` runs (``analyze_trace``,
done in wavelength); data and fit are then put on nu = c / lambda.

Usage:
    python manuscript_plotting_scripts/soeren_paper/single_spectrum.py [--spec F] [--window-nm LO HI] [--fit-thz LO HI] [--save PDF]
        [--data {auto,online,offline}]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_paths import MODES, data_root
from spectrum_analyzer_detached import C_NM_PER_PS, DATA_GROUP, SPEC_CFG, load_spectrum
from spectrum_fit import fringe_core as fc
from spectrum_fit.stabilization_fit import FitTunables, analyze_trace, display_curve


def nm_to_thz(wl_nm):
    """Wavelength (nm) -> optical frequency (THz): c / lambda with c in nm/ps."""
    return C_NM_PER_PS / np.asarray(wl_nm, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default=str(SPEC_CFG))
    ap.add_argument("--window-nm", nargs=2, type=float, default=list(fc.ZOOM),
                    help="wavelength range plotted")
    ap.add_argument("--fit-thz", nargs=2, type=float, default=[372.0, 376.7],
                    help="frequency range the fit is drawn over (it is fitted on the whole window)")
    ap.add_argument("--save", default=str(Path(__file__).with_suffix(".pdf")),
                    help="PDF written before the figure is shown")
    ap.add_argument("--data", choices=MODES, default=None,
                    help="online (Valery share), offline (local copy) or auto; "
                         "default: $SOEREN_DATA, else auto")
    args = ap.parse_args(argv)
    # A relative --spec (the default) is a day folder under the online/offline root.
    args.spec = str(data_root(DATA_GROUP, args.data) / args.spec)

    import matplotlib.pyplot as plt

    # One journal column (8.6 cm), fonts at final print size.
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 8, "pdf.fonttype": 42})

    wl_full, i_full = load_spectrum(args.spec)
    m = (wl_full >= args.window_nm[0]) & (wl_full <= args.window_nm[1])
    wl, inten = wl_full[m], i_full[m]
    # Fit in raw counts over the whole window; only the display is cut to --fit-thz
    # (nu = c / lambda, so the ends swap).
    r = analyze_trace(wl, inten, FitTunables(), anchor=fc.baseline_anchor(wl_full, i_full))
    fit_nm = sorted(nm_to_thz(args.fit_thz))

    # Normalise data and fit with the same map: baseline (5th percentile) -> 0, peak -> 1.
    base = np.percentile(inten, 5)
    scale = inten.max() - base

    def norm(y):
        return (y - base) / scale

    fig, ax = plt.subplots(figsize=(8.6 / 2.54, 6.0 / 2.54))
    ax.plot(nm_to_thz(wl), norm(inten), color="0.5", lw=0.5, label="spectrum")
    if r.accepted:
        xx = np.linspace(fit_nm[0], fit_nm[1], 4000)
        mid, half, phase = display_curve(r, xx)
        ax.plot(nm_to_thz(xx), norm(mid + half * np.cos(phase)), "C3", lw=1, label="fit")
        print(f"fit: status={r.status}  l0 = {r.l0:.3f} nm ({nm_to_thz(r.l0):.3f} THz)  "
              f"rms_frac={r.rms_frac:.3f}")
    else:
        print(f"fit rejected: {r.status} {r.msg}")
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel("normalised intensity")
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(args.save)
    print(f"saved {args.save}")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
