"""Plot spectrum recordings stacked as maps: frequency on x, time on y, counts as colour.

Reads the layout ``SpectrumRecorder`` writes (``wavelength_nm`` + ``traces/<time_ns>``),
either at the root of a ``SPEC_*.h5`` file or in a subgroup of a bigger file (``GROUP``,
found automatically when there is only one). Rig-state changes from ``metadata/`` are
drawn as dashed lines labelled with their reason.

Each recording is also fitted with the phase-stabilization fit ``single_spectrum.py`` uses
(``analyze_trace``): ``N_FIT`` single spectra, evenly spaced over the plotted time range,
each give the constant phase at ``LAMBDA_REF``. Their circular mean, spread and standard
error are printed and written into the panel.

Edit the settings below, then run
``python manuscript_plotting_scripts/soeren_paper/plot_spectrum_recording.py``.
"""
from __future__ import annotations

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import OptimizeWarning

from data_paths import OFFLINE_ROOT
from spectrum_fit import fringe_core as fc
from spectrum_fit.stabilization_fit import FitTunables, analyze_trace

# analyze_trace logs a FITDIAG warning per fit; hundreds of them bury the summary.
logging.getLogger("spectrum_fit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=OptimizeWarning)

#: Speed of light, nm/ps: nu (THz) = C_NM_PER_PS / lambda (nm).
C_NM_PER_PS = 299792.458

# -- settings -------------------------------------------------------------
_DRIFT = OFFLINE_ROOT / "thermal drift spectra"
#: Name -> recording, stacked top to bottom as (a), (b), ... on one colour scale and one
#: frequency axis. The names are only used in the console output.
FILES: dict[str, Path] = {
    "unstabilized": _DRIFT / "SPEC_log_30ms_1avg_unstabilized_20260921_135954.h5",
    "stabilized": _DRIFT / "SPEC_log_30ms_1avg_stabilized2_20260921_153251.h5",
}
GROUP: str | None = None                  # HDF5 group holding the recording; None = auto
WL_RANGE: tuple[float, float] | None = (799, 810)  # nm; None = full range
LOG_SCALE = False
SHOW_METADATA = True                      # dashed lines at rig-state changes
SAVE: Path | None = Path(__file__).with_suffix(".pdf")  # written before the figure is shown
MAX_ROWS = 1500                           # time rows plotted; longer recordings are block-averaged
N_FIT: int | None = 400                   # spectra fitted per recording; None = all (~0.6 s each)
LAMBDA_REF = 802.0                        # nm; phase reported here (the lock's lambda_ref)
FIT_WORKERS: int | None = None            # fit processes; None = all cores
SHOW_PHASE = True                         # write mean phase +- standard error into each panel


def find_group(f: h5py.File, name: str | None) -> h5py.Group:
    if name:
        return f[name]
    if "traces" in f:
        return f
    found: list[h5py.Group] = []
    f.visititems(lambda _, obj: found.append(obj)
                 if isinstance(obj, h5py.Group) and "traces" in obj else None)
    if len(found) != 1:
        raise SystemExit(f"Found {len(found)} recording groups; set GROUP to one of: "
                         + ", ".join(g.name for g in found))
    return found[0]


def read_binned(traces: h5py.Group, keys: list[str], sel: np.ndarray, block: int):
    """Mean of every ``block`` consecutive traces (wavelengths ``sel`` only), with the mean
    time of each block. Streams one trace at a time, so memory stays at the binned size."""
    n_rows = -(-len(keys) // block)
    counts = np.zeros((n_rows, int(sel.sum())))
    t_ns = np.zeros(n_rows)
    for i in range(n_rows):
        chunk = keys[i * block:(i + 1) * block]
        for k in chunk:
            counts[i] += traces[k][()][sel]
        counts[i] /= len(chunk)
        t_ns[i] = np.mean([int(k) for k in chunk])
        if i % 100 == 0:
            print(f"\rreading {i * block}/{len(keys)} spectra", end="", flush=True)
    print(f"\rread {len(keys)} spectra into {n_rows} rows of {block}")
    return t_ns, counts


def load_recording(path: Path) -> dict:
    """One recording, block-averaged to <= MAX_ROWS rows; times in s from its first spectrum."""
    with h5py.File(path, "r") as f:
        g = find_group(f, GROUP)
        wl = g["wavelength_nm"][()]
        keys = sorted(g["traces"].keys())
        if not keys:
            raise SystemExit(f"No traces in {path}")
        t0 = int(keys[0])
        sel = np.ones(wl.size, bool)
        if WL_RANGE:
            sel = (wl >= min(WL_RANGE)) & (wl <= max(WL_RANGE))
        block = max(1, -(-len(keys) // MAX_ROWS))
        t_ns, counts = read_binned(g["traces"], keys, sel, block)
        meta = []
        if SHOW_METADATA and "metadata" in g:
            for k in sorted(g["metadata"].keys()):
                raw = g["metadata"][k][()]
                entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                meta.append(((int(k) - t0) / 1e9,
                             entry.get("reason", "") if isinstance(entry, dict) else ""))
    return dict(wl=wl[sel], t_s=(t_ns - t0) / 1e9, counts=counts, meta=meta,
                n=len(keys), block=block, path=path, keys=keys)


def fit_phase(wl: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Wrapped phase (rad) at LAMBDA_REF of one full spectrum and its 1-sigma, as in
    single_spectrum.py; None when the fit cannot support the phase there."""
    m = (wl >= fc.ZOOM[0]) & (wl <= fc.ZOOM[1])
    r = analyze_trace(wl[m], y[m], FitTunables(), anchor=fc.baseline_anchor(wl, y),
                      lambda_ref_nm=LAMBDA_REF)
    # ref_fallback: trust_ok then speaks for the core centroid, not for LAMBDA_REF.
    if not (r.accepted and r.trust_ok) or r.ref_fallback:
        return None
    return float(np.angle(np.exp(1j * r.phase_at(LAMBDA_REF)))), r.csig_sigma[0]


def fit_recording(r: dict, t_end_s: float) -> dict:
    """Fit N_FIT spectra evenly spaced over [0, t_end_s]; circular statistics of the phase.

    The phase wraps (the stabilized one sits near +-pi), so the mean is the angle of the mean
    phasor and the spread the circular standard deviation sqrt(-2 ln R)."""
    keys = r["keys"]
    t0 = int(keys[0])
    keys = [k for k in keys if (int(k) - t0) / 1e9 <= t_end_s]
    if N_FIT and len(keys) > N_FIT:
        keys = [keys[i] for i in np.linspace(0, len(keys) - 1, N_FIT).round().astype(int)]
    with h5py.File(r["path"], "r") as f:
        g = find_group(f, GROUP)
        wl = g["wavelength_nm"][()].astype(float)
        ys = [g["traces"][k][()].astype(float) for k in keys]
    with ProcessPoolExecutor(FIT_WORKERS) as ex:
        out = list(ex.map(partial(fit_phase, wl), ys, chunksize=4))
    good = [(int(k), *o) for k, o in zip(keys, out) if o is not None]
    if not good:
        return dict(n_fit=len(keys), n_ok=0)
    t_ns, phi, sig = map(np.array, zip(*good))
    z = np.exp(1j * phi).mean()
    std = float(np.sqrt(-2 * np.log(abs(z))))
    return dict(n_fit=len(keys), n_ok=len(phi), t_s=(t_ns - t0) / 1e9, phi=phi,
                mean=float(np.angle(z)), R=float(abs(z)), std=std, sem=std / np.sqrt(len(phi)),
                fit_sigma=float(np.median(sig)))


def main() -> None:
    recs = {label: load_recording(path) for label, path in FILES.items()}

    # Cut every recording to the shortest one's duration, so the panels share one time axis.
    t_end = min(r["t_s"][-1] for r in recs.values())
    for r in recs.values():
        keep = r["t_s"] <= t_end
        r["t_s"], r["counts"] = r["t_s"][keep], r["counts"][keep]
    print(f"time axis cut to {t_end / 60:.1f} min")

    for label, r in recs.items():
        print(f"fitting {label} ...", flush=True)
        r["fit"] = p = fit_recording(r, t_end)
        if not p["n_ok"]:
            print(f"  no usable fits out of {p['n_fit']}")
            continue
        print(f"  phase at {LAMBDA_REF:g} nm: mean {p['mean']:+.3f} rad, "
              f"standard error {p['sem']:.3f} rad, circular std {p['std']:.3f} rad "
              f"(R = {p['R']:.3f}); {p['n_ok']}/{p['n_fit']} fits trusted, "
              f"median fit sigma {p['fit_sigma']:.3f} rad")

    # One colour scale for all panels, so the maps compare directly.
    lo = min(r["counts"].min() for r in recs.values())
    hi = max(r["counts"].max() for r in recs.values())
    if LOG_SCALE:
        pos = [r["counts"][r["counts"] > 0] for r in recs.values()]
        lo = min((p.min() for p in pos if p.size), default=1)
        norm = LogNorm(vmin=lo, vmax=max(hi, 1))
    else:
        norm = plt.Normalize(vmin=lo, vmax=hi)

    # One journal column (8.6 cm), fonts at final print size.
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "pdf.fonttype": 42})
    fig, axs = plt.subplots(len(recs), 1, figsize=(8.6 / 2.54, (1.5 + 3.0 * len(recs)) / 2.54),
                            sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    axs = axs[:, 0]
    for i, (ax, (label, r)) in enumerate(zip(axs, recs.items())):
        counts = r["counts"]
        nu = C_NM_PER_PS / r["wl"]  # THz; descending, which pcolormesh accepts
        t_min = r["t_s"] / 60.0
        if LOG_SCALE:
            counts = np.where(counts > 0, counts, np.nan)
        # "nearest" handles uneven time steps (dropped frames, pauses) without resampling.
        ax.pcolormesh(nu, t_min, counts, shading="nearest", cmap="viridis", norm=norm,
                             rasterized=True)
        for y_s, reason in r["meta"]:
            y = y_s / 60.0
            if t_min[0] <= y <= t_min[-1]:
                ax.axhline(y, color="w", lw=0.8, ls="--", alpha=0.7)
                ax.text(nu.max(), y, f" {reason}", color="w", fontsize=8, va="bottom", ha="right")
        ax.set_xlim(nu.min(), nu.max())
        ax.set_ylabel("Time (min)")
        ax.text(0.02, 0.93, f"({chr(ord('a') + i)})", transform=ax.transAxes, color="k",
                fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="0.85", edgecolor="k", lw=0.6))
        if SHOW_PHASE and r["fit"]["n_ok"]:
            p = r["fit"]
            ax.text(0.98, 0.05, rf"$\varphi_0 = {p['mean']:.2f} \pm {p['sem']:.2f}$ rad",
                    transform=ax.transAxes, color="k", fontsize=8, va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="0.85", edgecolor="k", lw=0.6))
        print(f"({chr(ord('a') + i)}) {label}: {r['n']} spectra over {t_min[-1]:.1f} min"
              + (f", mean of {r['block']} per row" if r["block"] > 1 else ""))
    axs[-1].set_xlabel("Frequency (THz)")

    if SAVE:
        fig.savefig(SAVE, dpi=300)  # also the resolution of the rasterized maps in a PDF
        print(f"Saved {SAVE}")
    plt.show()


if __name__ == "__main__":
    main()
