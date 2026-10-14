"""Plot spectrum recordings stacked as maps: frequency on x, time on y, counts as colour.

Reads the layout ``SpectrumRecorder`` writes (``wavelength_nm`` + ``traces/<time_ns>``),
either at the root of a ``SPEC_*.h5`` file or in a subgroup of a bigger file (``GROUP``,
found automatically when there is only one). Rig-state changes from ``metadata/`` are
drawn as dashed lines labelled with their reason.

Edit the settings below, then run
``python manuscript_plotting_scripts/soeren_paper/plot_spectrum_recording.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from data_paths import OFFLINE_ROOT

#: Speed of light, nm/ps: nu (THz) = C_NM_PER_PS / lambda (nm).
C_NM_PER_PS = 299792.458

# -- settings -------------------------------------------------------------
_DRIFT = OFFLINE_ROOT / "thermal drift spectra"
#: Name -> recording, stacked top to bottom as (a), (b), ... on one colour scale and one
#: frequency axis. The names are only used in the console output.
FILES: dict[str, Path] = {
    "stabilized": _DRIFT / "SPEC_log_30ms_1avg_stabilized2_20260921_153251.h5",
    "unstabilized": _DRIFT / "SPEC_log_30ms_1avg_unstabilized_20260921_135954.h5",
}
GROUP: str | None = None                  # HDF5 group holding the recording; None = auto
WL_RANGE: tuple[float, float] | None = (799, 810)  # nm; None = full range
LOG_SCALE = False
SHOW_METADATA = True                      # dashed lines at rig-state changes
SAVE: Path | None = None                  # write the figure here instead of showing it
MAX_ROWS = 1500                           # time rows plotted; longer recordings are block-averaged


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
                n=len(keys), block=block)


def main() -> None:
    recs = {label: load_recording(path) for label, path in FILES.items()}

    # Cut every recording to the shortest one's duration, so the panels share one time axis.
    t_end = min(r["t_s"][-1] for r in recs.values())
    for r in recs.values():
        keep = r["t_s"] <= t_end
        r["t_s"], r["counts"] = r["t_s"][keep], r["counts"][keep]
    print(f"time axis cut to {t_end / 60:.1f} min")

    # One colour scale for all panels, so the maps compare directly.
    lo = min(r["counts"].min() for r in recs.values())
    hi = max(r["counts"].max() for r in recs.values())
    if LOG_SCALE:
        pos = [r["counts"][r["counts"] > 0] for r in recs.values()]
        lo = min((p.min() for p in pos if p.size), default=1)
        norm = LogNorm(vmin=lo, vmax=max(hi, 1))
    else:
        norm = plt.Normalize(vmin=lo, vmax=hi)

    # Two journal columns (17.8 cm), fonts at final print size.
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "pdf.fonttype": 42})
    fig, axs = plt.subplots(len(recs), 1, figsize=(17.8 / 2.54, (2.0 + 3.2 * len(recs)) / 2.54),
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
        ax.text(0.01, 0.95, f"({chr(ord('a') + i)})", transform=ax.transAxes, color="w",
                fontweight="bold", va="top", ha="left")
        print(f"({chr(ord('a') + i)}) {label}: {r['n']} spectra over {t_min[-1]:.1f} min"
              + (f", mean of {r['block']} per row" if r["block"] > 1 else ""))
    axs[-1].set_xlabel("Frequency (THz)")

    if SAVE:
        fig.savefig(SAVE, dpi=300)  # also the resolution of the rasterized maps in a PDF
        print(f"Saved {SAVE}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
