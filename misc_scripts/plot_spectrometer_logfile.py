"""Plots a SPEC_log*.h5 spectrometer recording as a spectrogram (intensity vs wavelength vs time)."""
import json
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

H5_PATH1 = r"Z:\Droplets\20260921\SPEC_log_30ms_1avg_stabilized2_20260921_153251 - Copy.h5"
H5_PATH2= r"Z:\Droplets\20260921\SPEC_log_30ms_1avg_unstabilized_20260921_135954 - Copy.h5"

CMAP = "viridis"
MAX_ROWS = 2000  # downsample the time axis so the image stays a manageable size
WAVELENGTH_RANGE_NM = (795, 812)
MEAN_PERCENTILE_CUTOFF = 80  # keep spectra whose mean intensity falls within [100-CUTOFF, CUTOFF] percentile of all spectra


def load_spectrogram(path: str, max_rows: int = MAX_ROWS):
    with h5py.File(path, "r") as f:
        info = json.loads(f["info"][()].decode("utf-8"))
        wavelength_nm = np.asarray(f["wavelength_nm"][...], dtype=np.float64)

        trace_group = f["traces"]
        keys = sorted(trace_group.keys(), key=int)
        t0 = int(keys[0])
        n = len(keys)

        spectra_all = np.empty((n, wavelength_nm.size), dtype=np.float32)
        times_min_all = np.empty(n, dtype=np.float64)
        for i, k in enumerate(keys):
            spectra_all[i] = trace_group[k][...]
            times_min_all[i] = (int(k) - t0) / 1e9 / 60

    trace_means = spectra_all.mean(axis=1)
    lo_threshold, hi_threshold = np.percentile(trace_means, [100 - MEAN_PERCENTILE_CUTOFF, MEAN_PERCENTILE_CUTOFF])
    keep = (trace_means >= lo_threshold) & (trace_means <= hi_threshold)
    spectra_all = spectra_all[keep]
    times_min_all = times_min_all[keep]

    mean_spectrum = spectra_all.mean(axis=0)
    min_spectrum = spectra_all.min(axis=0)
    max_spectrum = spectra_all.max(axis=0)
    typical_spectrum = spectra_all[spectra_all.shape[0] // 2]

    n_kept = spectra_all.shape[0]
    stride = max(1, n_kept // max_rows)
    spectra = spectra_all[::stride]
    times_min = times_min_all[::stride]

    return info, wavelength_nm, times_min, spectra, mean_spectrum, min_spectrum, max_spectrum, typical_spectrum, n_kept


def plot_column(
    ax, ax_avg, info, wavelength_nm, times_min, spectra, mean_spectrum, min_spectrum, max_spectrum, typical_spectrum, n_kept
):
    lo, hi = WAVELENGTH_RANGE_NM
    outside_mask = (wavelength_nm < lo) | (wavelength_nm > hi)
    baseline = mean_spectrum[outside_mask].mean()

    mask = (wavelength_nm >= lo) & (wavelength_nm <= hi)
    wavelength_nm_crop = wavelength_nm[mask]
    spectra_crop = spectra[:, mask]
    mean_spectrum_crop = mean_spectrum[mask]
    min_spectrum_crop = min_spectrum[mask]
    max_spectrum_crop = max_spectrum[mask]
    typical_spectrum_crop = typical_spectrum[mask]

    extent = [wavelength_nm_crop[0], wavelength_nm_crop[-1], times_min[-1], times_min[0]]
    ax.imshow(spectra_crop, aspect="auto", extent=extent, cmap=CMAP, interpolation="nearest")

    ax.set_ylabel("Time (min)")
    ax.set_title(
        f"{info.get('source', '')}, {n_kept} traces\n"
        f"({100 - MEAN_PERCENTILE_CUTOFF}-{MEAN_PERCENTILE_CUTOFF}th percentile by mean intensity)"
    )
    ax.grid(True, color="white", alpha=0.3, linewidth=0.5)

    ax_avg.plot(wavelength_nm_crop, max_spectrum_crop, color="0.6", lw=0.8, label="Max")
    ax_avg.plot(wavelength_nm_crop, min_spectrum_crop, color="0.6", lw=0.8, label="Min")
    ax_avg.plot(wavelength_nm_crop, mean_spectrum_crop, color="black", lw=1, label="Mean")
    ax_avg.plot(wavelength_nm_crop, typical_spectrum_crop, color="tab:blue", lw=0.8, label="Typical")
    ax_avg.axhline(baseline, color="red", lw=1, label="Baseline (outside range)")
    ax_avg.set_xlabel("Wavelength (nm)")
    ax_avg.set_ylabel("Intensity")
    ax_avg.grid(True, alpha=0.3, linewidth=0.5)
    ax_avg.legend(fontsize=8, loc="upper right")


def main() -> None:
    if len(sys.argv) > 2:
        path1, path2 = sys.argv[1], sys.argv[2]
        out_path = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        path1, path2 = H5_PATH1, H5_PATH2
        out_path = None

    data1 = load_spectrogram(path1)
    data2 = load_spectrogram(path2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), height_ratios=[3, 1], constrained_layout=True)
    plot_column(axes[0, 0], axes[1, 0], *data1)
    plot_column(axes[0, 1], axes[1, 1], *data2)

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
