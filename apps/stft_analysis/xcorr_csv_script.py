"""
STFT (frequency vs probe-delay) analysis for a cross-correlation CSV.

Adapts a raw XCORR csv (delay-in-mm + N waveform columns per row) into the
`stft_analysis` pipeline, which was otherwise built around ion/VMI .dat data.

CSV format expected (tab-separated):
    col 0      : stage position in mm
    cols 1..N  : N intensity samples (waveforms) for that delay point

The stage position (mm) is converted to probe delay (time) via the usual
double-pass retroreflector relation  t = 2 * (pos - center) / c.
"""

import sys
from pathlib import Path

# Allow running this file directly (python xcorr_csv_script.py) by putting the
# project root on sys.path; running as a module (-m) already handles this.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.plotting import plot_Spectrogram, plot_nyquist_frequency
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.domain.stft_calculation import StftAnalysis
from base_core.lab_specifics.base_models import Measurement, ScanDataBase
from base_core.lab_specifics.helpers import calculate_time_delay
from base_core.math.models import Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time


# --- User config ------------------------------------------------------------
CSV_PATH = Path(
    r"D:\Documents\University\UBC research\2026\Data\20260718\XCORR\DA19.0_GA-75.0\202607181045_.csv"
)

# Stage position (mm) that corresponds to probe delay t = 0 (pump/probe overlap).
# Only shifts the absolute time axis; it does NOT affect the extracted frequencies.
DELAY_CENTER_MM = 135.0

# STFT window length in time. Larger -> better frequency resolution, worse time
# resolution. Should be a fraction of the total scan span.
WINDOWSIZE = Time(180, Prefix.PICO)

# Optional crop of the probe-delay axis (ps). Set to None to use full range.
DELAY_CROP_PS: tuple[float, float] | None = None

# Upper limit of the spectrogram frequency axis (GHz). None -> full Nyquist
# (fs/2, set by the finest delay spacing). The shared plot_Spectrogram caps the
# axis at 150 GHz by default; this overrides it.
FREQ_YLIM_GHZ: float | None = 250
# ---------------------------------------------------------------------------


# Light rcParams (no LaTeX dependency, unlike the main stft script).
mpl.rcParams.update({
    "axes.grid": True,
    "grid.linewidth": 0.3,
    "grid.color": "grey",
    "lines.linewidth": 0.8,
})


def load_xcorr_csv(path: Path, delay_center_mm: float, run_id: int = 0) -> ScanDataBase:
    """Read an XCORR csv into a ScanDataBase (delay-time vs mean intensity)."""
    # Rows may be jagged (a partial/interrupted acquisition leaves a short final
    # row); keep only rows that have the full, most-common column count.
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    width = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    data = np.array([r for r in rows if len(r) == width], dtype=float)

    stage_mm = data[:, 0]
    waveforms = data[:, 1:]

    mean_intensity = waveforms.mean(axis=1)
    sem_intensity = waveforms.std(axis=1, ddof=1) / np.sqrt(waveforms.shape[1])

    center = Length(delay_center_mm, Prefix.MILLI)
    delays = [
        calculate_time_delay(Length(float(mm), Prefix.MILLI), center)
        for mm in stage_mm
    ]
    measured = [Measurement(float(v), float(e)) for v, e in zip(mean_intensity, sem_intensity)]

    return ScanDataBase(delays=delays, measured_values=measured, run_id=run_id)


def main() -> None:
    scan = load_xcorr_csv(CSV_PATH, DELAY_CENTER_MM)

    stft_config = StftAnalysisConfig([scan], WINDOWSIZE)
    resampled_scans = resample_scans([scan], stft_config.axis)
    spectrogram = StftAnalysis(resampled_scans, stft_config).calculate_averaged_spectrogram()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"hspace": 0}
    )

    # Top: the cross-correlation trace (intensity vs probe delay).
    delay_ps = [d.value(Prefix.PICO) for d in scan.delays]
    intensity = [m.value for m in scan.measured_values]
    ax1.plot(delay_ps, intensity, color="tab:blue", marker="", label="Cross-correlation")
    ax1.set_ylabel("Intensity (arb.)")
    ax1.legend(loc="upper right")
    ax1.tick_params(axis="x", labelbottom=False)

    # Bottom: the spectrogram (frequency vs probe delay).
    plot_Spectrogram(ax2, spectrogram, v_range=Range(0, 1), shading="auto")
    plot_nyquist_frequency(ax2, scan)
    ax2.set_axisbelow(False)

    # Override plot_Spectrogram's hard-coded 150 GHz cap. Default: full Nyquist,
    # i.e. the highest frequency the STFT actually computed.
    nyquist_ghz = spectrogram.frequency[-1].value(Prefix.GIGA)
    ax2.set_ylim(0, FREQ_YLIM_GHZ if FREQ_YLIM_GHZ is not None else nyquist_ghz)

    fig.suptitle(f"XCORR STFT: {CSV_PATH.parent.name}/{CSV_PATH.name}", fontsize=11)
    fig.subplots_adjust(top=0.93, left=0.09, right=0.93, bottom=0.12)
    plt.show()


if __name__ == "__main__":
    main()
