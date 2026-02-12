from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft

from base_core.quantities.models import Frequency, Time
from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.models import (
    AggregateSpectrogram,
    ResampledScan,
    SpectrogramResult,
)

BACKUP_WINDFRACT = 2


@dataclass(slots=True)
class StftAnalysis:
    scans: list[ResampledScan]
    config: StftAnalysisConfig

    def __post_init__(self) -> None:
        if not self.scans:
            raise ValueError("scans must not be empty")

    def get_spectrogram(self) -> AggregateSpectrogram:
        """Public API: averaged spectrogram over all scans (no parameters)."""
        return self.calculate_averaged_spectrogram()

    def calculate_spectrogram(self, resampled_scan: ResampledScan) -> SpectrogramResult:
        """Compute one spectrogram for one scan."""
        c2t = resampled_scan.detrend()

        samplesperseg = min(
            int(len(c2t) / BACKUP_WINDFRACT),
            int(round(self.config.stft_window_size / self.config.resample_time)),
        )
        samplesperseg = max(1, samplesperseg)

        # enforce odd window length
        if samplesperseg % 2 == 0:
            samplesperseg = max(1, samplesperseg - 1)

        numoverlap = samplesperseg - 1 if samplesperseg > 1 else 0
        fs = 1 / self.config.resample_time

        f, t_s, Zxx = stft(
            c2t,
            fs=fs,
            nperseg=samplesperseg,
            noverlap=numoverlap,
            window="blackman",
        )

        power = (np.abs(Zxx)) ** 2
        t_s = t_s + resampled_scan.delays[0]  # time-zero back

        delay: list[Time] = [Time(val) for val in t_s]
        frequency: list[Frequency] = [Frequency(val) for val in f]

        return SpectrogramResult(
            delay=delay,
            frequency=frequency,
            power=power,
            file_path=resampled_scan.file_path,
        )

    def calculate_averaged_spectrogram(self) -> AggregateSpectrogram:
        """Average spectrogram over all scans in self.scans."""
        specs: list[SpectrogramResult] = [self.calculate_spectrogram(scan) for scan in self.scans]

        base_freq = np.asarray(specs[0].frequency, dtype=float)
        n_freq = base_freq.size

        for s in specs[1:]:
            f_i = np.asarray(s.frequency, dtype=float)
            if f_i.shape != base_freq.shape or not np.allclose(f_i, base_freq):
                raise ValueError(
                    "Frequency axes of individual spectrograms differ; "
                    "cannot average without interpolation."
                )

        global_time = np.asarray(self.config.axis, dtype=float)
        n_time_global = global_time.size

        cube = np.full((len(specs), n_freq, n_time_global), np.nan, dtype=float)

        for i, s in enumerate(specs):
            P = np.asarray(s.power, dtype=float)
            if P.shape != (n_freq, n_time_global):
                raise ValueError(
                    f"Spectrogram shape mismatch for scan #{i}: got {P.shape}, "
                    f"expected {(n_freq, n_time_global)}. "
                    "Make sure config.axis matches the STFT time axis."
                )
            cube[i, :, :] = P

        avg_power = np.nanmean(cube, axis=0)
        max_val = float(np.nanmax(avg_power))
        if max_val > 0:
            avg_power = avg_power / max_val

        delay_times: list[Time] = [Time(t) for t in global_time]
        freq_objs: list[Frequency] = [Frequency(f) for f in base_freq]
        file_paths: list[Path] = [scan.file_path for scan in self.scans]

        return AggregateSpectrogram(
            delay=delay_times,
            frequency=freq_objs,
            power=avg_power.tolist(),
            file_paths=file_paths,
        )
