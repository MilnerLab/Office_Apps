from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft

from apps.scan_averaging.domain.averaging import average_scans
from apps.stft_analysis.domain.resampling import resample_scan
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
        
        if not self.can_compute_spectrogram():
            self.scans = [resample_scan(average_scans(self.scans), self.config.axis)]

    def can_compute_spectrogram(self) -> bool:
        nperseg, _, _ = self._stft_params()

        for scan in self.scans:
            c2t = np.asarray(scan.detrend(), dtype=float)
            n_valid = int(np.count_nonzero(np.isfinite(c2t)))
            if n_valid < nperseg:
                return False

        return True

    def _stft_params(self) -> tuple[int, int, float]:
        """
        Internal helper to compute STFT parameters.
        Uses config.axis length as the (resampled) number of points.

        Returns:
            (nperseg, noverlap, fs)
        """
        num_points = len(self.config.axis)

        nperseg = min(
            int(num_points / BACKUP_WINDFRACT),
            int(round(self.config.stft_window_size / self.config.resample_time)),
        )
        nperseg = max(1, nperseg)

        # enforce odd window length
        if nperseg % 2 == 0:
            nperseg = max(1, nperseg - 1)

        noverlap = nperseg - 1 if nperseg > 1 else 0
        fs = 1 / self.config.resample_time
        return nperseg, noverlap, fs

    def calculate_spectrogram(self, resampled_scan: ResampledScan) -> SpectrogramResult:
        c2t = resampled_scan.detrend()
        nperseg, noverlap, fs = self._stft_params()

        f, t_s, Zxx = stft(
            c2t,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window="blackman",
        )

        power = (np.abs(Zxx)) ** 2
        t_s = t_s + resampled_scan.delays[0]

        delay: list[Time] = [Time(val) for val in t_s]
        frequency: list[Frequency] = [Frequency(val) for val in f]

        return SpectrogramResult(
            delay=delay,
            frequency=frequency,
            power=power,
            file_path=resampled_scan.file_path,
        )

    def calculate_averaged_spectrogram(self) -> AggregateSpectrogram:
        specs: list[SpectrogramResult] = [self.calculate_spectrogram(scan) for scan in self.scans]

        base_freq = np.asarray(specs[0].frequency, dtype=float)
        n_freq = int(base_freq.size)

        for s in specs[1:]:
            f_i = np.asarray(s.frequency, dtype=float)
            if f_i.shape != base_freq.shape or not np.allclose(f_i, base_freq):
                raise ValueError(
                    "Frequency axes of individual spectrograms differ; "
                    "cannot average without interpolation."
                )

        global_time = np.asarray(self.config.axis, dtype=float)
        n_time_global = int(global_time.size)

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
