from dataclasses import dataclass
from pathlib import Path
from base_core.fitting.functions import fit_gaussian
from base_core.framework import di
from base_core.lab_specifics.base_models import ScanDataBase
from base_core.math.functions import gaussian
from base_core.math.models import Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Time
import numpy as np



@dataclass(frozen=True)
class ResampledScan(ScanDataBase):
    scan_range: Range[Time] = None
    
    def detrend(self) -> tuple[list[float], list[float]]:
        """
        Gaussian detrend: fit Gaussian baseline g(x) to finite points and return (y-g, g).
        NaNs are ignored for the fit and preserved in the output.
        """
        y = np.asarray([c.value for c in self.measured_values], dtype=float)
        t = np.asarray([d.value(Prefix.PICO) for d in self.delays], dtype=float)
        x = t - t[0]

        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 4:  # für Gaussian-Fit typischerweise zu wenig
            trend = np.full_like(y, np.nan, dtype=float)
            return (y.tolist(), trend.tolist())

        fit = fit_gaussian(x[mask], y[mask])
        g = gaussian(x, fit.amplitude, fit.center, fit.sigma, fit.offset)

        return ((y - g).tolist(), g.tolist())


    def detrend_moving_average(self, window: int = 10) -> tuple[list[float], list[float]]:
        y = np.asarray([c.value for c in self.measured_values], dtype=float)

        mask = np.isfinite(y).astype(float)
        if mask.sum() == 0:
            return y.tolist()

        w = int(window)
        w = max(1, min(w, int(mask.sum())))
        if w % 2 == 0 and w > 1:
            w -= 1

        kernel = np.ones(w, dtype=float)
        pad = w // 2

        y0 = np.nan_to_num(y, nan=0.0)

        y0p = np.pad(y0, (pad, pad), mode="edge")
        mp  = np.pad(mask, (pad, pad), mode="edge")

        num = np.convolve(y0p, kernel, mode="valid")
        den = np.convolve(mp,  kernel, mode="valid")

        baseline = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        return ((y - baseline).tolist(), baseline.tolist())
        

@dataclass(frozen=True)
class SpectrogramBase:
    delay: list[Time]
    frequency: list[Frequency]
    power: list[list[float]]  # indexed as power[frequency][delay]

    def cut(self, start: int = 0, end: int = 0):
        """Index-based cut along the delay axis (same semantics as ScanDataBase.cut: end inclusive, 0 = no cut)."""
        n = len(self.delay)
        if start < 0 or end < 0:
            raise ValueError("Invalid cut range.")
        stop = end + 1 if end else n
        keep = [start <= i < stop for i in range(n)]
        return self._keep_delays(keep)

    def cut_time(self, t_min: Time | None = None, t_max: Time | None = None):
        """
        Keep only delay columns within [t_min, t_max] (inclusive).
        None means no threshold on that side. Plain floats are interpreted as seconds.
        """
        if t_min is not None and t_max is not None and t_min > t_max:
            raise ValueError("Invalid cut range: t_min > t_max.")
        keep = [
            (t_min is None or d >= t_min) and (t_max is None or d <= t_max)
            for d in self.delay
        ]
        return self._keep_delays(keep)

    def cut_frequency(self, f_min: Frequency | None = None, f_max: Frequency | None = None):
        """
        Keep only frequency rows within [f_min, f_max] (inclusive).
        None means no threshold on that side. Plain floats are interpreted as Hz.
        """
        if f_min is not None and f_max is not None and f_min > f_max:
            raise ValueError("Invalid cut range: f_min > f_max.")
        idx = [
            i for i, f in enumerate(self.frequency)
            if (f_min is None or f >= f_min) and (f_max is None or f <= f_max)
        ]

        new_frequency = [self.frequency[i] for i in idx]
        if isinstance(self.power, np.ndarray):
            new_power = self.power[idx, :]
        else:
            new_power = [self.power[i] for i in idx]

        object.__setattr__(self, "frequency", new_frequency)
        object.__setattr__(self, "power", new_power)
        return self

    def _keep_delays(self, keep: list[bool]):
        """Drop (in place) every delay column whose mask entry is False, from delay and power."""
        idx = [i for i, k in enumerate(keep) if k]

        new_delay = [self.delay[i] for i in idx]
        if isinstance(self.power, np.ndarray):
            new_power = self.power[:, idx]
        else:
            new_power = [[row[i] for i in idx] for row in self.power]

        # frozen dataclass: bypass the freeze, matching ScanDataBase's in-place cut behaviour
        object.__setattr__(self, "delay", new_delay)
        object.__setattr__(self, "power", new_power)
        return self


@dataclass(frozen=True)
class SpectrogramResult(SpectrogramBase):
    run_id: int

@dataclass(frozen=True)
class AggregateSpectrogram(SpectrogramBase):
    run_ids: list[int]
