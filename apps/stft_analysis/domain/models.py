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


    def detrend_moving_average(self, window: int = 10) -> list[float]:
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
        return (y - baseline).tolist()
        

@dataclass(frozen=True)
class SpectrogramBase:
    delay: list[Time]
    frequency: list[Frequency]
    power: list[list[float]]
    
    
@dataclass(frozen=True)
class SpectrogramResult(SpectrogramBase):
    run_id: int

@dataclass(frozen=True)
class AggregateSpectrogram(SpectrogramBase):
    run_ids: list[int]
