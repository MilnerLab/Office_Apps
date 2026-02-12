from dataclasses import dataclass
from pathlib import Path
from base_core.fitting.functions import fit_gaussian
from base_core.math.functions import gaussian
from base_core.math.models import Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Time
import numpy as np

from _domain.models import  Measurement, ScanDataBase

@dataclass(frozen=True)
class ResampledScan(ScanDataBase):
    scan_range: Range[Time]
    file_path: Path = None
    
    
    def detrend(self) -> list[float]:
        y = np.asarray([c.value for c in self.measured_values], dtype=float)
        t = np.asarray([d.value(Prefix.PICO) for d in self.delays], dtype=float)

        x = t - t[0]

        mask = np.isfinite(y) & np.isfinite(x)
        if mask.sum() < 4:  # je nach Fit mindestens ein paar Punkte nötig
            return (y - np.nan).tolist()  # oder: return y.tolist()

        fit = fit_gaussian(x[mask], y[mask])
        g = gaussian(x, fit.amplitude, fit.center, fit.sigma, fit.offset)

        return (y - g).tolist()

        

@dataclass(frozen=True)
class SpectrogramBase:
    delay: list[Time]
    frequency: list[Frequency]
    power: list[list[float]]
    
    
@dataclass(frozen=True)
class SpectrogramResult(SpectrogramBase):
    file_path: Path

@dataclass(frozen=True)
class AggregateSpectrogram(SpectrogramBase):
    file_paths: list[Path]
