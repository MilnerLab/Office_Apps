from dataclasses import dataclass
from pathlib import Path
from base_core.fitting.functions import fit_gaussian
from base_core.math.functions import gaussian
from base_core.math.models import Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Time
import numpy as np

from _domain.models import  C2TData, ScanDataBase

@dataclass(frozen=True)
class ResampledScan(ScanDataBase):
    file_path: Path
    scan_range: Range[Time]
    
    
    def detrend(self) -> list[float]:
        y = np.asarray([c.value for c in self.c2t], dtype=float)
        t = np.asarray([d.value(Prefix.PICO) for d in self.delay], dtype=float)

        x = t - t[0]

        fit = fit_gaussian(x, y)
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
