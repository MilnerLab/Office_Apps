from dataclasses import dataclass
from pathlib import Path
from base_core.math.models import Range
from base_core.quantities.models import Frequency, Time
import numpy as np

from _domain.models import  C2TData, ScanDataBase

@dataclass(frozen=True)
class ResampledScan(ScanDataBase):
    file_path: Path
    scan_range: Range[Time]
    
    def detrend(self)->list[float]:
        c2t = np.asarray([c.value for c in self.c2t])
        avg = np.nanmean(c2t)
        new_c2t: list[C2TData] = []
        [new_c2t.append(v) for v in c2t - avg]
        return new_c2t
        

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
