from dataclasses import dataclass
from pathlib import Path
from base_core.math.models import Range
from base_core.quantities.models import Frequency, Time
import numpy as np

from _domain.models import  ScanDataBase

@dataclass(frozen=True)
class ResampledScan(ScanDataBase):
    file_path: Path
    scan_range: Range[Time]

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
