
from dataclasses import dataclass
from pathlib import Path

from base_core.math.models import Point
from base_core.quantities.models import Time
import numpy as np



@dataclass(frozen=True)
class C2TData:
    value: float
    error: float

@dataclass(frozen=True)
class ScanDataBase:
    delay: list[Time]
    c2t: list[C2TData]

@dataclass(frozen=True)
class LoadableScanData(ScanDataBase):
    file_path: Path
    ions_per_frame: list[float] | None = None


@dataclass
class IonData:
    run_id: int
    delay: Time
    points: list[Point]
    c2t: C2TData | None = None

    def calculate_avg_c2t(self):
        x0 = [point.x for point in self.points]
        y0 = [point.y for point in self.points]

        N = int(x0.__len__())
        if N == 0:
            raise ValueError("No ions in data.")

        theta = np.arctan2(y0, x0)     # Angle to horizontal line through (0,0)
        c2 = np.cos(theta) ** 2
        mean = float(np.mean(c2))
        std  = float(np.std(c2, ddof=1)) if N > 1 else np.nan
        sem  = float(std / np.sqrt(N)) if (N > 1 and np.isfinite(std)) else np.nan
        self.c2t = C2TData(mean, sem)

    