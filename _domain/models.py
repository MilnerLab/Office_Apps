
import csv
from dataclasses import dataclass
import math
from pathlib import Path

from altair import DerivedStream
from git import Tree

from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from base_core.math.models import Point, Range
from base_core.quantities.models import Time
import numpy as np



@dataclass(frozen=True)
class Measurement:
    value: float
    error: float

@dataclass(frozen=True)
class ScanDataBase:
    delays: list[Time]
    measured_values: list[Measurement]
    
    def to_csv(self, path: str | Path) -> None:
        if path == None:
            print('No path specified, did not save data.')
        else:
            with Path(path).open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for d, m in zip(self.delays, self.measured_values):
                    w.writerow([d, m.value, m.error])
            print("Data saved to:",path)
            
    def cut(self, start: int = 0, end: int = 0) -> None:
        n = len(self.delays)
        if len(self.measured_values) != n or start < 0 or end < 0 or start + end > n:
            raise ValueError("Invalid cut range.")

        if end:
            del self.delays[-end:]
            del self.measured_values[-end:]
        if start:
            del self.delays[:start]
            del self.measured_values[:start]

@dataclass(frozen=True)
class LoadableScan(ScanDataBase):
    file_path: Path = None
    
@dataclass(frozen=True)
class C2TScanData(LoadableScan):
    ions_per_frame: list[float] | None = None

@dataclass
class IonData:
    run_id: int
    delay: Time
    points: list[Point]
    c2t: Measurement | None = None

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
        self.c2t = Measurement(mean, sem)
        
    def get_2D_histogram(
        self,
        xy_range: Range[float] = Range(0.0, 400.0),
        num_bins: int = 400,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.points:
            raise ValueError("IonData.points is empty.")

        xs = np.array([p.x for p in self.points], dtype=float)
        ys = np.array([p.y for p in self.points], dtype=float)

        r = ((xy_range.min, xy_range.max), (xy_range.min, xy_range.max))
        H, x_edges, y_edges = np.histogram2d(xs, ys, bins=num_bins, range=r)
        return H, x_edges, y_edges
   

@dataclass
class RawScanData:
    ion_datas: list[IonData]
    config: IonDataAnalysisConfig 

    def apply_config(self):
        for d in self.ion_datas:
            for point in d.points:
                point.subtract(self.config.center)
                point.affine_transform(self.config.transform_parameter)
                point.rotate(self.config.angle)
            d.points = [p for p in d.points if self.config.analysis_zone.is_in_range(p.distance_from_center())]
        