
import csv
from dataclasses import dataclass
import math
from pathlib import Path


from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from base_core.math.models import Point, Range, Points
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
    points: Points
    c2t: Measurement | None = None

    def calculate_avg_c2t(self) -> None:
        n = len(self.points)
        if n == 0:
            raise ValueError("No ions in data.")

        theta = np.arctan2(self.points.y, self.points.x)
        c2 = np.cos(theta) ** 2

        mean = float(np.mean(c2))
        if n > 1:
            std = float(np.std(c2, ddof=1))
            sem = float(std / math.sqrt(n)) if np.isfinite(std) else np.nan
        else:
            std = np.nan
            sem = np.nan

        self.c2t = Measurement(mean, sem)

    def get_2D_histogram(
        self,
        xy_range: Range[float] = Range(0.0, 400.0),
        num_bins: int = 400,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self.points) == 0:
            raise ValueError("IonData.points is empty.")

        r = ((float(xy_range.min), float(xy_range.max)),
             (float(xy_range.min), float(xy_range.max)))

        H, x_edges, y_edges = np.histogram2d(self.points.x, self.points.y, bins=num_bins, range=r)
        return H, x_edges, y_edges
   

@dataclass
class RawScanData:
    ion_datas: list[IonData]
    config: IonDataAnalysisConfig

    def apply_config(self) -> None:
        for d in self.ion_datas:
            pts = d.points
            pts.subtract(self.config.center)
            pts.affine_transform(self.config.transform_parameter)
            pts.rotate(self.config.angle)

            # keep only points inside analysis radius range
            d.points = pts.filter_by_distance_range(self.config.analysis_zone)
        