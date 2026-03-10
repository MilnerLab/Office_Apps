
import math
from pathlib import Path

import numpy as np
from base_core.lab_specifics.base_models import C2TScanData, IonDataAnalysisConfig, Measurement, RawScanData
from base_core.math.models import Points, Range
from base_core.quantities.models import Time


def run_pipeline(
    raw_datas: list[RawScanData],
    configs: list[IonDataAnalysisConfig],
    save_path: Path | None = None,
) -> list[C2TScanData]:
    if len(raw_datas) != len(configs):
        raise ValueError("No distinct assignment possible.")

    return [
        C2TScanData.from_raw(raw, cfg, save_path=save_path)
        for raw, cfg in zip(raw_datas, configs, strict=True)
    ]

def avg_c2t(points: Points) -> Measurement:
    """
    Compute <cos^2(theta)> and its SEM for a set of 2D points, where
    theta = arctan2(y, x).

    Returns Measurement(mean, sem).
    """
    n = len(points)
    if n == 0:
        raise ValueError("No ions in data.")

    theta = np.arctan2(points.y, points.x)
    c2 = np.cos(theta) ** 2

    mean = float(np.mean(c2))
    std = float(np.std(c2, ddof=1)) if n > 1 else np.nan
    sem = float(std / math.sqrt(n)) if (n > 1 and np.isfinite(std)) else np.nan
    return Measurement(mean, sem)
