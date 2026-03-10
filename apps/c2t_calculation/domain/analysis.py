
import math
from pathlib import Path

import numpy as np
from base_core.lab_specifics.base_models import C2TScanData, IonDataAnalysisConfig, Measurement, RawScanData
from base_core.math.models import Points, Range
from base_core.quantities.models import Time


def run_pipeline(
    raw_datas: list[RawScanData],
    configs: list[IonDataAnalysisConfig]
) -> list[C2TScanData]:
    if len(raw_datas) != len(configs):
        raise ValueError("No distinct assignment possible.")

    return [
        C2TScanData.from_raw(raw, cfg)
        for raw, cfg in zip(raw_datas, configs, strict=True)
    ]


