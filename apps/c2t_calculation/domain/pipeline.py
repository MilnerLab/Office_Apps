
from pathlib import Path
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from _domain.models import Measurement, IonData, C2TScanData, RawScanData
from base_core.quantities.models import Time


def run_pipeline(raw_datas: list[RawScanData], save_path: Path = None) -> list[C2TScanData]:
    scans: list[C2TScanData] = []
    
    for raw_data in raw_datas:
        
        delays: list[Time] = []
        c2t: list[Measurement] = []

        raw_data.apply_config()
        
        for data in raw_data.ion_datas:
            data.calculate_avg_c2t()
            delays.append(data.delay)

            assert data.c2t is not None
            c2t.append(data.c2t)
        
        scans.append(C2TScanData(delays, c2t, save_path))
        
    return scans

