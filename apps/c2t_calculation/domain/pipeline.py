
from pathlib import Path
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from _domain.models import C2TData, IonData, LoadableScanData, RawScanData
from base_core.quantities.models import Time


def run_pipeline(raw_datas: list[RawScanData], save_path: Path) -> list[LoadableScanData]:
    scans: list[LoadableScanData] = []
    
    for raw_data in raw_datas:
        
        delays: list[Time] = []
        c2t: list[C2TData] = []

        raw_data.apply_config()
        
        for data in raw_data.ion_datas:
            data.calculate_avg_c2t()
            delays.append(data.delay)

            assert data.c2t is not None
            c2t.append(data.c2t)
        
        scans.append(LoadableScanData(delays, c2t, save_path))
        
    return scans

