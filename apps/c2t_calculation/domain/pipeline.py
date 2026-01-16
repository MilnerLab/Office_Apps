
from pathlib import Path
from apps.c2t_calculation.domain.config import AnalysisConfig
from _domain.models import C2TData, IonData, LoadableScanData
from base_core.quantities.models import Time


def run_pipeline(ion_data: list[IonData], config: AnalysisConfig, save_path: Path) -> LoadableScanData:
    delays: list[Time] = []
    c2t: list[C2TData] = []

    for data in ion_data:
        data.apply_config(config)
        data.calculate_avg_c2t()

        delays.append(data.delay)

        assert data.c2t is not None
        c2t.append(data.c2t)
    
    return LoadableScanData(delays, c2t, save_path)

