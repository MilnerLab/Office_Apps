
from pathlib import Path
from Lab_apps._base.models import Time
from Lab_apps._domain.models import C2TData, IonData, LoadableScanData
from Lab_apps.c2t_calculation.domain.config import AnalysisConfig


def run_pipeline(ion_data: IonData, config: AnalysisConfig) -> IonData:
    
    for point in ion_data.points:
        point.subtract(config.center)
        point.affine_transform(config.transform_parameter)
        point.rotate(config.angle)
    ion_data.points = [p for p in ion_data.points if config.analysis_zone.is_in_range(p.distance_from_center())]
    ion_data.calculate_avg_c2t()

    if ion_data.c2t is None:
        raise ValueError("C2T should not be None.")

    return ion_data


def run_pipeline_for_multiple(ion_data: list[IonData], config: AnalysisConfig, save_path: Path) -> LoadableScanData:
    delays: list[Time] = []
    c2t: list[C2TData] = []

    for data in ion_data:
       processed_data = run_pipeline(data, config)

       delays.append(processed_data.delay)

       assert processed_data.c2t is not None
       c2t.append(processed_data.c2t)
    
    return LoadableScanData(delays, c2t, save_path)

