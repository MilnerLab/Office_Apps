# io/dat_loader.py
from pathlib import Path
from apps.c2t_calculation.domain.config import IonDataAnalysisConfig
from base_core.math.models import Point
from base_core.quantities.constants import SPEED_OF_LIGHT
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
import numpy as np
import pandas as pd
import time

from _domain.models import Measurement, IonData, C2TScanData, RawScanData, ScanDataBase



def load_time_scan(path: Path) -> C2TScanData:

    delays: list[Time] = []
    c2ts: list[Measurement] = []
    ions: list[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                # leere Zeile
                continue

            parts = stripped.split()
            if len(parts) < 5:
                continue

            try:
                delay = Time(float(parts[1]), prefix=Prefix.PICO)
                c2t = Measurement(float(parts[2]), float(parts[3]))
                ion_count = float(parts[4])
            except ValueError:
                continue

            delays.append(delay)
            c2ts.append(c2t)
            ions.append(ion_count)

    if not delays:
        raise ValueError("No valid data lines (>= 4 numeric columns) found in file.")

    file_name = path

    return C2TScanData(delays=delays, measured_values=c2ts, file_path=file_name, ions_per_frame=ions)


def load_time_scans(paths: list[Path]) -> list[C2TScanData]:
    
    scanDatas: list[C2TScanData] = []
    
    for path in paths:
        scanDatas.append(load_time_scan(path=path))

    return scanDatas


def load_ion_data(scans_paths: list[list[Path]], configs: list[IonDataAnalysisConfig]) -> list[RawScanData]:
    
    if len(scans_paths) != len(configs):
        raise ValueError("No distinct assignment possible.")
    
    raw_scans: list[RawScanData] = []
    
    for i in range(len(scans_paths)):
        output: list[IonData] = []
        idx_by_delay: dict[Time, int] = {}  
        
        t0 = time.perf_counter()
        
        for path in sorted(scans_paths[i]):
            run_id, delay = extract_infos_from_name(path,configs[i].delay_center)

            arr = np.loadtxt(path, usecols=(1, 2), ndmin=2)
            points = [Point(x, y) for x, y in arr]

            if delay in idx_by_delay:
                output[idx_by_delay[delay]].points.extend(points)
            else:
                idx_by_delay[delay] = len(output)
                output.append(IonData(run_id, delay, points))
        
        t1 = time.perf_counter()
        print("loadtxt:", t1 - t0)
        
        output.sort(key=lambda x: x.delay)
        
        raw_scans.append(RawScanData(output, configs[i]))

    return raw_scans

def load_xcorr_means(file_path:Path,pos_tzero:Length) -> ScanDataBase:
    ScopeData = np.array(pd.read_csv(file_path,header=None,sep='\t',lineterminator='\n',dtype=float))
    delay = [calculate_time_delay(Length(d,Prefix.MILLI),pos_tzero) for d in ScopeData[:,0]]
    signal = np.average(ScopeData[:,1:-1],axis=1)
    error = np.std(ScopeData[:,1:-1],axis=1)/np.sqrt(ScopeData.shape[1] - 1)

    values = [Measurement(signal[i], error[i]) for i in range(len(signal))]
    return ScanDataBase(delays = delay, measured_values = values)

###########
#  Helper functions
###########


def extract_infos_from_name(path: Path, delay_center: Length) -> tuple[int, Time]:
    stem = Path(path).stem
    time_part, stage_part = stem.split("DLY_", 1)
    
    stage_part = stage_part.replace("_", "")
    stage_part = stage_part[:-2]
    stage_part = stage_part.replace("p", ".")

    delay = calculate_time_delay(Length(float(stage_part), Prefix.MILLI), delay_center)

    return int(time_part), delay


def calculate_time_delay(stage_position: Length, delay_center: Length) -> Time:
    delta = (stage_position - delay_center) * 2

    return Time(delta / SPEED_OF_LIGHT)
