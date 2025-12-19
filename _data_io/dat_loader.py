# io/dat_loader.py
from pathlib import Path
from base_core.math.models import Point
from base_core.quantities.constants import SPEED_OF_LIGHT
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
import numpy as np

from _domain.config import DELAY_STAGE_CENTER_VALUE
from _domain.models import C2TData, IonData, LoadableScanData



def load_time_scan(path: Path) -> LoadableScanData:

    delays: list[Time] = []
    c2ts: list[C2TData] = []
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
                c2t = C2TData(float(parts[2]), float(parts[3]))
                ion_count = float(parts[4])
            except ValueError:
                continue

            delays.append(delay)
            c2ts.append(c2t)
            ions.append(ion_count)

    if not delays:
        raise ValueError("No valid data lines (>= 4 numeric columns) found in file.")

    file_name = path

    return LoadableScanData(delay=delays, c2t=c2ts, file_path=file_name, ions_per_frame=ions)


def load_time_scans(paths: list[Path]) -> list[LoadableScanData]:
    
    scanDatas: list[LoadableScanData] = []
    
    for path in paths:
        scanDatas.append(load_time_scan(path=path))

    return scanDatas


def load_ion_data(paths: list[Path]) -> list[IonData]:

    output: list[IonData] = []

    for path in sorted(paths):
        
        run_id, delay = extrect_infos_from_name(path)

        arr = np.loadtxt(path, usecols=(1, 2))  
        points = [Point(float(x), float(y)) for x, y in arr]

        output.append(IonData(run_id, delay, points))

    return output



###########
#  Helper functions
###########


def extrect_infos_from_name(path: Path) -> tuple[int, Time]:
    stem = Path(path).stem
    time_part, stage_part = stem.split("DLY_", 1)
    
    stage_part = stage_part.replace("_", "")
    stage_part = stage_part[:-2]
    stage_part = stage_part.replace("p", ".")

    delay = calculate_time_delay(Length(float(stage_part), Prefix.MILLI))

    return int(time_part), delay


def calculate_time_delay(stage_position: Length) -> Time:
    delta = (stage_position - DELAY_STAGE_CENTER_VALUE) * 2

    return Time(delta / SPEED_OF_LIGHT)
