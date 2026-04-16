# io/dat_loader.py
from pathlib import Path

from astropy import conf
from base_core.framework.services import runnable_service_base
from base_core.lab_specifics.base_models import C2TScanData, IonData, IonDataAnalysisConfig, Measurement, RawScanData, ScanDataBase, calculate_time_delay
from base_core.math.models import MarkedPoints, Points
from base_core.quantities.constants import SPEED_OF_LIGHT
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
import numpy as np
import pandas as pd
import time




def load_time_scan(path: Path, config: IonDataAnalysisConfig) -> C2TScanData:

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

    return C2TScanData(delays=delays, measured_values=c2ts, run_id=None, ions_per_frame=ions, config=config)


def load_time_scans(paths: list[Path], config: IonDataAnalysisConfig) -> list[C2TScanData]:
    
    scanDatas: list[C2TScanData] = []
    
    for path in paths:
        scanDatas.append(load_time_scan(path=path, config=config))

    return scanDatas


def load_ion_data(scans_paths: list[list[Path]]) -> list[RawScanData]:
    raw_scans: list[RawScanData] = []

    for scan_group in scans_paths:
        x_chunks_by_pos: dict[Length, list[np.ndarray]] = {}
        y_chunks_by_pos: dict[Length, list[np.ndarray]] = {}
        frame_chunks_by_pos: dict[Length, list[np.ndarray]] = {}
        run_id_by_pos: dict[Length, int] = {}
        next_frame_offset_by_pos: dict[Length, int] = {}

        number_of_scans = 0

        for path in sorted(scan_group):
            run_id, stage_position = extract_infos_from_name(path)

            arr = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2, dtype=np.float64)

            local_frames = arr[:, 2].astype(np.int64)
            xs = arr[:, 0]
            ys = arr[:, 1]

            if stage_position not in x_chunks_by_pos:
                x_chunks_by_pos[stage_position] = []
                y_chunks_by_pos[stage_position] = []
                frame_chunks_by_pos[stage_position] = []
                run_id_by_pos[stage_position] = run_id
                next_frame_offset_by_pos[stage_position] = 0

            offset = next_frame_offset_by_pos[stage_position]
            frames = local_frames + offset

            x_chunks_by_pos[stage_position].append(xs)
            y_chunks_by_pos[stage_position].append(ys)
            frame_chunks_by_pos[stage_position].append(frames)

            if frames.size > 0:
                next_frame_offset_by_pos[stage_position] = int(frames.max()) + 1

            number_of_scans += 1

        output: list[IonData] = []
        for stage_position in sorted(x_chunks_by_pos.keys()):
            xs = np.concatenate(x_chunks_by_pos[stage_position])
            ys = np.concatenate(y_chunks_by_pos[stage_position])
            frames = np.concatenate(frame_chunks_by_pos[stage_position])

            hits = MarkedPoints(xs, ys, frames)

            output.append(
                IonData(
                    id=run_id_by_pos[stage_position],
                    stage_position=stage_position,
                    ions_per_frame=hits.avg_points_per_marker(),
                    points=hits,
                )
            )

        raw_scans.append(
            RawScanData(
                run_id=output[0].id,
                ion_datas=output,
                number_of_scans=number_of_scans,
            )
        )

    return raw_scans

def load_xcorr_means(file_path:Path,pos_tzero:Length) -> ScanDataBase:
    ScopeData = np.array(pd.read_csv(file_path,header=None,sep='\t',lineterminator='\n',dtype=float))
    delay = [calculate_time_delay(Length(d,Prefix.MILLI),pos_tzero) for d in ScopeData[:,0]]
    signal = np.average(ScopeData[:,1:-1],axis=1)
    error = np.std(ScopeData[:,1:-1],axis=1)/np.sqrt(ScopeData.shape[1] - 1)

    values = [Measurement(signal[i], error[i]) for i in range(len(signal))]
    
    return ScanDataBase(run_id=0,delays = delay, measured_values = values)

###########
#  Helper functions
###########


def extract_infos_from_name(path: Path) -> tuple[int, Length]:
    stem = Path(path).stem
    time_part, stage_part = stem.split("DLY_", 1)
    
    stage_part = stage_part.replace("_", "")
    stage_part = stage_part[:-2]
    stage_part = stage_part.replace("p", ".")

    return int(time_part), Length(float(stage_part), Prefix.MILLI)

