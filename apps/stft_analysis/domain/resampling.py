# domain/resampling.py
from base_core.lab_specifics.base_models import Measurement, ScanDataBase
from base_core.math.models import Range
from base_core.quantities.models import Time
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

from apps.stft_analysis.domain.models import ResampledScan

def resample_scan(raw: ScanDataBase, axis: list[Time]) -> ResampledScan:
    x = raw.delays
    y = [c.value for c in raw.measured_values]

    mask_measured = list((np.array(axis) >= x[0]) & (np.array(axis) <= x[-1]))
    resample_axis = np.where(mask_measured, axis, np.nan)
    
    #cs = CubicSpline(x, y, extrapolate=True)
    cs = make_interp_spline(x,y,1)
    y_res = cs(resample_axis)
    new_c2t = [Measurement(y, 0) for y in y_res]

    return ResampledScan(
        delays=axis.copy(),
        measured_values=new_c2t,
        run_id=raw.run_id,
        scan_range=Range(min(resample_axis), max(resample_axis))
    )

def resample_scans(scans: list[ScanDataBase], axis: list[Time]) -> list[ResampledScan]:
    if not isinstance(scans, list):
        scans = [scans]
    resampled_scans: list[ResampledScan] = []
    
    for scan in scans:
        resampled_scans.append(resample_scan(scan, axis))

    return resampled_scans