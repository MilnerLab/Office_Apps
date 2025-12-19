from typing import List
from base_core.quantities.models import Time
import numpy as np



from typing import List
from pathlib import Path

import numpy as np

from _domain.models import C2TData, LoadableScanData
from apps.scan_averaging.domain.models import AveragedScansData


from typing import List
import numpy as np

def average_scans(scans: List[LoadableScanData], *, key_digits: int = 12) -> AveragedScansData:
    """
    Average scans with possibly different x-axes.

    - Build a union x-axis containing all x values that appear in any scan.
    - Map each scan onto that axis into matrices (n_scans, n_x_union),
      filling missing entries with np.nan.
    - y-average: np.nanmean over scans.
    - error-average (ignoring scatter): sqrt(sum(sigma^2)) / N  (over available errors).

    key_digits: rounding precision for matching Time values robustly.
    """
    if not scans:
        raise ValueError("Scan list is empty.")

    # --- helpers ------------------------------------------------------------
    def _key(t: Time) -> float:
        # robust matching: treat Time as float-like (or fall back to .value)
        try:
            v = float(t)
        except TypeError:
            v = float(getattr(t, "value"))
        return round(v, key_digits)

    # --- build union x-axis (keep first seen Time object for each key) ------
    key_to_time: dict[float, Time] = {}
    for s in scans:
        for t in s.delay:
            k = _key(t)
            if k not in key_to_time:
                key_to_time[k] = t

    union_keys = sorted(key_to_time.keys())
    x_union: list[Time] = [key_to_time[k] for k in union_keys]
    key_to_idx = {k: i for i, k in enumerate(union_keys)}

    # --- allocate matrices --------------------------------------------------
    n_scans = len(scans)
    n_x = len(x_union)
    y_mat = np.full((n_scans, n_x), np.nan, dtype=float)
    s_mat = np.full((n_scans, n_x), np.nan, dtype=float)

    # --- map scans onto union axis -----------------------------------------
    for i, s in enumerate(scans):
        if len(s.delay) != len(s.c2t):
            raise ValueError(f"Scan {i} has mismatched delay/c2t lengths.")

        for t, c in zip(s.delay, s.c2t):
            j = key_to_idx[_key(t)]
            y_mat[i, j] = float(c.value)
            s_mat[i, j] = float(c.error)

    # --- average y using nanmean -------------------------------------------
    avg_y = np.nanmean(y_mat, axis=0)

    # --- average errors as requested: sqrt(sum(sigma^2)) / N ---------------
    sum_sq = np.nansum(s_mat ** 2, axis=0)
    count = np.sum(~np.isnan(s_mat), axis=0).astype(float)

    avg_sigma = np.full(n_x, np.nan, dtype=float)
    mask = count > 0
    avg_sigma[mask] = np.sqrt(sum_sq[mask]) / count[mask]

    # --- build output -------------------------------------------------------
    avg_c2t: list[C2TData] = [
        C2TData(value=float(v), error=float(e))
        for v, e in zip(avg_y, avg_sigma)
    ]

    return AveragedScansData(
        delay=x_union.copy(),
        c2t=avg_c2t,
        file_names=[s.file_path for s in scans],
    )
