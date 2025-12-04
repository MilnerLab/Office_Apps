from typing import List
import numpy as np

from Lab_apps._domain.models import C2TData, LoadableScanData
from Lab_apps.scan_averaging.domain.models import AveragedScansData


from typing import List
from pathlib import Path

import numpy as np


def average_scans(scans: List[LoadableScanData]) -> AveragedScansData:
    """
    Compute the weighted average of several scans and the corresponding
    standard deviation, taking the y-errors into account.

    Assumptions:
    - delay (x-axis) is identical for all scans.
    - C2TData.error are 1σ uncertainties of each y-value.
    """
    if not scans:
        raise ValueError("Scan list is empty.")

    # --- Check that all scans share the same x-axis -------------------------
    delay_scan_0 = scans[0].delay      # list[Time]
    for s in scans[1:]:
        if s.delay != delay_scan_0:
            raise ValueError("All scans must have the same delay axis.")

    x0 = delay_scan_0                  # list[Time]

    # --- Stack arrays: shape (n_scans, n_points) ---------------------------
    y = np.stack(
        [[c.value for c in s.c2t] for s in scans],
        axis=0,
    )
    sigma = np.stack(
        [[c.error for c in s.c2t] for s in scans],
        axis=0,
    )

    # Weights from errors: w = 1 / sigma^2
    w = 1.0 / sigma**2

    # Weighted mean for each x point
    sum_w = np.sum(w, axis=0)
    avg_y = np.sum(w * y, axis=0) / sum_w

    # "Internal" error of the mean from the measurement errors
    sigma_int = np.sqrt(1.0 / sum_w)

    # --- Include additional scatter between scans, if present --------------
    n = len(scans)
    if n > 1:
        # Chi-square of deviations
        chi2 = np.sum(w * (y - avg_y) ** 2, axis=0)
        dof = n - 1
        chi2_red = chi2 / dof

        # If real scatter is larger than expected (chi2_red > 1),
        # scale the error accordingly (PDG prescription).
        scale = np.sqrt(np.maximum(1.0, chi2_red))
        sigma_tot = sigma_int * scale
    else:
        sigma_tot = sigma_int

    # --- Create C2TData-Object ----------------------------------
    avg_c2t: list[C2TData] = [
        C2TData(value=float(v), error=float(e))
        for v, e in zip(avg_y, sigma_tot)
    ]

    return AveragedScansData(
        delay=x0.copy(),                       # list[Time]
        c2t=avg_c2t,                          # list[C2TData]
        file_names=[s.file_path for s in scans],  # list[Path]
    )
