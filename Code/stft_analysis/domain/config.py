# domain/config.py
from dataclasses import dataclass

import numpy as np

from Lab_apps._base.models import Range, Time
from Lab_apps._domain.models import LoadableScanData

@dataclass
class AnalysisConfig:
    resample_time: Time 
    stft_window_size: Time
    axis: list[Time]
    
    def __init__(self, scan_data: list[LoadableScanData], stft_window_size: Time | None = None) -> None:
        self.axis = []
        self.set_from_data(scan_data)
        
        if stft_window_size is not None:
            self.stft_window_size = stft_window_size

    def set_from_data(self, scan_data: list[LoadableScanData]) -> None:
        starts = [scan.delay[0] for scan in scan_data]
        ends = [scan.delay[-1] for scan in scan_data]
        
        furthest_scan = scan_data[ends.index(max(ends))]
        self.resample_time = Time(furthest_scan.delay[-1] - furthest_scan.delay[-2])
        
        current_delay = min(starts)
        while(current_delay <= max(ends)):
            self.axis.append(Time(current_delay))
            current_delay += self.resample_time
        
        self.stft_window_size = min(np.array(ends) - np.array(starts)) / 4