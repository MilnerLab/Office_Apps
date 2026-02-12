# domain/config.py
from dataclasses import dataclass

from base_core.quantities.models import Time
import numpy as np

from _domain.models import C2TScanData, ScanDataBase

@dataclass
class StftAnalysisConfig:
    resample_time: Time 
    stft_window_size: Time
    axis: list[Time]
    
    def __init__(self, scan_data: list[ScanDataBase], stft_window_size: Time | None = None) -> None:
        self.axis = []
        self.set_from_data(scan_data)
        
        if stft_window_size is not None:
            self.stft_window_size = stft_window_size

    def set_from_data(self, scan_data: list[ScanDataBase]) -> None:
        starts = [scan.delays[0] for scan in scan_data]
        ends = [scan.delays[-1] for scan in scan_data]
        min_delay_spacing = min(
                b - a
                for scan in scan_data
                if len(scan.delays) > 1
                for a, b in zip(scan.delays, scan.delays[1:])
            )

        self.resample_time = Time(min_delay_spacing)
        
        current_delay = min(starts)
        while(current_delay <= max(ends)):
            self.axis.append(Time(current_delay))
            current_delay += self.resample_time
        
        self.stft_window_size = min(np.array(ends) - np.array(starts)) / 4
        
