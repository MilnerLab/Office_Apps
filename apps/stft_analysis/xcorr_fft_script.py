"""FFT (frequency spectrum along probe delay) of a cross-correlation CSV.

Companion to `xcorr_csv_script.py`: same CSV loader and config, but takes a
single FFT of the whole trace instead of an STFT.
"""

import sys
from pathlib import Path

# Allow running this file directly (python xcorr_fft_script.py) by putting the
# project root on sys.path; running as a module (-m) already handles this.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from base_core.lab_specifics.base_models import Measurement, ScanDataBase
from base_core.quantities.enums import Prefix

from apps.stft_analysis.domain.config import StftAnalysisConfig
from apps.stft_analysis.domain.resampling import resample_scans
from apps.stft_analysis.xcorr_csv_script import (
    calculate_xcorr_fft,
    load_xcorr_csv,
)

from scipy.fft import rfft, rfftfreq
from base_core.math.functions import gaussian

CSV_PATH_4f = Path(r"Z:\Droplets\20261015\XCORR\202610151201_CFG_4F_Flipped_Slow_ish.csv")
CSV_PATH_cf = Path(r"Z:\Droplets\20261015\XCORR\20261015132_CFG_cf_slow.csv")
DELAY_CENTER_MM = 120

# Probe-delay window (ps) the FFT is taken over.
FFT_DELAY_RANGE_PS: tuple[float, float] = (-200, 200)

# Number of neighbouring points averaged by the moving-average smoothing.
SMOOTHING_POINTS = 3


def crop_scan(scan: ScanDataBase, delay_range_ps: tuple[float, float]) -> ScanDataBase:
    """Keep only the points whose probe delay lies inside delay_range_ps."""
    lo, hi = delay_range_ps
    kept = [
        (d, m) for d, m in zip(scan.delays, scan.measured_values)
        if lo <= d.value(Prefix.PICO) <= hi
    ]
    return ScanDataBase(
        delays=[d for d, _ in kept],
        measured_values=[m for _, m in kept],
        run_id=scan.run_id,
    )


def smooth_scan(scan: ScanDataBase, n_points: int) -> ScanDataBase:
    """n-point moving average of the intensity. Edges are padded by repeating the end values."""
    intensity = np.array([m.value for m in scan.measured_values], dtype=float)
    smoothed = uniform_filter1d(intensity, size=n_points, mode="nearest")
    return ScanDataBase(
        delays=list(scan.delays),
        measured_values=[Measurement(float(v), 0) for v in smoothed],
        run_id=scan.run_id,
    )

def xcorr_sim(f0,epsilon,gamma,tau,I,dt,offset):
    x = np.arange(-800,800,dt)
    sigma = tau / (2 * np.sqrt(2 * np.log(2))) 
    U = gaussian(x,1,0,sigma)
    phi = 2*np.pi*(f0*x + epsilon*x**2 + gamma*x**3 )
    return (x, I*np.square(U * np.cos(phi))+offset)
    

def main() -> None:
    scan_4f = load_xcorr_csv(CSV_PATH_4f, DELAY_CENTER_MM)
    scan_cf = load_xcorr_csv(CSV_PATH_cf, DELAY_CENTER_MM)
    #frequency_ghz, spectrum = calculate_xcorr_fft(crop_scan(scan, FFT_DELAY_RANGE_PS))
    
    freq_4f, spectrum_4f = calculate_xcorr_fft(scan_4f)
    freq_cf, spectrum_cf = calculate_xcorr_fft(scan_cf)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    # Top: the cross-correlation trace (intensity vs probe delay).
    # delay_4f = [d.value(Prefix.PICO) for d in scan_4f.delays]
    # intensity_4f = [m.value for m in scan_4f.measured_values]
    # delay_cf = [d.value(Prefix.PICO) for d in scan_cf.delays]
    # intensity_cf = [m.value for m in scan_cf.measured_values]
    
    
    
    #ax1.plot(delay_4f, intensity_4f, color="tab:blue", label="cfCFG4f compensated")

    # Resampled trace on the uniform grid, i.e. what calculate_xcorr_fft actually FFTs.
    #resampled = resample_scans([scan], StftAnalysisConfig([scan]).axis)[0]
    
    #Simulated CFG 
    f0 = 44.25*1e-3 # 2-photon in THz
    #gamma = 0.015 # creates about 5 Ghz cfg bandwidth 
    gamma = 0
    epsilon = 0
    tau = 400 # in ps
    I = 0.06 # to match the photodiode voltage amplitude
    dt = 0.1
    offset = 0.0024
    x, y = xcorr_sim(f0,epsilon,gamma,tau,I,dt,offset)
    
    scalefactor = 0.037
    power_offset = 0.13
    power = scalefactor * np.abs(rfft(y)) + power_offset
    f_ghz = rfftfreq(y.size, d=dt) * 1e3 
    

    
    #ax1.plot(x,y,color="tab:green",lw=1.2,label=f"Simulated single frequency")
    
    # ax1.plot(
    #     delay_ps, [m.value for m in smoothed.measured_values],
    #     color="tab:green", lw=1.2, label=f"{SMOOTHING_POINTS}-pt smoothed",
    # )
    #ax1.axvspan(*FFT_DELAY_RANGE_PS, color="tab:orange", alpha=0.15, label="FFT window")
    # ax1.set_xlabel("Probe delay (ps)")
    # ax1.set_ylabel("Photodiode (V)")
    # ax1.legend(loc="upper right")
    # ax1.set_visible(False)

    # Bottom: FFT magnitude of the trace inside FFT_DELAY_RANGE_PS.
    ax.plot(freq_4f, np.abs(spectrum_4f), color="tab:blue", label="cfCFG4f compensated")
    ax.plot(freq_cf,np.abs(spectrum_cf),color="tab:red",label="cfCFG")
    ax.plot(f_ghz,power,color="tab:green",lw=1.2,label=f"Simulated single frequency")
    
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|FFT| (arb.)")
    ax.legend(loc="upper right")
    ax.set_xlim(10,150)
    #ax.set_ylim(0,1)

    #fig.suptitle(f"XCORR FFT: {CSV_PATH.parent.name}/{CSV_PATH.name}", fontsize=11)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
