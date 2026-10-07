import json
import sys
from pathlib import Path

#Make the repo root importable no matter where the script is launched from
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
from matplotlib import pyplot as plt

from _data_io.dat_finder import DatFinder
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.scan_averaging.domain.averaging import average_scans
from apps.scan_averaging.domain.plotting import plot_averaged_scan
from base_core.lab_specifics.averaging.models import AveragedScansData
from base_core.lab_specifics.base_models import IonDataAnalysisConfig
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.plotting.enums import PlotColor
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length

#Update the matplotlib settings
plt.style.use(str(REPO_ROOT / "stylefiles" / "compare_c2t_spectrogram.mplstyle"))

POSZEROSHIFT = 0  # millimetres :)

#--- Spectrum-recording panel settings
SPEC_FOLDER = Path(r"Z:\Droplets\20261006")
WL_RANGE: tuple[float, float] | None = (780, 825)  #nm; None = full detector range
TIME_STRIDE = 10       #plot every Nth spectrum; 0.4 s/spectrum is far finer than the figure
SHOW_METADATA = True   #dashed lines at rig-state changes

#Folder to save figures
savefig_folder = r"Z:\Droplets\plots\\"


#FUNCTION TO GENERATE THE PLOTTABLE DATA
def calculating(folders: list[Path], configs: list[IonDataAnalysisConfig]) -> AveragedScansData:
    scans_paths = DatFinder(folders).find_datafiles()
    raw_datas = load_ion_data(scans_paths)
    calculated_scans = run_pipeline(raw_datas, configs)

    return average_scans(calculated_scans)


def find_recording_group(f: h5py.File) -> h5py.Group:
    """SpectrumRecorder writes at the root, but a merged file nests it in a subgroup."""
    if "traces" in f:
        return f

    found: list[h5py.Group] = []
    f.visititems(lambda _, obj: found.append(obj)
                 if isinstance(obj, h5py.Group) and "traces" in obj else None)
    if len(found) != 1:
        raise SystemExit(f"Found {len(found)} recording groups in {f.filename}")
    return found[0]


def load_spectrum_recording(path: Path):
    """Read a SPEC_*.h5 recording as (wavelength_nm, time_s, counts, rig-state changes)."""
    with h5py.File(path, "r") as f:
        g = find_recording_group(f)
        wl = g["wavelength_nm"][()]

        keys = sorted(g["traces"].keys())
        if not keys:
            raise SystemExit(f"No traces in {path}")
        t_start = int(keys[0])

        strided = keys[::TIME_STRIDE]
        counts = np.stack([g["traces"][k][()] for k in strided])
        time_s = np.array([(int(k) - t_start) / 1e9 for k in strided])

        changes: list[tuple[float, str]] = []
        if SHOW_METADATA and "metadata" in g:
            for k in sorted(g["metadata"].keys()):
                raw = g["metadata"][k][()]
                entry = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                reason = entry.get("reason", "") if isinstance(entry, dict) else ""
                changes.append(((int(k) - t_start) / 1e9, reason))

    if WL_RANGE:
        sel = (wl >= min(WL_RANGE)) & (wl <= max(WL_RANGE))
        wl, counts = wl[sel], counts[:, sel]

    return wl, time_s, counts, changes


#====================================================================================================
#20261006 - rotating stabilized vs unstabilized, both sharing one config.
#====================================================================================================

#Common config for every scan in the folder
common_config = IonDataAnalysisConfig(
    delay_center=Length(80.0 - POSZEROSHIFT, Prefix.MILLI),
    center=Point(168, 190),
    angle=Angle(-12, AngleUnit.DEG),
    analysis_zone=Range[int](30, 90),
    transform_parameter=0.920)

#One column per dataset, all sharing the common config above
datasets: list[Path] = [
    Path(r"20261006\RotatingStabilized1"),
    Path(r"20261006\RotatingUnstabilized1"),
]

#Bottom row: the last two spectrum recordings, paired with the datasets in order
spec_files = sorted(SPEC_FOLDER.glob("SPEC_*.h5"))[-len(datasets):]
for folder, spec in zip(datasets, spec_files):
    print(f"{folder.name} <- {spec.name}")

scans = [calculating([folder], [common_config]) for folder in datasets]
recordings = [load_spectrum_recording(spec) for spec in spec_files]

fig, axs = plt.subplots(
    nrows=2,
    ncols=len(datasets),
    figsize=(6.75 * len(datasets) / 2 + 3.4, 5.5),
    squeeze=False,
    height_ratios=[1, 1.4])

for col, folder in enumerate(datasets):
    ax_scan = axs[0, col]
    ax_spec = axs[1, col]

    #--- Top row: averaged <cos^2> trace
    with plt.rc_context({'lines.linewidth': 1.5}):
        plot_averaged_scan(ax_scan, scans[col], PlotColor.BLUE, ecolor=PlotColor.RED,
                           marker='d', label=folder.name)

    ax_scan.grid(True)
    ax_scan.set_xlabel('Probe Delay (ps)')
    ax_scan.legend(loc='upper right')

    #--- Bottom row: recorded spectra over time
    wl, time_s, counts, changes = recordings[col]
    mesh = ax_spec.pcolormesh(wl, time_s, counts, shading='nearest', cmap='viridis')
    fig.colorbar(mesh, ax=ax_spec, label='Counts')

    for t_change, reason in changes:
        if time_s[0] <= t_change <= time_s[-1]:
            ax_spec.axhline(t_change, color='w', lw=0.8, ls='--', alpha=0.7)
            ax_spec.text(wl[-1], t_change, f" {reason} ", color='w', fontsize=7,
                         va='bottom', ha='right')

    ax_spec.set_xlabel('Wavelength (nm)')
    ax_spec.grid(False)

    #Only the leftmost column keeps its y-labels
    if col > 0:
        ax_scan.set_ylabel('')
        ax_spec.set_ylabel('')

#Common y-scale per row so the two columns can be compared directly
for row in range(2):
    lo = min(ax.get_ylim()[0] for ax in axs[row, :])
    hi = max(ax.get_ylim()[1] for ax in axs[row, :])
    for ax in axs[row, :]:
        ax.set_ylim(lo, hi)

axs[0, 0].set_ylabel(r'$\langle \cos^2\theta_{2D}\rangle$')
axs[1, 0].set_ylabel('Time since first spectrum (s)')

fig.tight_layout()
fig.savefig(savefig_folder + "20261006_" + "_vs_".join(f.name for f in datasets) + ".pdf",
            format='pdf', dpi=300)

plt.show()
print('Done!')
