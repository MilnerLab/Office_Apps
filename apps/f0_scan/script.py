import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from _data_io.dat_loader import load_ion_data
from apps.c2t_calculation.domain.analysis import run_pipeline
from apps.c2t_calculation.domain.plotting import plot_calculated_scan, plot_ions_square, plot_ScanData
from _data_io.dat_finder import MOST_RECENT_FOLDER, DatFinder
from _data_io.dat_saver import create_save_path_for_calc_ScanFile
from apps.scan_averaging.domain import averaging

from base_core.lab_specifics.base_models import IonDataAnalysisConfig
import base_core.physics.molecules as mol
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_core.fitting.functions import fit_gaussian
from base_core.math.functions import gaussian


POSZEROSHIFT = 0 #millimetres :)

def get_scan_number(folder_path: Path = MOST_RECENT_FOLDER) -> int:
        scan_files = DatFinder(folder_path).find_scanfiles()
        return len(scan_files)


ocs = mol.OCS()
j_arr = np.arange(0,15,1)
n_transitions = len(j_arr)
transitions = np.zeros(n_transitions)

for i,j in enumerate(j_arr):
        transitions[i] = mol.twophoton(ocs,j).value(Prefix.GIGA)


folder_path = Path(r"Z:\Droplets\20261026\Scan3_FREQ")
file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()

config = IonDataAnalysisConfig(
    delay_center= Length(15.735, Prefix.MILLI),
    center=Point(186,196),
    angle= Angle(12, AngleUnit.DEG),
    analysis_zone= Range[int](30, 120),
    transform_parameter=0.7)

raw_scans = load_ion_data(file_paths)
calculated_Scan = run_pipeline(raw_scans, [config])[0]
#scan_avg = averaging.average_scans(calculated_Scan)

x = [time*1e12 for time in calculated_Scan.delays]
y = [c2t.value for c2t in calculated_Scan.measured_values]
y_error = np.array([c.error for c in calculated_Scan.measured_values])

freq = 2*46.7*np.array(x)/6.6#cameron's slope from 2026/10/16 to convert mm to ghz and then from mm to ps
fit_result = fit_gaussian(x,y)

fig,ax = plt.subplots()

label = "Time Zero = " + str(round(fit_result.center,1)) + ". Center = (" + str(config.center.x) + ", " + str(config.center.y) + "), Angle = " + str(round(config.angle.Deg,1)) + "\n"\
        + "Ring: (" + str(config.analysis_zone.min) + ", " + str(config.analysis_zone.max) + "), ScaleX = " + str(config.transform_parameter)
#plot_calculated_scan(ax3, calculated_Scan[0],label=label)

ax.grid(visible=True,which='major',alpha=1.0)
#plot_ScanData(ax,scan_avg,label=label)
#plot_ScanData(ax,calculated_Scan,label=label)
ax.errorbar(freq,y,y_error,elinewidth=1,ecolor="r")
#ax.plot(transitions,0.56*np.ones(n_transitions),linestyle='',marker='|',markersize=150,color='k')
ax.set_xlabel('Centrifuge Two-Photon Frequency (GHz)')
ax.set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")

# Top horizontal axis
ax2 = ax.twiny()
# Move the second x-axis to the bottom
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 40))
ax2.spines['top'].set_visible(False)
ax2.set_xlabel("Time Between Arms (ps)")
ax2.plot(x,y)
x_a = np.array(x)

# Reserve room for the second axis and its label
fig.subplots_adjust(bottom=0.25)

plt.show()