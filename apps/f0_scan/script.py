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
from base_core.physics.helpers import twophoton, rot_energy
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle, Point, Range
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_core.fitting.functions import fit_gaussian
from base_core.math.functions import gaussian

from matplotlib.widgets import Button, TextBox


POSZEROSHIFT = 0 #millimetres :)
PS_TO_GHZ = 2*46.7/6.6 #cameron's slope from 2026/10/16 to convert mm to ghz and then from mm to ps

DEFAULT_FOLDER = r"Z:\Droplets\20261110\FreqScan_1"
DEFAULT_MOLECULE = "OCS"

j_arr = np.arange(0,1,1)

# (label, default) for each IonDataAnalysisConfig widget
CONFIG_DEFAULTS = {
    "delay_center": ("Delay center (mm) ", 15.735),
    "center_x": ("Center x ", 139),
    "center_y": ("Center y ", 210),
    "angle": ("Angle (deg) ", 12),
    "zone_min": ("Zone min ", 50),
    "zone_max": ("Zone max ", 120),
    "transform_parameter": ("ScaleX ", 0.8),
}


def build_config(values: dict[str, float]) -> IonDataAnalysisConfig:
        return IonDataAnalysisConfig(
            delay_center= Length(values["delay_center"], Prefix.MILLI),
            center=Point(values["center_x"],values["center_y"]),
            angle= Angle(values["angle"], AngleUnit.DEG),
            analysis_zone= Range[int](int(values["zone_min"]), int(values["zone_max"])),
            transform_parameter=values["transform_parameter"])


def get_scan_number(folder_path: Path = MOST_RECENT_FOLDER) -> int:
        scan_files = DatFinder(folder_path).find_scanfiles()
        return len(scan_files)


def get_transitions(molecule_name: str) -> np.ndarray:
        molecule_cls = getattr(mol, molecule_name.strip(), None)
        if not (isinstance(molecule_cls, type) and issubclass(molecule_cls, mol.Molecule)):
                raise ValueError(f"Unknown molecule '{molecule_name}'")
        molecule = molecule_cls()
        return np.array([twophoton(molecule,j,True).value(Prefix.GIGA) for j in j_arr])


def plot_f0_scan(fig, ax, folder_path: Path, molecule_name: str, config: IonDataAnalysisConfig):
        transitions = get_transitions(molecule_name)

        file_paths = DatFinder(folder_path,is_full_path=True).find_datafiles()
        raw_scans = load_ion_data(file_paths)
        calculated_Scan = run_pipeline(raw_scans, [config])[0]
        #scan_avg = averaging.average_scans(calculated_Scan)

        x = [time*1e12 for time in calculated_Scan.delays]
        y = [c2t.value for c2t in calculated_Scan.measured_values]
        y_error = np.array([c.error for c in calculated_Scan.measured_values])

        freq = PS_TO_GHZ*np.array(x)
        #fit_result = fit_gaussian(x,y)

        # label = "Time Zero = " + str(round(fit_result.center,1)) + ". Center = (" + str(config.center.x) + ", " + str(config.center.y) + "), Angle = " + str(round(config.angle.Deg,1)) + "\n"\
        #         + "Ring: (" + str(config.analysis_zone.min) + ", " + str(config.analysis_zone.max) + "), ScaleX = " + str(config.transform_parameter)

        ax.grid(visible=True,which='major',alpha=1.0)
        #plot_ScanData(ax,scan_avg,label=label)
        #plot_ScanData(ax,calculated_Scan,label=label)
        ax.errorbar(freq,y,y_error,elinewidth=1,ecolor="r")
        ax.plot(transitions,0.52*np.ones(len(transitions)),linestyle='',marker='|',markersize=150,color='k')
        ax.set_xlabel('Centrifuge Two-Photon Frequency (GHz)')
        ax.set_ylabel(r"$\langle \cos^2 \theta_\mathrm{2D} \rangle$")
        ax.set_title(f"{folder_path.name} ({len(file_paths)} files), {molecule_name}")

        # Top horizontal axis
        ax2 = ax.twiny()
        # Move the second x-axis to the bottom
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))
        ax2.spines['top'].set_visible(False)
        ax2.set_xlabel("Time Between Arms (ps)")
        # Same data as ax, so just match the limits rather than plotting it twice
        ax2.set_xlim(np.array(ax.get_xlim())/PS_TO_GHZ)
        return ax2


def main() -> None:
        fig,ax = plt.subplots(figsize=(11, 6))
        # Reserve room for the second axis, its label and the widgets
        fig.subplots_adjust(bottom=0.3, right=0.7)

        config_boxes = {}
        for i, (key, (label, default)) in enumerate(CONFIG_DEFAULTS.items()):
                config_boxes[key] = TextBox(fig.add_axes((0.86, 0.85 - 0.07*i, 0.1, 0.05)), label, initial=str(default))

        folder_box = TextBox(fig.add_axes((0.12, 0.04, 0.45, 0.05)), "Folder ", initial=DEFAULT_FOLDER)
        molecule_box = TextBox(fig.add_axes((0.67, 0.04, 0.1, 0.05)), "Molecule ", initial=DEFAULT_MOLECULE)
        refresh_button = Button(fig.add_axes((0.8, 0.04, 0.15, 0.05)), "Refresh")

        twin_axes = []

        def on_refresh(event):
                for twin in twin_axes:
                        twin.remove()
                twin_axes.clear()
                ax.clear()
                try:
                        config = build_config({key: float(box.text) for key, box in config_boxes.items()})
                        twin_axes.append(plot_f0_scan(fig, ax, Path(folder_box.text.strip()), molecule_box.text, config))
                except Exception as e:
                        ax.set_title(f"Error: {e}", color="r")
                fig.canvas.draw_idle()

        refresh_button.on_clicked(on_refresh)
        on_refresh(None)

        plt.show()


if __name__ == "__main__":
        main()
