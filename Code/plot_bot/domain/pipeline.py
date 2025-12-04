from pathlib import Path

from Lab_apps._io.dat_finder import MOST_RECENT_FOLDER, DatFinder
from Lab_apps._io.dat_loader import load_time_scan
from Lab_apps.plot_bot.domain.color_picker import pick_color
from Lab_apps.plot_bot.domain.plotting import PlottingBotPlotting


def process_scan_file(file_path: Path) -> Path:
    
    

    png_path = MOST_RECENT_FOLDER / "plots" / f"PlotBot_cos2.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    
    finder = DatFinder()
    if file_path.stem == finder.find_scanfiles()[-1].stem:
        PlottingBotPlotting(
        finder,
        png_path,
        color_cos2=pick_color(file_path),
        color_ions=pick_color(file_path,num_color=2)
    )

    return png_path
