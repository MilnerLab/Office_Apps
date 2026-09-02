from pathlib import Path

from _data_io.dat_finder import MOST_RECENT_FOLDER, DatFinder
from apps.plot_bot.domain.color_picker import pick_color
from apps.plot_bot.domain.plotting import PlottingBotPlotting


def process_scan_file(file_path: Path) -> Path | None:
    """
    Plot the newest batch of scans.

    Returns the path of the written png, or None if `file_path` is not the
    newest scan file (an even newer one is already being watched, so this one
    would only produce a duplicate post).
    """
    finder = DatFinder()

    scan_files = finder.find_scanfiles()
    if not scan_files or file_path.stem != scan_files[-1].stem:
        return None

    png_path = MOST_RECENT_FOLDER / "plots" / "PlotBot_cos2.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)

    PlottingBotPlotting(
        finder,
        png_path,
        color_cos2=pick_color(file_path),
        color_data_ions=pick_color(file_path, num_color=2),
    )

    return png_path
