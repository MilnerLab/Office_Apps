from pathlib import Path

from Lab_apps._base.models import PlotColor
from Lab_apps._io.dat_finder import SCAN_FILE_PATTERN


def pick_color(file_path: Path, num_color=1) -> PlotColor:
    date = file_path.stem
    date = date.split(SCAN_FILE_PATTERN)[0]
    date = date[4:8]
    date = int(date)
    
    if date%2 == 0:
        use_color = PlotColor.RED
        if num_color == 2:
            use_color = PlotColor.GRAY
                        
    else:
        use_color = PlotColor.BLUE
        if num_color == 2:
            use_color = PlotColor.GRAY
    
    return use_color



