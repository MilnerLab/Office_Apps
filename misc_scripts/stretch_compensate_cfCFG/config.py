# misc_scripts/stretch_compensate_cfCFG/config.py
import shutil
import subprocess
from pathlib import Path

# --- Locations ---------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OFFICE_APPS = HERE.parents[1]
OUTPUT_DIR = OFFICE_APPS / "_temp" / "stretch_compensate_cfCFG"

LAYOUT_PNG = Path(r"Z:\Droplets\plots\possible layout.png")   # the sketch the layout is read off
CORKSCREW_PNG = HERE / "corkscrew_uscfg.png"                     # from Fig. 2 of PRA 113, 033107

# --- Inkscape (PDF export) ---------------------------------------------------
INKSCAPE = Path(shutil.which("inkscape") or r"C:\Program Files\Inkscape\bin\inkscape.exe")


def _cmu_dir() -> Path:
    """The TeX distribution's CMU fonts, found with kpsewhich wherever TeX is installed."""
    try:
        hit = subprocess.run(["kpsewhich", "cmunrm.otf"], capture_output=True, text=True).stdout.strip()
    except OSError:
        hit = ""
    return Path(hit).parent if hit else Path(r"C:\texlive\2025\texmf-dist\fonts\opentype\public\cm-unicode")


# Computer Modern, to match LaTeX documents. CMU Serif ships with TeX Live, so Inkscape is
# pointed at it through a fontconfig file rather than an install.
CMU_DIR = _cmu_dir()
FONTS_CONF = ('<?xml version="1.0"?><fontconfig>'
              f'<include ignore_missing="yes">{INKSCAPE.parents[1] / "etc" / "fonts" / "fonts.conf"}</include>'
              f'<dir>{CMU_DIR}</dir></fontconfig>')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
