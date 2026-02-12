from dataclasses import dataclass
from pathlib import Path

from _domain.models import LoadableScan, ScanDataBase

@dataclass(frozen=True)
class AveragedScansData(LoadableScan):
    file_names: list[Path] = None