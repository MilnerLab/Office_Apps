from dataclasses import dataclass
from pathlib import Path

from Lab_apps._domain.models import ScanDataBase


@dataclass(frozen=True)
class AveragedScansData(ScanDataBase):
    file_names: list[Path]