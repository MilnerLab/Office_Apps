from pathlib import Path
CALC_FILE_NAME = str("_CalcScan")

def create_save_path_for_calc_ScanFile(folder_path: Path, name: str) -> Path:
    name = f"{name + CALC_FILE_NAME}.dat"
    folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path / name