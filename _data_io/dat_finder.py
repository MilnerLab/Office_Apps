from pathlib import Path
import os

if os.name == "nt":  # Windows
    MOST_RECENT_FOLDER = Path(r"Z:\Droplets")
else:                # Linux
    MOST_RECENT_FOLDER = Path("/mnt/valeryshare/Droplets/")

SCAN_FILE_PATTERN = '*ScanFile.dat'
BATCH_PATTERN = '*ScanFile.txt'
ION_FILE_PATTERN = '*mm.dat'

class DatFinder:
    def __init__(self, folder_path: Path = MOST_RECENT_FOLDER):
        self.folder_path = folder_path
        self.batch_finished = False
        
    def find_scanfiles(self,merge_batches = False) -> list[Path]:
        file_list: list[Path] = sorted(self.folder_path.glob(SCAN_FILE_PATTERN))

        if not file_list:
            return []

        if self.folder_path == MOST_RECENT_FOLDER:
            txt_files = sorted(self.folder_path.glob(BATCH_PATTERN))

            if txt_files:
                newest_batch_stem = txt_files[-1].stem
                if merge_batches == True:
                    file_list = [f for f in file_list]
                else:
                    file_list = [f for f in file_list if f.stem >= newest_batch_stem]

            if not file_list:
                return []

            first_size = file_list[0].stat().st_size
            last_size = file_list[-1].stat().st_size

            if last_size < first_size:
                file_list = file_list[:-1]
                self.batch_finished = False
            else:
                self.batch_finished = True
            
            return file_list
        else:
            self.batch_finished = True
            return file_list

    def find_most_recent_scanfile(self) -> Path:
        file_list = list(MOST_RECENT_FOLDER.glob(SCAN_FILE_PATTERN)) 
        file_list = sorted(file_list)

        return file_list[-1]

    def find_datafiles(self)-> list[Path]:
        all_files = list(self.folder_path.glob(ION_FILE_PATTERN))
        all_files.sort()
        self.batch_finished = True
        
        return all_files
    