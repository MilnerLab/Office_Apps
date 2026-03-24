from __future__ import annotations

import os
import shutil
from pathlib import Path


# --- Server root (Valery share) ------------------------------------------------
if os.name == "nt":  # Windows
    MOST_RECENT_FOLDER = Path(r"Z:\Droplets")
else:  # Linux
    MOST_RECENT_FOLDER = Path("/mnt/valeryshare/Droplets/")


# --- File patterns -------------------------------------------------------------
SCAN_FILE_PATTERN = "*ScanFile.dat"
BATCH_PATTERN = "*ScanFile.txt"
ION_FILE_PATTERN = "*mm.dat"


# --- Repo-local cache (_temp) --------------------------------------------------
def _find_repo_root(start: Path) -> Path:
    """Walk upwards until we find a repo marker. Falls back to the file's parent."""
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / "setup.cfg").exists():
            return parent
    return start.parent


_REPO_ROOT = _find_repo_root(Path(__file__))
_TEMP_ROOT = _REPO_ROOT / "_temp"


def _cache_dest_for(folder: Path) -> Path | None:
    """
    Returns the local cache destination for a folder under MOST_RECENT_FOLDER.
    - Only caches paths strictly *below* MOST_RECENT_FOLDER (not the root itself).
    - Returns None if the folder is not cache-eligible.
    """
    try:
        rel = folder.resolve().relative_to(MOST_RECENT_FOLDER.resolve())
    except Exception:
        return None

    if rel == Path("."):
        return None  # don't cache whole Droplets root

    return _TEMP_ROOT / rel


def _ensure_cached(folder: Path) -> Path:
    """
    If `folder` is under MOST_RECENT_FOLDER and not yet in _temp, copy it there.
    Returns the path that should be used (cached or original).
    """
    dest = _cache_dest_for(folder)
    if dest is None:
        return folder
    
    if not folder.exists():
        raise FileNotFoundError(f"Source folder does not exist: {folder}")
    
    if folder.is_dir(): 
        if dest.exists():
            for path in folder.rglob(ION_FILE_PATTERN):
                rel = path.relative_to(folder)
                target = dest / rel
                if not target.exists():
                    
                    #target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path,target)
                    # compare modification time + size
                    
                    """ if (
                        path.stat().st_mtime > dest.stat().st_mtime
                        or path.stat().st_size != dest.stat().st_size
                    ):
                        shutil.copy2(path, dest) """
        #dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            shutil.copytree(folder, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(folder, dest)

    return dest

            
def convert_to_system_path(paths: list[Path]) -> list[Path]:
    """
    Normalize incoming folder paths.

    Accepted inputs:
      - absolute paths (kept as-is)
      - relative paths under Droplets (treated as relative to MOST_RECENT_FOLDER)
      - relative cache paths like "_temp/20260206/Scan7" (resolved to repo/_temp/...)

    Also fixes mixed separators like "20260206\\Scan7" on Linux.
    """
    out: list[Path] = []

    for p in paths:
        s = str(p).replace("\\", "/").strip()

        # (1) Cache-relative paths: "_temp/..." or "./_temp/..."
        if s == "_temp" or s.startswith("_temp/") or s.startswith("./_temp/"):
            rel = s.removeprefix("./")
            out.append((_REPO_ROOT / rel).resolve())
            continue

        # (2) Absolute paths stay absolute (includes absolute cache paths)
        if Path(s).is_absolute():
            out.append(Path(s))
            continue

        # (3) Everything else is relative to MOST_RECENT_FOLDER
        parts = [part for part in s.split("/") if part]
        out.append(MOST_RECENT_FOLDER / Path(*parts))

    return out


# --- Main class ----------------------------------------------------------------
class DatFinder:
    def __init__(
        self,
        folder_paths: Path | list[Path] | None = None,
        is_full_path: bool = False,
        *,
        cache_to_local: bool = True,
    ):
        if folder_paths is None:
            folder_paths = [MOST_RECENT_FOLDER]
            is_full_path = True
        elif isinstance(folder_paths, Path):
            folder_paths = [folder_paths]
        else:
            folder_paths = list(folder_paths)

        folders = folder_paths if is_full_path else convert_to_system_path(folder_paths)

        if cache_to_local:
            folders = [_ensure_cached(f) for f in folders]

        self.folder_paths: list[Path] = folders

    def find_scanfiles(self, merge_batches: bool = False) -> list[Path]:
        scans_paths: list[Path] = []

        for f in self.folder_paths:
            file_list: list[Path] = sorted(f.glob(SCAN_FILE_PATTERN))
            if not file_list:
                continue

            # This batching logic is meant for the "live" Droplets root only.
            # We intentionally do not cache MOST_RECENT_FOLDER itself.
            if f == MOST_RECENT_FOLDER:
                txt_files = sorted(f.glob(BATCH_PATTERN))
                if txt_files:
                    newest_batch_stem = txt_files[-1].stem
                    if not merge_batches:
                        file_list = [p for p in file_list if p.stem >= newest_batch_stem]

                if not file_list:
                    continue

            scans_paths.extend(file_list)

        return scans_paths

    def find_most_recent_scanfile(self) -> Path:
        file_list = sorted(MOST_RECENT_FOLDER.glob(SCAN_FILE_PATTERN))
        return file_list[-1]

    def find_datafiles(self) -> list[list[Path]]:
        scans_paths: list[list[Path]] = []

        for f in self.folder_paths:
            all_files = sorted(f.glob(ION_FILE_PATTERN))
            scans_paths.append(all_files)

        return scans_paths
    

        