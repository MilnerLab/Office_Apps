# monitoring/scan_detector.py
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

from Lab_apps._io.dat_finder import SCAN_FILE_PATTERN
from Lab_apps.plot_bot.domain.config import BotConfig


FileState = Tuple[int, float]  # (size in bytes, mtime)


@dataclass
class DirectoryWatcher:
    """
    Watches a directory and reports scan files that have not changed for some time.

    Behaviour:
    - Existing files at startup can be ignored.
    - For each new/changed file:
        * First, the inactivity threshold is set to the global
          `inactivity_threshold`.
        * As soon as the file contains at least two data points, a dynamic
          threshold is computed from the first two timestamps (column 0) and
          replaces the base threshold.
    """

    

    def __init__(self, config: BotConfig) -> None:
        self.watch_dir: Path = config.WATCH_DIR
        self.pattern: str = SCAN_FILE_PATTERN
        self.inactivity_threshold: float = config.INACTIVITY_THRESHOLD           
        self.dynamic_factor: Optional[int] = config.DYNAMIC_INACTIVITY_MULTIPLIER
        self.ignore_existing_on_start: bool = True

        self._last_seen: Dict[Path, FileState] = {}
        self._stable_since: Dict[Path, float] = {}
        self._processed: Set[Path] = set()
        self._per_file_threshold: Dict[Path, float] = {}
        self._dynamic_active: Set[Path] = set() 

        if self.ignore_existing_on_start:
            self._prime_existing_files()

    # ------------------------------------------------------------------ #
    # Initial: treat existing files as already processed (optional)
    # ------------------------------------------------------------------ #
    def _prime_existing_files(self) -> None:
        now = time.time()
        for path in self.watch_dir.glob(self.pattern):
            if not path.is_file():
                continue

            key = path.resolve()
            stat = path.stat()
            self._last_seen[key] = (stat.st_size, stat.st_mtime)
            self._stable_since[key] = now
            self._processed.add(key)
            # existing files get base threshold, but we never switch them
            self._per_file_threshold[key] = self.inactivity_threshold

    # ------------------------------------------------------------------ #
    # Utilities for thresholds
    # ------------------------------------------------------------------ #
    def _ensure_base_threshold(self, path: Path) -> Path:
        """
        Make sure the file has at least the base threshold registered.
        Returns the canonical key (resolved path).
        """
        key = path.resolve()
        if key not in self._per_file_threshold:
            self._per_file_threshold[key] = self.inactivity_threshold
            # not dynamic yet
            print(
                f"[Watcher] Using base threshold {self.inactivity_threshold:.2f}s "
                f"for new file {key.name}"
            )
        return key

    @staticmethod
    def _parse_timestamp(ts: str) -> datetime:
        """
        Parse timestamps of the form YYYYMMDDhhmmss.ffffff
        (example: 20251113183352.00000) into a datetime object.
        """
        if "." in ts:
            main, frac = ts.split(".", 1)
        else:
            main, frac = ts, ""

        dt_main = datetime.strptime(main, "%Y%m%d%H%M%S")

        micro = 0
        if frac:
            digits = "".join(ch for ch in frac if ch.isdigit())
            if digits:
                frac_float = float("0." + digits)
                micro = int(round(frac_float * 1_000_000))

        return dt_main + timedelta(microseconds=micro)

    def _try_compute_dynamic_threshold(self, path: Path) -> Optional[float]:
        """
        Try to compute a dynamic threshold from the first two timestamps
        in the file.

        Returns:
            - float: dynamic threshold in seconds (>= base threshold)
            - None: if there are fewer than 2 data points or an error occurs.
        """
        if self.dynamic_factor is None or self.dynamic_factor <= 0:
            return None

        try:
            timestamps: List[str] = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if not parts:
                        continue
                    timestamps.append(parts[0])
                    if len(timestamps) >= 2:
                        break

            # Not enough data points yet -> do NOT switch to dynamic
            if len(timestamps) < 2:
                return None

            t0 = self._parse_timestamp(timestamps[0])
            t1 = self._parse_timestamp(timestamps[1])
            dt_sec = (t1 - t0).total_seconds()

            if dt_sec <= 0:
                print(
                    f"[Watcher] Warning: non-positive dt ({dt_sec}) for {path.name}, "
                    "keeping base threshold."
                )
                return None

            dyn = dt_sec * self.dynamic_factor
            return max(dyn, self.inactivity_threshold)

        except Exception as exc:
            print(
                f"[Watcher] Warning: could not compute dynamic threshold for "
                f"{path.name}: {exc}. Keeping base threshold."
            )
            return None

    def _get_threshold_for_file(self, path: Path) -> float:
        """
        Return the threshold for this file.

        Logic:
        - Ensure the file has the base threshold stored.
        - If we have not switched to dynamic yet AND there are at least
          two data points, compute a dynamic threshold and switch once.
        """
        key = self._ensure_base_threshold(path)

        # already using dynamic threshold?
        if key not in self._dynamic_active:
            dyn = self._try_compute_dynamic_threshold(key)
            if dyn is not None:
                self._per_file_threshold[key] = dyn
                self._dynamic_active.add(key)
                print(
                    f"[Watcher] Switched {key.name} to dynamic threshold: "
                    f"{dyn:.2f}s"
                )

        return self._per_file_threshold[key]

    # ------------------------------------------------------------------ #
    # Main scan method
    # ------------------------------------------------------------------ #
    def scan(self, now: float | None = None) -> List[Path]:
        """
        Perform a single scan:

        - update internal state
        - return a list of files that are now considered "finished"
        """
        if now is None:
            now = time.time()

        finished_files: List[Path] = []

        for path in self.watch_dir.glob(self.pattern):
            if not path.is_file():
                continue

            key = path.resolve()
            stat = path.stat()
            current_state: FileState = (stat.st_size, stat.st_mtime)
            last_state = self._last_seen.get(key)

            # New file or changed file
            if last_state != current_state:
                self._last_seen[key] = current_state
                self._stable_since[key] = now
                # make sure it has at least the base threshold
                self._ensure_base_threshold(path)
                # if it was already processed before and changed again, re-activate it
                self._processed.discard(key)
                continue

            # Unchanged
            if key in self._processed:
                continue

            stable_for = now - self._stable_since.get(key, now)
            threshold = self._get_threshold_for_file(path)

            if stable_for >= threshold:
                finished_files.append(key)
                self._processed.add(key)

        # Cleanup: remove deleted files
        to_delete = [p for p in self._last_seen.keys() if not p.exists()]
        for p in to_delete:
            self._last_seen.pop(p, None)
            self._stable_since.pop(p, None)
            self._processed.discard(p)
            self._per_file_threshold.pop(p, None)
            self._dynamic_active.discard(p)

                # Console log: which files are currently active?
        active: List[Tuple[Path, float, float, Optional[float]]] = []
        for key in self._last_seen.keys():
            if key in self._processed:
                continue
            stable_for = now - self._stable_since.get(key, now)
            # use _get_threshold_for_file to potentially switch to dynamic
            threshold = self._get_threshold_for_file(key)
            last_delay = self._get_last_delay(key)
            active.append((key, stable_for, threshold, last_delay))

        if active:
            print("[Watcher] Active file(s):")
            for key, stable_for, threshold, last_delay in active:
                if last_delay is not None:
                    print(
                        f"  - {key.name}: stable for {stable_for:.1f}s "
                        f"(threshold {threshold:.1f}s, last delay {last_delay:.3f} ps)"
                    )
                else:
                    print(
                        f"  - {key.name}: stable for {stable_for:.1f}s "
                        f"(threshold {threshold:.1f}s, last delay unknown)"
                    )
        else:
            print("[Watcher] No active file (waiting for new scan...)")


        return finished_files

    def reset_processed(self, path: Path) -> None:
        """Manually mark a file as not processed yet."""
        key = path.resolve()
        self._processed.discard(key)

    def _get_last_delay(self, path: Path) -> Optional[float]:
        """
        Return the last probe-delay value (column 1) found in the file.
        If it cannot be parsed, return None.
        """
        try:
            last_delay: Optional[float] = None
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) < 2:
                        continue
                    try:
                        # column 1 (index 1) is the probe delay in ps
                        last_delay = float(parts[1])
                    except ValueError:
                        continue

            return last_delay
        except Exception as exc:
            print(
                f"[Watcher] Warning: could not read last delay from {path.name}: {exc}"
            )
            return None

