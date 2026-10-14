"""Where the paper's data lives: online (the Valery share) or offline (a local copy).

Online, every measurement day is a folder under Droplets (``Droplets/20261007/...``).
The offline copy keeps the same day folders, grouped by figure:
``<OFFLINE_ROOT>/characterizing/20261007/...``. So a script asks for its group's root and
appends the same day-relative path in both modes.

Mode: ``online``, ``offline`` or ``auto`` (the share if it is reachable, else the local
copy). Passed explicitly, or taken from the ``SOEREN_DATA`` environment variable,
defaulting to ``auto``.
"""
from __future__ import annotations

import os
from pathlib import Path

MODES = ("auto", "online", "offline")

#: Local copy of the data used in the latest draft.
OFFLINE_ROOT = Path("/mnt/data/paper/soeren_paper/latest draft data")


def share_root() -> Path:
    """Droplets on the Valery share: Z: on Windows; on Ubuntu /mnt/valeryshare if mounted,
    otherwise the GVFS mount the file manager creates for smb://."""
    if os.name == "nt":
        return Path(r"Z:\Droplets")
    mnt = Path("/mnt/valeryshare/Droplets")
    if mnt.is_dir():
        return mnt
    return (Path(f"/run/user/{os.getuid()}/gvfs")
            / "smb-share:server=phas-researchfs.phas.ubc.ca,share=valeryshare" / "Droplets")


def resolve_mode(mode: str | None = None) -> str:
    """'online' or 'offline'. ``auto`` picks online only if the share answers."""
    mode = (mode or os.environ.get("SOEREN_DATA") or "auto").lower()
    if mode not in MODES:
        raise ValueError(f"data mode must be one of {MODES}, got {mode!r}")
    if mode == "auto":
        try:
            mode = "online" if share_root().is_dir() else "offline"
        except OSError:  # a stale network mount can raise instead of returning False
            mode = "offline"
    return mode


def data_root(group: str, mode: str | None = None) -> Path:
    """Root that day folders ("20261007/...") are relative to.

    ``group`` is the offline subfolder: "characterizing", "stabilized_vs_unstabilized", ...
    """
    mode = resolve_mode(mode)
    root = share_root() if mode == "online" else OFFLINE_ROOT / group
    print(f"[data] {mode}: {root}")
    return root
