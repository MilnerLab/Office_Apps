"""Run the XCORR analysis chain for the shaped usCFG paper; all outputs go to _temp/shaped_usCFG_paper.

Reads Z:\\Droplets\\shaped_usCFG_paper\\20260825 (scan_L) and \\20260831 (scan_d), and for
the jet stage Jet\\Jet Truncation\\20260904 (the jet-accompanying cross-correlation).
Stages, in order: traces -> spectra -> seeds -> joint -> scans -> jet.
"""
import argparse
import time

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import (
    jet_prediction, joint_fit, scan_fits, spectra, xcorr_fit)

STAGES = ["traces", "spectra", "seeds", "joint", "scans", "jet"]
SEED_CSV = config.INPUTS_DIR / "seed_scanL.csv"


def run_traces() -> None:
    xcorr_fit.write_csv(xcorr_fit.fit_all(), config.TEMP_DIR / "fits.csv")


def run_spectra() -> None:
    spectra.run(out=config.TEMP_DIR, seed_from=SEED_CSV)


def run_seeds() -> None:
    joint_fit.run_seeds(config.TEMP_DIR)


def run_joint() -> None:
    joint_fit.run_joint(config.TEMP_DIR)


def run_scans() -> None:
    scan_fits.run(config.TEMP_DIR)


def run_jet() -> None:
    jet_prediction.run(config.TEMP_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    stage = ap.parse_args().stage
    todo = STAGES if stage == "all" else [stage]
    for s in todo:
        fn = globals().get(f"run_{s}")
        if fn is None:
            print(f"stage {s}: not ported yet, skipped")
            continue
        t0 = time.perf_counter()
        fn()
        print(f"stage {s}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
