"""Run the whole analysis for the shaped usCFG paper; all outputs go to _temp/shaped_usCFG_paper.

Reads, under Z:\\Droplets\\shaped_usCFG_paper:
  20260825 (scan_L) and 20260831 (scan_d)   the two XCORR characterization scans
  Jet                                       the 2026-09 gas-jet CS2 scans and the
                                            jet-accompanying cross-correlation
  truncation                                the 2026-09-18 truncation calibration data
Stages, in order: traces -> spectra -> seeds -> joint -> scans -> jet -> truncation
-> jet_reductions (the last reads jet_prediction.json, written by jet).
"""
import argparse
import time

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import (
    jet, jet_prediction, joint_fit, scan_fits, spectra, truncation, xcorr_fit)

STAGES = ["traces", "spectra", "seeds", "joint", "scans", "jet", "truncation", "jet_reductions"]
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


def run_truncation() -> None:
    truncation.run(config.TEMP_DIR)


def run_jet_reductions() -> None:
    jet.run_oscillations(config.TEMP_DIR)
    jet.run_truncation(config.TEMP_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    stage = ap.parse_args().stage
    todo = STAGES if stage == "all" else [stage]
    for s in todo:
        t0 = time.perf_counter()
        globals()[f"run_{s}"]()
        print(f"stage {s}: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
