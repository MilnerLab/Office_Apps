"""Run the whole analysis for the shaped usCFG paper; all outputs go to _temp/shaped_usCFG_paper.

Reads, under Z:\\Droplets\\shaped_usCFG_paper:
  20260825 (scan_L) and 20260831 (scan_d)   the two XCORR characterization scans
  Jet                                       the 2026-09 gas-jet CS2 scans and the
                                            jet-accompanying cross-correlation
  truncation                                the 2026-09-18 truncation calibration data
Stages, in order: traces -> spectra -> seeds -> joint -> scans -> jet -> truncation
-> jet_reductions (the last reads jet_prediction.json, written by jet). ``--stage all``
runs the traces..jet chain and truncation concurrently, then jet_reductions.

After ``--stage all`` finishes, regenerate the manuscript's figures by running each of
the seven figure scripts (each writes its PDF straight into Latex/shaped_usCFG_paper/
figures/): fig_cfg_arms, fig_char_fitdemo, fig_char_freqtime, fig_char_scans,
fig_char_truncation, fig_jet_oscillations, fig_jet_truncation -- e.g.
``python -m manuscript_plotting_scripts.shaped_usCFG_paper.fig_cfg_arms``.
"""
import argparse
import threading
import time
import traceback

from base_core.framework.concurrency.task_runner import TaskRunner

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


#: Independent chains that ``--stage all`` runs concurrently, each in order on its own
#: TaskRunner thread. `jet_reductions` reads chain A's `jet_prediction.json`, so it
#: runs once both chains have finished.
CHAINS = {"xcorr": ["traces", "spectra", "seeds", "joint", "scans", "jet"],
          "truncation": ["truncation"]}
AFTER_CHAINS = ["jet_reductions"]


def run_stage(s: str) -> None:
    t0 = time.perf_counter()
    globals()[f"run_{s}"]()
    print(f"stage {s}: {time.perf_counter() - t0:.1f} s", flush=True)


def run_chains(chains: dict[str, list[str]]) -> None:
    """Run each chain's stages in order on its own TaskRunner, the chains concurrently.

    A failing stage stops its own chain; the other chains finish, and the first error
    is then raised here, in the main thread.
    """
    errors: list[BaseException] = []
    done: list[threading.Event] = []
    runners: list[TaskRunner] = []
    for name, stages in chains.items():
        ev = threading.Event()

        def chain(stages=stages, ev=ev) -> None:
            try:
                for s in stages:
                    run_stage(s)
            finally:
                ev.set()

        runner = TaskRunner(f"shaped_usCFG_paper.{name}")
        runner.run(chain, on_error=errors.append)
        done.append(ev)
        runners.append(runner)
    for ev in done:
        while not ev.wait(1.0):          # a timed wait keeps Ctrl-C responsive
            pass
    for runner in runners:
        runner.shutdown(wait=True, timeout=None)   # also lets on_error finish recording
    for extra in errors[1:]:
        traceback.print_exception(extra)
    if errors:
        raise errors[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ["all"], default="all")
    stage = ap.parse_args().stage
    if stage != "all":
        run_stage(stage)
        return
    t0 = time.perf_counter()
    run_chains(CHAINS)
    for s in AFTER_CHAINS:
        run_stage(s)
    print(f"all stages: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
