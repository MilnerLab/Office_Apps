"""Cubic-phase fringe fit (frequency vs probe delay) for a cross-correlation CSV.

Companion to `xcorr_csv_script.py`. That script measures the chirp with an STFT
(a spectrogram); this one measures it with the deterministic fringe-fit pipeline
developed for the spectrometer traces, which fits ONE global model

    y(t) = mid(t) + half(t) * cos(Phi(t)),   Phi = c0 + c1 u + c2 u^2 + c3 u^3

so the instantaneous frequency is the analytic derivative of a fitted polynomial
rather than a stack of windowed FFTs. It resolves chirp far below the STFT's
time-bandwidth floor, at the cost of assuming the phase really is a low-order
polynomial (no mode hops inside the scan).

The pipeline itself is `apps/stft_analysis/vendor/fringe_core.py`, a verbatim
copy of the standalone module (pure numpy/scipy). See `vendor/README.md` for its
provenance and the re-copy rule -- edit it upstream and re-copy the whole file,
never hand-patch it here.

Units -- the whole adaptation
----------------------------
`fringe_core` was built for fringes in WAVELENGTH: the trace is a Gaussian bump
in nm carrying a chirped fringe, and frequencies come out in cycles/nm. An XCORR
trace is the same object in DELAY: a Gaussian-ish envelope in ps carrying a
chirped fringe. So instead of forking the pipeline we affine-map the delay axis
onto a pseudo-wavelength axis inside its analysis window, fit, and map the answer
back:

    lam = LAM0 + (t - t_mid) * scale        [scale in nm/ps]
    f[GHz] = 1e3 * f[cycles/nm] * scale

The map is affine, so it commutes with everything the fit does -- a polynomial
phase in t is a polynomial phase in lam of the same order, and the mapped
frequency is exact, not an approximation. Only the SCALE is a choice, and it is
chosen (SPAN_NM) so the scan fills the window the pipeline's tolerances were
tuned on.

CSV format expected (tab-separated), same as `xcorr_csv_script.py`:
    col 0      : stage position in mm
    cols 1..N  : N intensity samples (waveforms) for that delay point

Output: raw trace with the fitted reconstruction overlaid on top, and the
instantaneous frequency of that reconstruction underneath.

Known limit -- delay sampling
-----------------------------
The fringe frequency in these scans runs right up to the sampling Nyquist (on
202607181032_ the fit reaches 137 GHz against a 150 GHz Nyquist, i.e. ~2.2
samples per fringe at the fast end). The Nyquist line is drawn on the frequency
panel for that reason: where the curve approaches it, the fit is reading an
undersampled fringe and r2_fringe suffers (0.61 on that scan) no matter what the
pipeline does. Cropping the fast arm away does NOT help -- it costs the cubic
term and scores worse. The fix is a finer stage step, not a different fit.
"""

import sys
from pathlib import Path

# Allow running this file directly (python xcorr_fringe_fit_script.py) by putting
# the project root on sys.path; running as a module (-m) already handles this.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

from apps.stft_analysis.vendor import fringe_core as fc


# --- User config ------------------------------------------------------------
CSV_PATH = Path(
    r"D:\Documents\University\UBC research\2026\Data\20260718\XCORR\DA17.0_GA-75.0\202607181032_.csv"
)

# Stage position (mm) at pump/probe overlap (probe delay t = 0). Only shifts the
# absolute time axis; it does not affect the extracted frequencies.
DELAY_CENTER_MM = 135.0

# Optional crop of the probe-delay axis (ps), applied BEFORE the fit. Use it to
# cut a bad arm off the scan by hand. None -> use the whole scan.
DELAY_CROP_PS: tuple[float, float] | None = None

# Upper limit of the frequency axis (GHz). None -> autoscale to the fit.
FREQ_YLIM_GHZ: float | None = None
# ---------------------------------------------------------------------------


# --- Mapping constants (see the module docstring) ---------------------------
# Delay span mapped onto the pseudo-wavelength axis, in nm. fringe_core's window
# is ZOOM = (790, 814), i.e. 24 nm wide, and its envelope/crop heuristics assume
# the bump fills it; 22 nm leaves a little margin at both ends.
SPAN_NM = 22.0
LAM0 = 802.0            # window centre (fringe_core.RF_BAND_CENTRE_NM)

# Pseudo-counts the trace is rescaled to. NOT cosmetic: fit_upper_envelope
# refines the envelope with Nelder-Mead under ABSOLUTE tolerances (FIT_FATOL =
# 1e-4 on the pinball loss), so a trace whose amplitude is ~0.1 "converges"
# immediately at the warm start. The spectrometer traces this was tuned on peak
# in the 1e4 range; matching that keeps the tolerances meaningful. A pure gain
# cannot change the fitted phase.
COUNTS_PEAK = 1.0e4

C_MM_PER_PS = 0.299792458   # speed of light


mpl.rcParams.update({
    "axes.grid": True,
    "grid.linewidth": 0.3,
    "grid.color": "grey",
    "lines.linewidth": 0.8,
})


def load_xcorr_csv(path: Path, delay_center_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Read an XCORR csv into (probe delay in ps, mean intensity).

    Rows may be jagged (a partial/interrupted acquisition leaves a short final
    row); keep only rows that have the full, most-common column count.
    """
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    width = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    data = np.array([r for r in rows if len(r) == width], dtype=float)

    stage_mm = data[:, 0]
    intensity = data[:, 1:].mean(axis=1)

    # Double-pass retroreflector: t = 2 * (pos - centre) / c.
    delay_ps = 2.0 * (stage_mm - delay_center_mm) / C_MM_PER_PS
    order = np.argsort(delay_ps)
    return delay_ps[order], intensity[order]


def nyquist_ghz(delay_ps: np.ndarray) -> float:
    """Highest fringe frequency the delay sampling can represent."""
    return 1e3 / (2.0 * float(np.median(np.diff(delay_ps))))


class DelayMap:
    """Affine delay(ps) <-> pseudo-wavelength(nm) map, plus the frequency map."""

    def __init__(self, delay_ps: np.ndarray, span_nm: float = SPAN_NM, lam0: float = LAM0):
        t_lo, t_hi = float(delay_ps.min()), float(delay_ps.max())
        self.t_mid = 0.5 * (t_lo + t_hi)
        self.lam0 = lam0
        self.scale = span_nm / (t_hi - t_lo)        # nm per ps

    def to_nm(self, t_ps):
        return self.lam0 + (np.asarray(t_ps, float) - self.t_mid) * self.scale

    def to_ps(self, lam_nm):
        return self.t_mid + (np.asarray(lam_nm, float) - self.lam0) / self.scale

    def to_ghz(self, f_cyc_per_nm):
        """cycles/nm on the mapped axis -> GHz in the lab's delay axis."""
        return 1e3 * np.asarray(f_cyc_per_nm, float) * self.scale


def fit_xcorr(delay_ps: np.ndarray, intensity: np.ndarray):
    """Run the fringe-fit pipeline on an XCORR trace. Returns (R, mapping, gain).

    Pipeline choice: the scan-free branch with NO truncation detection.
    `detect_truncation`/`knife_edge_cut` are the one part of fringe_core that is
    genuinely calibrated in nm (window widths, minimum dead-run lengths), and
    they exist to find a knife-edge clip in the spectrometer arm -- something an
    XCORR delay scan does not have. Crop by hand with DELAY_CROP_PS instead.
    """
    mapping = DelayMap(delay_ps)
    gain = COUNTS_PEAK / float(np.max(intensity))

    R = fc.analyze(mapping.to_nm(delay_ps), intensity * gain,
                   anchor=None,            # no continuum outside the scan to pin the offset
                   scanfree=True, trunc_method="none")
    return R, mapping, gain


def plot_fit(R, mapping, gain, delay_ps, intensity, title):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [2, 1]}
    )

    # Top: raw trace + the fitted reconstruction.
    ax1.plot(delay_ps, intensity, color="0.55", lw=0.8, label="XCORR (raw)")
    ax1.set_ylabel("Intensity (arb.)")
    ax1.tick_params(axis="x", labelbottom=False)

    if R.get("csig") is None:
        ax1.set_title(f"no fit: status={R['status']} -- {R.get('msg', '')}", fontsize=9)
        ax1.legend(loc="upper right", fontsize=8)
        ax2.set_xlabel("Probe delay (ps)")
        return fig

    t_core = mapping.to_ps(R["x"])
    u = R["x"] - R["l0"]
    y_model = fc.signal_model(R["csig"], u, R["mid"], R["half"]) / gain
    env_hi = (R["mid"] + R["half"]) / gain
    env_lo = (R["mid"] - R["half"]) / gain

    # Grey out what the contrast crop / end-trim excluded, so the fit span is
    # unambiguous -- the model is only claimed on the core.
    for lo, hi in ((delay_ps[0], t_core[0]), (t_core[-1], delay_ps[-1])):
        if hi > lo:
            ax1.axvspan(lo, hi, color="0.85", alpha=0.5, zorder=0)

    ax1.plot(t_core, env_hi, color="tab:orange", lw=0.7, ls="--", alpha=0.8,
             label="fitted envelope")
    ax1.plot(t_core, env_lo, color="tab:orange", lw=0.7, ls="--", alpha=0.8)
    ax1.plot(t_core, y_model, color="crimson", lw=1.0,
             label=r"reconstruction  mid + half$\cdot\cos\Phi$")
    ax1.legend(loc="upper right", fontsize=8)

    # Bottom: instantaneous frequency of that reconstruction.
    f_ghz = np.abs(mapping.to_ghz(R["f_model"]))
    ax2.plot(t_core, np.abs(mapping.to_ghz(R["f_inst"])), color="0.7", lw=0.7,
             label="Hilbert |f| (raw)")
    ax2.plot(t_core, f_ghz, color="tab:green", lw=1.6, label="fitted |f|")
    if R["null_wl"] is not None:
        t_null = float(mapping.to_ps(R["null_wl"]))
        ax2.axvline(t_null, color="purple", ls="--", lw=1.0, label=f"null @ {t_null:.0f} ps")
    # The delay step sets a hard sampling limit: above it the fringe is aliased and
    # both the Hilbert seed and the fit are reading noise, however smooth they look.
    nyq = nyquist_ghz(delay_ps)
    ax2.axhline(nyq, color="0.4", ls=":", lw=1.0, label=f"Nyquist {nyq:.0f} GHz")
    ax2.set_xlabel("Probe delay (ps)")
    ax2.set_ylabel("Frequency (GHz)")
    top = max(f_ghz.max(), nyq) * 1.15
    ax2.set_ylim(0, FREQ_YLIM_GHZ if FREQ_YLIM_GHZ is not None else top)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"{title}\norder q={R['order']}  r2_fringe={R['r2_fringe']:.3f}  "
        f"rms_frac={R['rms_frac']:.3f}  f = {f_ghz.min():.1f}-{f_ghz.max():.1f} GHz"
        f"{'' if R['shape_ok'] else '  (shape unverified)'}",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.88, left=0.10, right=0.96, bottom=0.10)
    return fig


def main() -> None:
    delay_ps, intensity = load_xcorr_csv(CSV_PATH, DELAY_CENTER_MM)
    if DELAY_CROP_PS is not None:
        m = (delay_ps >= DELAY_CROP_PS[0]) & (delay_ps <= DELAY_CROP_PS[1])
        delay_ps, intensity = delay_ps[m], intensity[m]

    R, mapping, gain = fit_xcorr(delay_ps, intensity)

    print(f"{CSV_PATH.name}: {len(delay_ps)} pts, "
          f"{delay_ps[0]:.1f} to {delay_ps[-1]:.1f} ps, "
          f"map {mapping.scale:.4f} nm/ps, Nyquist {nyquist_ghz(delay_ps):.0f} GHz")
    print(f"  status={R['status']}", end="")
    if R.get("csig") is not None:
        f_ghz = np.abs(mapping.to_ghz(R["f_model"]))
        print(f" order={R['order']} r2_fringe={R['r2_fringe']:.3f} "
              f"rms_frac={R['rms_frac']:.3f} trust_ok={R['trust_ok']} "
              f"shape_ok={R['shape_ok']}\n"
              f"  fit span {mapping.to_ps(R['fit_span'][0]):.1f} to "
              f"{mapping.to_ps(R['fit_span'][1]):.1f} ps, "
              f"f = {f_ghz.min():.2f}-{f_ghz.max():.2f} GHz")
    else:
        print(f" -- {R.get('msg', '')}")

    plot_fit(R, mapping, gain, delay_ps, intensity,
             f"XCORR fringe fit: {CSV_PATH.parent.name}/{CSV_PATH.name}")
    plt.show()


if __name__ == "__main__":
    main()
