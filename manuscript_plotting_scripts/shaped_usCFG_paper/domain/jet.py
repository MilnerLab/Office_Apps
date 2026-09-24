"""Loading, conventions and reductions for the 2026-09 gas-jet CS2 measurements (ported
from the paper repo's ``analysis/jet/``: ``jetdata.py``, and the computation halves of
``run_oscillations.py`` and ``run_truncation.py``).

Data root
---------
``config.JET_ROOT`` with two campaigns:

``Jet Centrifuge Oscillations/``  (2026-09-03)
    Nine repeats of one probe-delay scan, phase **stabilized**.  Each
    ``*_ScanFile.dat`` is already reduced by the acquisition VI to one row per
    delay point:

        0 unix-ish timestamp
        1 probe delay (ps)          <- signed, zero-delay stage 64.500 mm
        2 <cos^2 theta_2D>
        3 uncertainty on col 2
        4 ions per frame
        5 copy of col 2

``Jet Truncation/{20260904,20260907}/Scan*_CFG/``
    Probe-delay scans with the centrifuge polarization **mechanically averaged**
    (the stabilization half-wave plate spun at 0.05 rev/s), one file per delay
    point per repeat, named ``...DLY_<pos>mm.dat``.  These are *raw* VMI hit
    lists, three columns:

        0 camera frame index within the file
        1 x (px)
        2 y (px)

    so <cos^2 theta_2D> has to be formed here.  See `c2t`.

``Jet Truncation/20260904/XCORR_20260903_jet_accompany_scan.h5``
    The cross-correlation taken at the oscillation scan's own settings.  It is
    read with the 2026-08-25 XCORR pipeline (``domain/xcorr_fit.py``),
    unchanged; see `load_accompanying_xcorr`.

Conventions
-----------
Delay.  The probe stage is double-passed, so a stage position ``p`` (mm)
corresponds to a probe delay ``2 (p - p0) / c`` with ``p0 = 64.500`` mm.  This
reproduces the -450 ps ... +1501 ps grid the scan files were written on.

Image.  Centre and aspect from the run sheets: ``(x0, y0) = (162, 220)`` px,
``x`` scaled by 0.85, radial mask ``30 < r < 120`` px.

Alignment axis.  With the polarization phase averaged there is no preferred
*direction* left in the plane of rotation, but the plane itself is fixed: the
centrifuge propagates in the detector plane, so a molecule confined to the
rotation plane still projects onto the detector as an axis.  The observable is
therefore the second Fourier moment

    z = < exp(2 i theta) >

read along one fixed laboratory axis ``psi``:

    <cos^2 theta_2D> = 1/2 + Re[ z exp(-2 i psi) ] / 2 .

``psi`` is not assumed.  `fit_alignment_axis` takes it from the data as the
axis along which the centrifuge *changes* the image, i.e. the argument of
``z(t) - z(t_ref)`` accumulated over the scan, with ``t_ref`` the earliest
(pre-centrifuge) delay.  All four truncation scans return psi = -12 deg to
better than a degree, and the residual ``|z(t_ref)| ~ 0.13`` at -49 deg to that
axis is a static detector anisotropy that survives as a small constant offset,
not as signal.

Oscillations (``fig_jet_oscillations``)
---------------------------------------
Panel (a) is the averaged <cos^2 theta_2D> trace (nine repeats of one probe
delay scan, phase stabilized).  Panel (b) is the short-time Fourier transform
of its oscillating part, with the fringe frequency of the accompanying
cross-correlation drawn over it, and the same fringe frequency as the
calibration constants predict it with nothing fitted (coefficients from
``jet_prediction.json``, written by ``domain/jet_prediction.py``, so run that
first).  Panel (c) is that cross-correlation.

**The two frequencies are the same quantity.**  The xcorr sees the corkscrew
through a cos^2 projection and the molecules align headlessly, so both run at
twice the centrifuge frequency.  Nothing is rescaled to make them agree.

*The one fitted number.*  The two scans set their delay zeros independently --
the xcorr file's ``probe_offset_mm`` is just where the sweep was started, so
its only intrinsic origin is the fitted envelope centre -- and a constant
offset between the axes is therefore unavoidable.  It is measured here by least
squares against the spectrogram ridge and printed on every run.  Slope and
curvature are not touched.

*Resampling.*  The delay grid is deliberately non-uniform -- the acquisition
steps finely where the centrifuge is fast -- so the STFT is taken on a uniform
grid at the finest step present (1.57 ps), reached by cubic spline.

*Where the axis stops.*  The step was planned for a slower beat than the
centrifuge actually produced, so the ORIGINAL grid's local Nyquist frequency is
surpassed at both ends of the record: below about -330 ps, where 50 ps steps
have to carry a beat that has folded through zero and is running backwards, and
above about +320 ps, where the beat outruns the finest step.  The figure is cut
where the beat reaches NYQ_FRAC of the local Nyquist, computed from the xcorr
law, which is sampled 500x more finely and knows nothing about the jet grid.
Everything outside that window is aliased and is not shown.

*Window.*  The chirp, not the record, limits the resolution here.  The beat
sweeps at ~0.5 GHz/ps, so a window of T sweeps 0.5T GHz against a transform
width 1/T; the two are balanced at T ~ 45 ps and a window a small multiple of
that is the usable range.  90 ps is used.  The slow envelope is removed with a
moving average of the same length, which leaves everything above ~1/T alone.

Truncation (``fig_jet_truncation``)
-----------------------------------
Released rotation vs truncation position (mechanically phase-averaged): the four
scans of `TRUNCATION_SCANS` read along the fitted axis ``psi``, with frame-level
error bars.

Types
-----
The public functions take and return base_core quantities (``Time``, ``Length``,
``Frequency``, ``Angle``, ``Measurement``, ``ScanDataBase``); each converts to numpy
once, at its boundary.  Laws such as the beat frequency are `TypedLaw` objects.  The
STFT steps inside `oscillations` (`_spectrogram`, `_ridge`, `_align`) work on numpy
arrays throughout; the comment above them gives the measured cost of typing the
beat law inside `_align`.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter, uniform_filter1d

from _data_io.dat_loader import load_time_scan
from base_core.lab_specifics.base_models import Measurement, ScanDataBase
from base_core.lab_specifics.helpers import calculate_time_delay
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Length, Time
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import xcorr_fit as P
from manuscript_plotting_scripts.shaped_usCFG_paper.domain.typed import TypedLaw

STAGE_ZERO_MM = 64.500             # probe stage position of zero delay

X0, Y0 = 162.0, 220.0              # VMI centre (px), from the run sheets
SCALE_X = 0.85                     # aspect correction applied to x
R_MIN, R_MAX = 30.0, 120.0         # radial mask (px, after scaling)

DATA_ROOT = config.JET_ROOT
OSC_DIR = os.path.join(DATA_ROOT, "Jet Centrifuge Oscillations")
TRUNC_ROOT = os.path.join(DATA_ROOT, "Jet Truncation")

#: the four mechanically-averaged scans, in the order they are plotted
TRUNCATION_SCANS = [
    ("20260904/Scan3_CFG", "Full centrifuge", None),
    ("20260907/Scan2_CFG", "Prism 15.0 mm", 15.0),
    ("20260904/Scan4_CFG", "Prism 14.5 mm", 14.5),
    ("20260907/Scan1_CFG", "Prism 13.5 mm", 13.5),
]

WIN_PS = 90.0           # STFT window (Blackman), full width
STEP_PS = 3.0           # STFT hop
NFFT = 8192
FMAX_GHZ = 270.0        # top of the displayed band
ZOOM = (120.0, 220.0)   # inset window on panels (a) and (c)
NYQ_FRAC = 0.95         # cut the axis where the beat reaches this x local Nyquist
FMIN_FIT_GHZ = 20.0     # ridge below this is envelope leakage, not the beat
PRED_JSON = config.TEMP_DIR / "jet_prediction.json"


# --------------------------------------------------------------------------
# types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AveragedScan(ScanDataBase):
    """A probe-delay scan averaged over repeats: ``measured_values`` are
    Measurement(mean, standard error of the repeats)."""
    n_repeats: int = 0


@dataclass(frozen=True)
class MomentScan:
    """One mechanically-averaged truncation scan reduced to the second Fourier moment.

    ``z`` (complex, one per delay) and ``per_frame`` (complex per-frame moments, one
    array per delay) have no base_core type and stay numpy.
    """
    delays: list[Time]
    z: np.ndarray
    per_frame: list[np.ndarray]
    n_ions: list[int]


@dataclass(frozen=True)
class TruncationTrace(ScanDataBase):
    """One truncation scan read along the alignment axis: ``measured_values`` are
    Measurement(<cos^2 theta_2D>, frame-level standard error)."""
    label: str = ""
    prism: Length | None = None        # truncation prism position; None = full centrifuge
    rel: str = ""                      # scan directory under ``Jet Truncation/``
    n_ions: list[int] | None = None    # ions inside the radial mask, per delay


# --------------------------------------------------------------------------
# delay bookkeeping
# --------------------------------------------------------------------------

def stage_to_delay(pos: Length) -> Time:
    """Probe stage position -> probe delay, double pass."""
    return calculate_time_delay(pos, Length(STAGE_ZERO_MM, Prefix.MILLI))


_DLY_RE = re.compile(r"DLY_+(-?\d+)p(\d+)mm\.dat$")


def _stage_of(path: str) -> Length:
    m = _DLY_RE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"cannot parse stage position from {path!r}")
    return Length(float(f"{m.group(1)}.{m.group(2)}"), Prefix.MILLI)


# --------------------------------------------------------------------------
# the stabilized oscillation scans (already reduced)
# --------------------------------------------------------------------------

def load_oscillations() -> AveragedScan:
    """The nine repeats averaged: Measurement(<cos^2 theta_2D>, standard error) per delay.

    The nine scans share one delay grid, which is asserted rather than
    interpolated.  The uncertainty is the scatter of the repeats about their
    mean (it agrees with the per-scan column-3 uncertainty divided by 3).
    """
    files = sorted(glob.glob(os.path.join(OSC_DIR, "*_ScanFile.dat")))
    if not files:
        raise FileNotFoundError(OSC_DIR)
    # The lab loader carries an IonDataAnalysisConfig for provenance only; these files
    # are already reduced by the acquisition VI, so there is none to give it.
    scans = [load_time_scan(Path(f), None) for f in files]
    ts = np.array([[d.value(Prefix.PICO) for d in s.delays] for s in scans])
    if not np.allclose(ts, ts[0]):
        raise ValueError("the oscillation scans do not share a delay grid")
    c = np.array([[m.value for m in s.measured_values] for s in scans])
    mean, sem = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(files))
    return AveragedScan(delays=list(scans[0].delays),
                        measured_values=[Measurement(float(v), float(e)) for v, e in zip(mean, sem)],
                        run_id=None, n_repeats=len(files))


# --------------------------------------------------------------------------
# the mechanically-averaged truncation scans (raw VMI hits)
# --------------------------------------------------------------------------

def _group_by_stage(scan_dir: str):
    g = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(scan_dir, "*.dat"))):
        g[_stage_of(f)].append(f)
    return dict(sorted(g.items()))


def _angles(hits: np.ndarray):
    """Masked in-plane angles of a hit list, and the frame key of each hit."""
    x = (hits[:, 1] - X0) * SCALE_X
    y = hits[:, 2] - Y0
    r = np.hypot(x, y)
    m = (r > R_MIN) & (r < R_MAX)
    return np.arctan2(y[m], x[m]), hits[m, 0]


def _load_point(files):
    """Second Fourier moment z and per-frame moments for one delay point."""
    th, frame = [], []
    for i, f in enumerate(files):
        t, fr = _angles(np.loadtxt(f))
        th.append(t)
        frame.append(fr + 1000 * i)          # keep repeats' frames distinct
    th = np.concatenate(th)
    frame = np.concatenate(frame).astype(np.int64)
    e = np.exp(2j * th)
    # per-frame means, for a frame-level (not ion-level) error bar
    idx, inv = np.unique(frame, return_inverse=True)
    cnt = np.bincount(inv)
    per_frame = (np.bincount(inv, e.real) + 1j * np.bincount(inv, e.imag)) / cnt
    return e.mean(), per_frame, th.size


def load_truncation_scan(rel_dir: str) -> MomentScan:
    """One truncation scan: delays, second moment z, per-frame moments and ion counts."""
    g = _group_by_stage(os.path.join(TRUNC_ROOT, rel_dir))
    t, z, pf, n = [], [], [], []
    for pos, files in g.items():
        zi, pfi, ni = _load_point(files)
        t.append(stage_to_delay(pos))
        z.append(zi)
        pf.append(pfi)
        n.append(int(ni))
    return MomentScan(delays=t, z=np.array(z), per_frame=pf, n_ions=n)


def fit_alignment_axis(scans: Sequence[MomentScan]) -> Angle:
    """Laboratory alignment axis psi from the centrifuge-induced change.

    For each scan the earliest delay is the pre-centrifuge reference; the axis
    is the argument of the summed change, which weights the delays that carry
    signal without any hand-set window.
    """
    total = 0.0 + 0.0j
    for s in scans:
        t = np.array([d.value(Prefix.PICO) for d in s.delays])
        ref = s.z[np.argmin(t)]
        total += np.sum(s.z - ref)
    # psi is already in (-pi/2, pi/2]; wrap=False keeps it bit for bit
    return Angle(0.5 * np.angle(total), AngleUnit.RAD, wrap=False)


def c2t(z: np.ndarray, psi: Angle) -> np.ndarray:
    """<cos^2 theta_2D> along the axis psi, from complex second moments z."""
    return 0.5 + 0.5 * np.real(np.asarray(z) * np.exp(-2j * psi.Rad))


# --------------------------------------------------------------------------
# the cross-correlation taken alongside the oscillation scan
# --------------------------------------------------------------------------

#: One probe sweep at the settings the oscillation scan ran at -- DA = 20.42 mm,
#: GA = -75 mm, i.e. L = -103.12 mm -- taken 2026-09-03/04.  It sits under the
#: truncation tree because that is where it was written; nothing about it is
#: truncation-related.
XCORR_H5 = os.path.join(TRUNC_ROOT, "20260904",
                        "XCORR_20260903_jet_accompany_scan.h5")


def load_accompanying_xcorr() -> tuple[ScanDataBase, TypedLaw, P.FitResult]:
    """Return ``(scan, beat, fit)`` for that sweep.

    ``scan`` holds Measurement(v_mean_pos, v_std) against the probe delay
    referenced to the **fitted envelope centre**, which is the only origin the
    file defines: its ``probe_offset_mm`` is just where the sweep was started.
    The jet scans reference their own stage zero, so the two axes differ by a
    constant, which `_align` measures.

    ``beat`` (delay about the envelope centre -> ``Frequency``) is the fringe
    frequency of the nine-parameter fit.  It is in the **same units as the jet's
    alignment beat**: the cross-correlation sees the corkscrew through a cos^2
    projection and the molecules align headlessly, so both run at twice the
    centrifuge frequency.  This is the whole point of the comparison and no
    factor is applied anywhere to make it work.
    """
    sc = P.load_scans(XCORR_H5)[0]
    fit = P.full_fit(sc)
    if not fit.ok:
        raise RuntimeError(f"xcorr fit failed: {fit.status}")
    mu = fit["mu"]
    f_cfg = fit.f_uscfg.numpy
    beat = TypedLaw(lambda u: 2.0 * f_cfg(u), Prefix.PICO, Frequency, Prefix.GIGA)
    t, _, _ = sc.to_numpy()
    scan = ScanDataBase(delays=[Time(u, Prefix.PICO) for u in t - mu],
                        measured_values=list(sc.measured_values), run_id=None)
    return scan, beat, fit


# --------------------------------------------------------------------------
# oscillations: spectrogram, ridge and the xcorr law against it
# --------------------------------------------------------------------------

# The STFT steps below are private steps of `oscillations`, on its uniform numpy grid;
# `oscillations` is their typed boundary. `_align` evaluates the beat law ~10^6 times, so it
# takes the law's numpy form (`TypedLaw.numpy`).
# numpy here: base_core types cost 15x (baseline 0.139 s, typed 2.13 s)

def _spectrogram(tu, s, dt):
    """STFT, zero padded by half a window so the columns span the whole scan."""
    n = int(round(WIN_PS / dt))
    n += n % 2
    w = np.blackman(n)
    hop = max(1, int(round(STEP_PS / dt)))
    s = np.concatenate([np.zeros(n // 2), s, np.zeros(n // 2)])
    t0 = tu[0] - (n // 2) * dt
    centres, cols = [], []
    for i in range(0, len(s) - n, hop):
        seg = s[i:i + n]
        cols.append(np.abs(np.fft.rfft((seg - seg.mean()) * w, NFFT)))
        centres.append(t0 + (i + n / 2) * dt)
    f = np.fft.rfftfreq(NFFT, dt) * 1e3          # ps^-1 -> GHz
    return np.array(centres), f, np.array(cols).T


def _ridge(f, S, fmin=8.0):
    """Interpolated spectral peak of each STFT column, lightly median filtered."""
    m = (f > fmin) & (f < FMAX_GHZ)
    fm, Sm = f[m], S[m]
    df = fm[1] - fm[0]
    out = np.empty(Sm.shape[1])
    for j in range(Sm.shape[1]):
        col = Sm[:, j]
        k = int(np.argmax(col))
        if 0 < k < len(col) - 1:
            a, b, c = np.log(col[k - 1:k + 2] + 1e-30)
            out[j] = fm[k] + 0.5 * (a - c) / (a - 2 * b + c) * df
        else:
            out[j] = fm[k]
    return median_filter(out, size=9, mode="nearest")


def _window(tc, nyq, beat, shift):
    """Largest run of STFT columns around t = 0 that the delay grid supports."""
    ok = np.abs(beat(tc - shift)) <= NYQ_FRAC * nyq
    k = int(np.argmin(np.abs(tc)))
    if not ok[k]:
        raise RuntimeError("the delay grid does not support the beat at t = 0")
    lo = hi = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return float(tc[lo]), float(tc[hi])


def _align(tc, fr, nyq, beat):
    """Delay-origin offset between the xcorr axis and the jet axis, ps.

    One parameter, by least squares of |beat| against the ridge over the
    columns the grid supports.  Iterated because the usable window itself
    depends on the offset; it settles in two or three passes.
    """
    grid = np.arange(-120.0, 120.0, 0.05)
    shift = 0.0
    lo, hi = _window(tc, nyq, beat, shift)
    for _ in range(6):
        lo, hi = _window(tc, nyq, beat, shift)
        m = (tc >= lo) & (tc <= hi) & (np.abs(beat(tc - shift)) > FMIN_FIT_GHZ)
        r = [np.sqrt(np.mean((np.abs(beat(tc[m] - s)) - fr[m]) ** 2)) for s in grid]
        new = float(grid[int(np.argmin(r))])
        done = abs(new - shift) < 0.05
        shift = new
        if done:
            break
    lo, hi = _window(tc, nyq, beat, shift)
    m = (tc >= lo) & (tc <= hi) & (np.abs(beat(tc - shift)) > FMIN_FIT_GHZ)
    rms = float(np.sqrt(np.mean((np.abs(beat(tc[m] - shift)) - fr[m]) ** 2)))
    return shift, (lo, hi), rms, m


def _peak(tt, yy):
    """Dominant frequency of one trace inside the inset window, GHz.

    A direct check that the two panels' insets show the same thing: both are
    resampled to 0.2 ps, detrended, Hann windowed and transformed.
    """
    m = (tt >= ZOOM[0]) & (tt <= ZOOM[1])
    tt, yy = tt[m], yy[m]
    g = np.arange(tt[0], tt[-1], 0.2)
    v = CubicSpline(tt, yy)(g)
    v = (v - np.polyval(np.polyfit(g, v, 3), g)) * np.hanning(len(g))
    F = np.abs(np.fft.rfft(v, 1 << 16))
    f = np.fft.rfftfreq(1 << 16, 0.2) * 1e3
    k = (f > 20) & (f < FMAX_GHZ + 100)
    return float(f[k][np.argmax(F[k])])


def predicted_beat() -> TypedLaw:
    """The calibration's prediction of the accompanying xcorr's beat, ``2 f_CFG(u)``
    (delay about the envelope centre -> ``Frequency``), from ``jet_prediction.json``:
    f_CFG(u) coefficients about the envelope centre, folded to f(0) > 0 as the fit is.
    Drawn over panel (b) of ``fig_jet_oscillations``."""
    pc = json.load(open(PRED_JSON))["predicted"]
    return TypedLaw(lambda u: 2.0 * (pc[0] + pc[1] * u + pc[2] * u * u),
                    Prefix.PICO, Frequency, Prefix.GIGA)


def oscillations() -> dict:
    """The whole oscillation reduction. Returns, keyed by name:

    ``trace``   AveragedScan: the averaged jet trace, Measurement(<cos^2 theta_2D>, sem)
                against delay (panel (a)); ``trace.n_repeats`` repeats
    ``dt``      Time: the finest step of the delay grid (the STFT's uniform step)
    ``tc``      list[Time]: the spectrogram's column centres (panel (b))
    ``f``       list[Frequency]: the spectrogram's frequencies
    ``S``       numpy array (len(f), len(tc)): STFT magnitude.  A 2-D image stays
                numpy: it is drawn with imshow and has no per-element meaning.
    ``fr``      list[Frequency]: the ridge (spectral peak) of each column
    ``nyq``     list[Frequency]: the ORIGINAL grid's local Nyquist at each column
    ``xcorr``   ScanDataBase: the accompanying xcorr, delay about its envelope centre
                (panel (c) draws it at delay + ``shift``)
    ``beat``    TypedLaw, delay about the envelope centre -> Frequency: its beat law
    ``xfit``    xcorr_fit.FitResult: its nine-parameter fit
    ``shift``   Time: the delay-origin offset; the xcorr's envelope centre on the jet axis
    ``lo, hi``  Time: the window the grid supports; the figure's x range
    ``rms``     Frequency: law-vs-ridge rms over the window
    ``m``       numpy bool array over ``tc``: the columns used in the offset fit
    ``pbeat``   TypedLaw: the predicted beat (`predicted_beat`)
    """
    trace = load_oscillations()
    t = np.array([d.value(Prefix.PICO) for d in trace.delays])
    y = np.array([v.value for v in trace.measured_values])
    dt = float(np.diff(t).min())

    tu = np.arange(t.min(), t.max(), dt)
    yu = CubicSpline(t, y)(tu)
    s = yu - uniform_filter1d(yu, int(round(WIN_PS / dt)) | 1, mode="nearest")

    tc, f, S = _spectrogram(tu, s, dt)
    fr = _ridge(f, S)

    # local Nyquist of the ORIGINAL grid, carried onto the STFT centres
    nyq = 1e3 / (2 * np.interp(tc, 0.5 * (t[1:] + t[:-1]), np.diff(t)))

    xcorr, beat, xfit = load_accompanying_xcorr()
    shift, (lo, hi), rms, m = _align(tc, fr, nyq, beat.numpy)
    ps = lambda v: Time(v, Prefix.PICO)
    ghz = lambda v: Frequency(v, Prefix.GIGA)
    return dict(trace=trace, dt=ps(dt), tc=[ps(v) for v in tc], f=[ghz(v) for v in f], S=S,
                fr=[ghz(v) for v in fr], nyq=[ghz(v) for v in nyq], xcorr=xcorr, beat=beat,
                xfit=xfit, shift=ps(shift), lo=ps(lo), hi=ps(hi), rms=ghz(rms), m=m,
                pbeat=predicted_beat())


def run_oscillations(out_dir: Path) -> None:
    """Reduce the oscillation scans; write ``oscillations.csv``, ``oscillations_ridge.csv``
    and ``accompanying_xcorr.csv`` (the xcorr on the jet axis)."""
    k = oscillations()
    ps = lambda xs: np.array([x.value(Prefix.PICO) for x in xs])
    ghz = lambda xs: np.array([x.value(Prefix.GIGA) for x in xs])
    trace, xcorr, xfit, m = k["trace"], k["xcorr"], k["xfit"], k["m"]
    t, nrep, dt = ps(trace.delays), trace.n_repeats, k["dt"].value(Prefix.PICO)
    y = np.array([v.value for v in trace.measured_values])
    sem = np.array([v.error for v in trace.measured_values])
    tc, fr, nyq = ps(k["tc"]), ghz(k["fr"]), ghz(k["nyq"])
    ux, vx = ps(xcorr.delays), np.array([v.value for v in xcorr.measured_values])
    shift, lo, hi = (k[n].value(Prefix.PICO) for n in ("shift", "lo", "hi"))
    rms = k["rms"].value(Prefix.GIGA)
    beat, pbeat = k["beat"].numpy, k["pbeat"].numpy

    print(f"{len(t)} delays, {t.min():.0f}..{t.max():.0f} ps, "
          f"step {np.diff(t).max():.1f}..{dt:.2f} ps, {nrep} repeats")
    print("xcorr: envelope FWHM %.0f ps, visibility %.2f, fringe correlation %.3f"
          % (2.3548 * xfit["sigma"], xfit["vis"], xfit.rho2))
    print("delay-origin offset %+.1f ps: the xcorr envelope centre sits there on "
          "the jet axis" % shift)
    print("xcorr law vs jet ridge, %.0f..%.0f ps: rms %.1f GHz over a %.0f GHz sweep"
          % (lo, hi, rms, abs(beat(hi - shift)) - abs(beat(lo - shift))))

    mp = m & (beat(tc - shift) > FMIN_FIT_GHZ)      # unfolded branch only
    pj = np.polyfit(tc[mp], fr[mp], 2)
    ug = np.linspace(tc[mp].min() - shift, hi - shift, 400)
    px = np.polyfit(ug + shift, beat(ug), 2)
    print("jet ridge  beat/GHz = %.4g t^2 %+.5f t %+.2f" % tuple(pj))
    print("xcorr      beat/GHz = %.4g t^2 %+.5f t %+.2f" % tuple(px))
    print("  slopes differ by %+.1f%%; f_CFG at t = 0 is %.1f GHz (jet) vs %.1f "
          "GHz (xcorr)" % (100 * (px[1] - pj[1]) / pj[1], pj[2] / 2, px[2] / 2))

    pm = m & (np.abs(pbeat(tc - shift)) > FMIN_FIT_GHZ)
    print("prediction vs jet ridge: rms %.1f GHz; vs xcorr law: rms %.1f GHz"
          % (np.sqrt(np.mean((np.abs(pbeat(tc[pm] - shift)) - fr[pm]) ** 2)),
             np.sqrt(np.mean((np.abs(pbeat(tc[pm] - shift))
                              - np.abs(beat(tc[pm] - shift))) ** 2))))

    zero = shift + max(np.roots([3 * xfit["c3"], 2 * xfit["c2"], xfit["c1"]]))
    print("beat zero crossing at %+.0f ps; at the end of the window it is %.0f GHz, "
          "f_CFG %.0f GHz" % (zero, beat(hi - shift), beat(hi - shift) / 2))
    print("in the %g..%g ps inset both traces peak at %.1f GHz (jet) and %.1f GHz "
          "(xcorr)" % (ZOOM[0], ZOOM[1], _peak(t, y), _peak(ux + shift, vx)))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "oscillations.csv"),
               np.column_stack([t, y, sem]), delimiter=",",
               header="delay_ps,c2t,sem", comments="")
    np.savetxt(os.path.join(out_dir, "oscillations_ridge.csv"),
               np.column_stack([tc, fr, nyq, np.abs(beat(tc - shift))]),
               delimiter=",", comments="",
               header="delay_ps,ridge_beat_GHz,local_nyquist_GHz,xcorr_beat_GHz")
    np.savetxt(os.path.join(out_dir, "accompanying_xcorr.csv"),
               np.column_stack([ux + shift, vx]), delimiter=",",
               header="delay_ps,v_mean_pos", comments="")


# --------------------------------------------------------------------------
# truncation: released rotation vs truncation position
# --------------------------------------------------------------------------

def sem_from_frames(per_frame: np.ndarray, psi: Angle) -> float:
    """Frame-level standard error on <cos^2 theta_2D>, from complex per-frame moments."""
    v = 0.5 * np.real(per_frame * np.exp(-2j * psi.Rad))
    return v.std(ddof=1) / np.sqrt(v.size)


def truncation_traces() -> tuple[Angle, list[TruncationTrace]]:
    """The four `TRUNCATION_SCANS` read along the fitted axis. Returns ``(psi, traces)``:
    ``psi`` the fitted alignment axis, ``traces`` the `TruncationTrace` of each scan in
    plotting order (delays; Measurement(<cos^2 theta_2D>, frame-level error); label;
    prism position, None for the full centrifuge; ions per delay)."""
    scans = []
    for rel, label, prism in TRUNCATION_SCANS:
        ms = load_truncation_scan(rel)
        scans.append((ms, label, prism, rel))
        print(f"{label:18s} {rel:22s} {len(ms.delays):2d} delays, {sum(ms.n_ions):,} ions")

    psi = fit_alignment_axis([s[0] for s in scans])
    print(f"alignment axis psi = {psi.Deg:+.2f} deg")

    traces = []
    for ms, label, prism, rel in scans:
        y = c2t(ms.z, psi)
        e = [sem_from_frames(p, psi) for p in ms.per_frame]
        traces.append(TruncationTrace(
            delays=ms.delays,
            measured_values=[Measurement(float(yi), float(ei)) for yi, ei in zip(y, e)],
            run_id=None, label=label, rel=rel, n_ions=ms.n_ions,
            prism=None if prism is None else Length(prism, Prefix.MILLI)))
    return psi, traces


def run_truncation(out_dir: Path) -> None:
    """Reduce the four truncation scans; write ``truncation.csv``."""
    psi, traces = truncation_traces()
    rows = []
    for tr in traces:
        prism = np.nan if tr.prism is None else tr.prism.value(Prefix.MILLI)
        for d, v, ni in zip(tr.delays, tr.measured_values, tr.n_ions):
            rows.append((tr.rel, tr.label, prism, d.value(Prefix.PICO), v.value, v.error, ni))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv = os.path.join(out_dir, "truncation.csv")
    with open(csv, "w") as fh:
        fh.write("scan,label,prism_mm,delay_ps,c2t,sem,n_ions\n")
        for r in rows:
            fh.write("%s,%s,%s,%.3f,%.6f,%.6f,%d\n" % r)
    print("wrote", csv, f"(psi = {psi.Deg:+.3f} deg)")
