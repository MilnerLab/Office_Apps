"""Spectral-interferogram analysis of the 2026-08-25 two-scan dataset.

The companion to `xcorr_fit.py`. Every setpoint of both runs carries the *same*
measurement twice: the scope trace against probe delay (that is `xcorr_fit.py`), and
the spectrometer trace against wavelength (this file). The two arms interfere in the
spectrometer, so a single averaged spectrum is a **spectral interferogram** whose
fringe phase is the difference of the two arms' spectral phases,

    Delta psi(w) = phi0 + T0 u + (1/2) psi2 u^2 + (1/6) psi3 u^3,     u = w - W0_REF

and whose group delay is `T(w) = d Delta psi / dw = T0 + psi2 u + ...`.

Why this settles things `xcorr_fit.py` cannot:

* **`T0` is the true arm delay**, measured with no stage calibration at all. It tests
  a `dt` stage-zero offset directly.
* **`psi2` is the arms' chirp mismatch, signed.** The xcorr readout `Delta f` comes
  from a Hilbert transform and is therefore folded -- the *sign* of the mismatch is
  unobservable there. Here it is not.
* The link to the fringe chirp needs no convention at all. With `t_j(w)` the group
  delay of arm `j`, `dt_j/dw = 1/(dw_j/dt)`, so

      psi2 = -(dOmega/dt) / (dwbar/dt)^2 ,

  i.e. `psi2` is proportional to the fringe chirp `dOmega/dt` that the xcorr fit
  measures, with a positive constant of proportionality that does not depend on
  `tau`, on the `cos^2` doubling, or on how `beta` is defined. **Ratios of `psi2`
  values are therefore ratios of fringe chirps, full stop** -- which is how Q1 and Q2
  are answered without inheriting any of those conventions.

The one irreducible ambiguity: the interferogram is real, so
`(phi0, T0, psi2, psi3) -> -(phi0, T0, psi2, psi3)` leaves it unchanged. The branch is
fixed by `T0 > 0` wherever the arm delay is known to be positive (all of `scan_d`).
On `scan_L`, `dt = 0` gives no anchor, so `scan_L` measures `|psi2|` and the sign is
carried in from `scan_d` by continuity in `L`.

No plotting here.

Types
-----
The public functions take and return base_core quantities: a `SpecTrace` holds its
grid as ``AngularFrequency`` and the averaged spectrum as Measurement(mean, standard
error), with L as ``Length`` and dt as ``Time``; a `SpecFit` gives ``T0`` as ``Time``
and ``psi2`` as ``GDD`` (``psi3``, ps^3, and the visibility roll-off, ps^-2, have no
base_core type and stay floats). Each fit converts its trace to numpy once
(`_arrays`); the underscored steps inside are numpy. The raw exposures stay numpy: a
setpoint is a 2-D block of thousands of spectra, regridded and phase-aligned as one.
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from base_core.lab_specifics.base_models import Measurement
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length, Time
from base_core.quantities.specific_models import GDD, AngularFrequency
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain.xcorr_fit import (
    RUNS, load_scans, load_spectra)

#: Speed of light, nm/ps.
C_NM_PER_PS = 299792.458

#: Reference angular frequency, rad/ps (~802.0 nm). Fixed so `psi2` and `psi3` mean
#: the same thing at every setpoint; `T0` is the group delay *at this frequency*.
W0_REF = 2349.0

#: Common analysis band, rad/ps = **790 - 814 nm**, App_Apps' own `fringe_core.ZOOM`
#: (Kevin, 2026-08-31). The earlier 795.6 - 808.4 nm was set from an intensity-moment
#: width, which is wrong: the moment swallows the broad continuum pedestal into the
#: width and reports sigma ~13.6 nm. `fringe_core` instead fits `gauss(a, mu, sigma,
#: off)` with `off` anchored to continuum measured OUTSIDE the bump, giving the true
#: sigma ~3.9 nm -- so 790-814 nm is +-3.1 sigma, not the +-0.5 sigma it looks like.
#: The wings are ~1-2% of peak but carry SNR 60-150 in the averaged spectrum, and
#: `psi2`'s leverage grows as the band squared, so the wide band is nearly free: the
#: fit is 1/s_err weighted, and low-signal wings down-weight themselves.
W_LO, W_HI = 2313.7, 2384.0

#: Samples on the uniform-omega grid. The 790-814 nm band spans 70.3 rad/ps and holds
#: only ~590 raw pixels, so 1024 is already a 1.7x oversample -- 2048 was tried and
#: only bought a 4x slower fit for interpolation detail that is not in the data.
N_GRID = 1024

#: The spectrometer's dark pedestal is not subtracted in the file
#: (`@dark_subtraction = 0`); it is ~145 counts. Taken as the 1st percentile of the
#: setpoint's own mean spectrum over the full recorded range.
PEDESTAL_PCT = 1.0

#: This module's grating axis is ``stage - 30.0 mm``: it *derives* the zero and cannot
#: reference its axis to its own answer (see ``xcorr_fit.GRATING_ZERO_MM``).
GRATING_REF = Length(30.0, Prefix.MILLI)

_RAD_PER_PS = Prefix.TERA          # AngularFrequency prefix: rad/ps = 1e12 rad/s


@dataclass(frozen=True)
class SpecTrace:
    """One setpoint's averaged spectrum, on the common uniform-omega grid.

    ``w`` is ascending; ``spectrum`` is Measurement(mean counts with the pedestal
    removed, standard error of that mean) at each ``w``.
    """

    name: str
    setpoint: int
    L: Length                     # grating - GRATING_REF
    dt: Time                      # arm delay
    n_spectra: int
    w: list[AngularFrequency]
    spectrum: list[Measurement]
    align_gain: float = 1.0       # aligned / unaligned fringe amplitude; see _align_phi0

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(w in rad/ps, counts, standard error)`` as numpy arrays."""
        return (np.array([v.value(_RAD_PER_PS) for v in self.w], dtype=float),
                np.array([m.value for m in self.spectrum], dtype=float),
                np.array([m.error for m in self.spectrum], dtype=float))


class _Arrays(NamedTuple):
    """A `SpecTrace` as the fits use it: ``u = w - W0_REF`` (rad/ps), counts, error."""
    u: np.ndarray
    s: np.ndarray
    s_err: np.ndarray


def _arrays(tr: SpecTrace) -> _Arrays:
    w, s, e = tr.to_numpy()
    return _Arrays(w - W0_REF, s, e)


#: Power-iteration sweeps used to strip each exposure's own `phi0` before averaging.
#: Three is empirically past convergence; the reference moves by <1 mrad after two.
ALIGN_ITERS = 3

#: Group delays below this (ps) are envelope residual, not fringe, and are excluded
#: from the complex fringe used for phase referencing.
ALIGN_T_MIN = 0.05


def _analytic_fringe(N: np.ndarray, du: float, t_min: float = ALIGN_T_MIN):
    """Complex fringe of each row of `N`, keeping only group delays `T > t_min`.

    The rows are envelope-normalised real fringes. Zeroing the `T <= t_min` half of
    the spectrum is the analytic signal with the envelope residual removed; the
    result carries `exp(+i phi0)` linearly, which is what the alignment needs.
    """
    F = np.fft.fft(N, axis=1)
    T = np.fft.fftfreq(N.shape[1], du) * 2.0 * np.pi
    F[:, T <= t_min] = 0.0
    return 2.0 * np.fft.ifft(F, axis=1)


def _align_phi0(N: np.ndarray, du: float, iters: int = ALIGN_ITERS):
    """Strip each exposure's own `phi0`, then average. -> `(mean, sem, phi0, gain)`.

    A step of `load_traces` on the 2-D block of one setpoint's exposures (numpy).

    **Why this exists** (Kevin, 2026-08-31). Interferometric drift moves `phi0` from
    exposure to exposure. Averaging the raw spectra first leaves `T0` and `psi2`
    unbiased -- drift only multiplies the visibility by the coherence factor
    `|<exp(i phi0)>| = exp(-sigma_phi^2 / 2)` -- but that factor is a pure loss:
    measured at 0.84-0.91 on `scan_L` and 0.65-0.73 on `scan_d`, i.e. up to a third
    of the fringe amplitude thrown away to a parameter nobody wants. Only the first
    derivative of the phase and up carry physics; `phi0` is exactly the discarded
    one, and exactly the one that drifts.

    So: align first, average second. Each exposure is projected onto the current
    reference fringe, its own `phi0` divided out, and the reference rebuilt -- a
    power iteration for the rank-1 phase, which is the maximum-likelihood estimate
    under per-exposure phase noise. `T0`, `psi2` and `psi3` are untouched by
    construction: multiplying by `exp(-i phi0_i)` is a constant in `omega`, so it
    cannot move any derivative of the spectral phase.

    Returns the aligned mean fringe, its standard error, the per-exposure `phi0`, and
    `gain`, the ratio of aligned to unaligned fringe amplitude (1.0 = drift-free,
    >1 = amplitude recovered).
    """
    Z = _analytic_fringe(N, du)
    w = np.ones(len(Z), dtype=complex)
    for _ in range(iters):
        ref = (Z * w[:, None]).mean(axis=0)
        c = Z @ np.conj(ref)
        w = np.conj(c) / np.maximum(np.abs(c), 1e-300)
    aligned = N * 0.0 + np.real(Z * w[:, None])
    m = aligned.mean(axis=0)
    e = aligned.std(axis=0, ddof=1) / np.sqrt(len(aligned))
    plain = np.real(Z).mean(axis=0)
    gain = float(np.sqrt(np.mean(m ** 2) / max(np.mean(plain ** 2), 1e-300)))
    return m, e, np.angle(np.conj(w)), gain


def cache_path(path, align: bool) -> Path:
    import pathlib as _pl
    tag = "aligned" if align else "plain"
    return config.TEMP_DIR / f"_spec_{tag}_{_pl.Path(path).stem}.npz"


def save_traces(traces: list[SpecTrace], dest) -> None:
    """Persist averaged traces so a rerun skips the ~12 s/run regrid+align."""
    import pathlib as _pl
    dest = _pl.Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    arr = [tr.to_numpy() for tr in traces]
    np.savez_compressed(
        dest, w=arr[0][0],
        s=np.array([a[1] for a in arr]), s_err=np.array([a[2] for a in arr]),
        name=np.array([t.name for t in traces]),
        setpoint=np.array([t.setpoint for t in traces]),
        L_mm=np.array([t.L.value(Prefix.MILLI) for t in traces]),
        dt_ps=np.array([t.dt.value(Prefix.PICO) for t in traces]),
        n_spectra=np.array([t.n_spectra for t in traces]),
        align_gain=np.array([t.align_gain for t in traces]),
        band=np.array([W_LO, W_HI, N_GRID]))


def _trace(name, setpoint, L: Length, dt: Time, n_spectra, w: list[AngularFrequency],
           s, s_err, align_gain) -> SpecTrace:
    return SpecTrace(name=str(name), setpoint=int(setpoint), L=L, dt=dt,
                     n_spectra=int(n_spectra), w=w,
                     spectrum=[Measurement(float(v), float(e)) for v, e in zip(s, s_err)],
                     align_gain=float(align_gain))


def restore_traces(src) -> list[SpecTrace] | None:
    """Inverse of :func:`save_traces`; None if absent or built for another band."""
    import pathlib as _pl
    src = _pl.Path(src)
    if not src.exists():
        return None
    z = np.load(src, allow_pickle=False)
    if not np.allclose(z["band"], [W_LO, W_HI, N_GRID]):
        return None                      # band changed: the cache is stale, refuse it
    w = [AngularFrequency(float(v), _RAD_PER_PS) for v in z["w"]]
    return [_trace(z["name"][i], z["setpoint"][i],
                   Length(float(z["L_mm"][i]), Prefix.MILLI),
                   Time(float(z["dt_ps"][i]), Prefix.PICO),
                   z["n_spectra"][i], w, z["s"][i], z["s_err"][i], z["align_gain"][i])
            for i in range(len(z["setpoint"]))]


def load_traces(path, grating_zero: Length = GRATING_REF, align: bool = True,
                cache: bool = True) -> list[SpecTrace]:
    """Average every setpoint's spectra onto the common grid.

    A setpoint's rows are 6-7 spectra per probe point over the whole sweep, so this
    averages a few thousand exposures. The two-arm spectrum does not depend on the
    probe stage, so the average is of identical measurements.
    """
    if cache:
        got = restore_traces(cache_path(path, align))
        if got is not None:
            return got
    lam, counts, sidx = load_spectra(path)
    scans = {s.setpoint: s for s in load_scans(path, grating_zero)}

    lam_nm = np.array([v.value(Prefix.NANO) for v in lam], dtype=float)
    w_raw = 2.0 * np.pi * C_NM_PER_PS / lam_nm
    order = np.argsort(w_raw)
    w_sorted = w_raw[order]
    wg = np.linspace(W_LO, W_HI, N_GRID)
    w_typed = [AngularFrequency(float(v), _RAD_PER_PS) for v in wg]

    du = float(wg[1] - wg[0])
    out: list[SpecTrace] = []
    for k in range(int(sidx.max()) + 1):
        block = counts[sidx == k][:, order]
        # pedestal from the FULL recorded frame (734-882 nm), where there is real
        # continuum -- inside the 790-814 nm band there is none to measure.
        ped = np.percentile(block.mean(axis=0), PEDESTAL_PCT)
        sc = scans[k]
        if not align:
            m = block.mean(axis=0) - ped
            e = block.std(axis=0, ddof=1) / np.sqrt(len(block))
            s_g, e_g, gain = (np.interp(wg, w_sorted, m),
                              np.interp(wg, w_sorted, e), 1.0)
        else:
            # regrid every exposure, then align on phi0 before averaging
            G = np.array([np.interp(wg, w_sorted, r) for r in block]) - ped
            env = np.maximum(_smooth_env(G.mean(axis=0), 401), 1e-6)
            fr, fr_e, _, gain = _align_phi0(G / env - 1.0, du)
            s_g, e_g = env * (1.0 + fr), env * fr_e
        out.append(_trace(sc.name, k, sc.L, sc.dt, len(block), w_typed, s_g, e_g, gain))
    if cache:
        save_traces(out, cache_path(path, align))
    return out


# ------------------------------------------------------------------ seed search

def _smooth_env(s: np.ndarray, win: int) -> np.ndarray:
    win = int(win) | 1
    return savgol_filter(s, min(win, len(s) - 1), 2)


def _chirp_matched_filter(u: np.ndarray, y: np.ndarray, psi2_grid: np.ndarray,
                          t_min: float = 0.05):
    """Peak of |FT{ y . exp(-i psi2 u^2 / 2) }| over a grid of `psi2`.

    A quadratic spectral phase spreads the fringe over a range of group delays; the
    de-chirp collapses it back to a single peak, so the peak height is maximal at the
    true `psi2`. Returns `(psi2, T0, height)` arrays over the grid.
    """
    n = len(u)
    dw = float(u[1] - u[0])
    win = np.hanning(n)
    T = np.fft.fftfreq(n, dw) * 2.0 * np.pi
    ok = T >= t_min
    best_T = np.empty(len(psi2_grid))
    best_h = np.empty(len(psi2_grid))
    for i, p2 in enumerate(psi2_grid):
        Y = np.abs(np.fft.fft(y * win * np.exp(-0.5j * p2 * u ** 2)))
        Y = np.where(ok, Y, 0.0)
        j = int(np.argmax(Y))
        best_T[i] = T[j]
        best_h[i] = Y[j]
    return np.asarray(psi2_grid, float), best_T, best_h


# --------------------------------------------------------------------- the fit

@dataclass(frozen=True)
class SpecFit:
    ok: bool
    reason: str
    T0: Time                # group delay at W0_REF = the arm delay
    T0_err: Time
    psi2: GDD               # d(group delay)/dw = the arms' chirp mismatch
    psi2_err: GDD
    psi3: float             # ps^3 (no base_core type)
    phi0: float             # rad
    vis: float              # fringe visibility at W0_REF
    vis_decay: float        # ps^-2, visibility roll-off in local group delay (no type)
    env: np.ndarray         # log-envelope polynomial coefficients (numpy order)
    sse: float
    rms: float              # rms residual as a fraction of the envelope
    model: np.ndarray


_ENV_DEG = 4
_PNAMES = ("phi0", "T0", "psi2", "psi3", "vis", "vis_decay")


def _unpack(th):
    return th[:_ENV_DEG + 1], th[_ENV_DEG + 1:]


def _spec_model(th, u):
    ec, (phi0, T0, psi2, psi3, vis, vdec) = _unpack(th)
    env = np.exp(np.polyval(ec, u))
    phase = phi0 + T0 * u + 0.5 * psi2 * u ** 2 + psi3 * u ** 3 / 6.0
    gd = T0 + psi2 * u + 0.5 * psi3 * u ** 2          # local group delay
    v = vis * np.exp(-vdec * gd ** 2)
    return env * (1.0 + v * np.cos(phase)), env


def _failed(reason: str) -> SpecFit:
    nan_t, nan_g = Time(np.nan), GDD(np.nan)
    return SpecFit(False, reason, nan_t, nan_t, nan_g, nan_g, *([np.nan] * 4),
                   np.array([]), np.nan, np.nan, np.array([]))


def fit_spectrum(tr: SpecTrace, seed_psi2: GDD, seed_T0: Time,
                 seed_vdec: float = 0.02, anchor: str = "T0",
                 fix_psi2: bool = False, max_nfev: int = 20000) -> SpecFit:
    """Full nonlinear fit of the interferogram, envelope and fringe together.

    The envelope is `exp(poly_4(u))` -- positive by construction and flexible enough
    for the ~13 nm band -- fitted *simultaneously* with the fringe, so no smoothing
    step can eat fringe amplitude or manufacture structure. The visibility is allowed
    to fall off with the local group delay, which is the spectrometer's finite
    resolution: a fringe of period `2 pi / T` washes out as `T` approaches the
    instrument's limit. Without it the wings of the `L` scan, where the fringe runs
    fastest, would pull the envelope instead. ``seed_vdec`` is in ps^-2.
    """
    return _fit_spectrum(_arrays(tr), seed_psi2.value(Prefix.PICO),
                         seed_T0.value(Prefix.PICO), seed_vdec, anchor, fix_psi2, max_nfev)


def _fit_spectrum(a: _Arrays, seed_psi2: float, seed_T0: float,
                  seed_vdec: float = 0.02, anchor: str = "T0",
                  fix_psi2: bool = False, max_nfev: int = 20000) -> SpecFit:
    """:func:`fit_spectrum` on numpy arrays, seeds in ps^2 and ps."""
    u, s = a.u, a.s
    ec0 = np.polyfit(u, np.log(np.maximum(_smooth_env(s, 201), 1e-6)), _ENV_DEG)
    th0 = np.r_[ec0, 0.0, seed_T0, seed_psi2, 0.0, 0.3, seed_vdec]

    big = np.full(_ENV_DEG + 1, np.inf)
    lo = np.r_[-big, -np.inf, -60.0, -3.0, -1.0, 1e-3, 0.0]
    hi = np.r_[big, np.inf, 60.0, 3.0, 1.0, 1.5, 0.05]
    scale = np.r_[np.abs(ec0) + 1e-3, 1.0, 0.5, 0.02, 1e-3, 0.1, 1e-3]
    if fix_psi2:
        lo[_ENV_DEG + 3] = seed_psi2 - 1e-9
        hi[_ENV_DEG + 3] = seed_psi2 + 1e-9

    def resid(th):
        return (_spec_model(th, u)[0] - s) / np.maximum(a.s_err, 1e-9)

    try:
        r = least_squares(resid, th0, bounds=(lo, hi), x_scale=scale,
                          max_nfev=max_nfev)
    except Exception as exc:                                   # pragma: no cover
        return _failed(f"solver: {exc}")

    th = r.x.copy()
    sse = float(np.sum(r.fun ** 2))
    try:
        _, sv, VT = np.linalg.svd(r.jac, full_matrices=False)
        good = sv > 1e-10 * sv.max()
        cov = (VT[good].T / sv[good] ** 2) @ VT[good]
        cov = cov * sse / max(1, len(u) - len(th))
        sig = {n: float(np.sqrt(max(cov[_ENV_DEG + 1 + i, _ENV_DEG + 1 + i], 0.0)))
               for i, n in enumerate(_PNAMES)}
    except Exception:                                          # pragma: no cover
        sig = {n: float("nan") for n in _PNAMES}

    # canonical branch: a real interferogram cannot tell (phi0,T0,psi2,psi3) from
    # its negative. `anchor` says which coefficient carries the known sign -- "T0"
    # where the arm delay is known positive (scan_d), "psi2" where it is not
    # (scan_L, dt = 0), in which case only |psi2| is measured.
    key = th[_ENV_DEG + 2] if anchor == "T0" else th[_ENV_DEG + 3]
    if key < 0:
        th[_ENV_DEG + 1:_ENV_DEG + 5] *= -1.0

    mod, env = _spec_model(th, u)
    ec, (phi0, T0, psi2, psi3, vis, vdec) = _unpack(th)
    ps, ps2 = (lambda v: Time(float(v), Prefix.PICO)), (lambda v: GDD(float(v), Prefix.PICO))
    return SpecFit(True, "ok", ps(T0), ps(sig["T0"]), ps2(psi2), ps2(sig["psi2"]),
                   float(psi3), float(phi0), float(vis), float(vdec), ec, sse,
                   float(np.std((mod - s) / env)), mod)


#: Visibility-roll-off starts to try. The roll-off and the envelope trade against
#: each other, so a single start can land in a basin where the fringe is explained as
#: envelope structure; these three settle it.
VDEC_STARTS = (0.0, 0.03)

#: Iteration cap inside the profile. The profile only has to rank basins; the winner
#: is refitted without a cap afterwards.
PROFILE_NFEV = 2000


def _gdd_list(g: np.ndarray) -> list[GDD]:
    return [GDD(float(v), Prefix.PICO) for v in g]


def _ps2(g: Sequence[GDD]) -> np.ndarray:
    return np.array([v.value(Prefix.PICO) for v in g], dtype=float)


def profile_psi2(tr: SpecTrace, psi2_grid: Sequence[GDD], anchor: str = "T0",
                 t_min: float = 0.05) -> tuple[list[GDD], np.ndarray]:
    """Cost of the best fit with `psi2` pinned, at each point of the grid.

    This is the honest seed search and the honest identifiability statement in one.
    The matched filter (:func:`_chirp_matched_filter`) is a linear statistic and can be fooled
    where the fringe is slow -- on the `L` scan below |L| ~ 70 mm the fringe period
    approaches the width of the whole band, and the filter's peak lands at
    `psi2 = 0`, where the "fringe" is absorbed into the envelope. Refitting
    everything else at each pinned `psi2` cannot be fooled that way: it asks the only
    question that matters, which `psi2` explains the raw spectrum best.

    `T0` is seeded from the matched filter at that same `psi2` -- from the data, not
    from the commanded stage position, which is itself one of the things under test.
    ``t_min`` is in ps.

    Returns `(psi2_grid, sse)`.
    """
    g, sse = _profile_psi2(tr, _ps2(psi2_grid), anchor, t_min)
    return _gdd_list(g), sse


def _profile_psi2(tr: SpecTrace, psi2_grid: np.ndarray, anchor: str, t_min: float):
    """:func:`profile_psi2` on a numpy grid in ps^2."""
    a = _arrays(tr)
    T0s = _profile_T0s(a, psi2_grid, t_min)
    sse = np.empty(len(psi2_grid))
    for i, p2 in enumerate(psi2_grid):
        sse[i] = _profile_point_np(a, p2, float(T0s[i]), anchor)
    return np.asarray(psi2_grid, float), sse


def _profile_T0s(a: _Arrays, psi2_grid: np.ndarray, t_min: float) -> np.ndarray:
    """Matched-filter `T0` seed at each `psi2` of the grid (see :func:`profile_psi2`)."""
    _, T0s, _ = _chirp_matched_filter(
        a.u, a.s / np.maximum(_smooth_env(a.s, 201), 1e-9) - 1.0,
        psi2_grid, t_min)
    return T0s


def _profile_point_np(a: _Arrays, p2, t0_seed: float, anchor: str) -> float:
    """One point of the profile: the lowest cost over the starts, with `psi2` pinned."""
    best = np.inf
    for t0 in (t0_seed, 0.0):
        for v0 in VDEC_STARTS:
            f = _fit_spectrum(a, p2, t0, seed_vdec=v0, anchor=anchor,
                              fix_psi2=True, max_nfev=PROFILE_NFEV)
            if f.ok:
                best = min(best, f.sse)
    return best


def _profile_point(tr: SpecTrace, p2, t0_seed: float, anchor: str) -> float:
    """:func:`_profile_point_np` on a trace. Top-level so a process pool can run it
    (see :func:`fit_all`); the worker converts its trace once."""
    return _profile_point_np(_arrays(tr), p2, t0_seed, anchor)


def _fine_grid(g: np.ndarray, sse: np.ndarray, refine: int) -> np.ndarray:
    """Grid one coarse step either side of the coarse profile's minimum."""
    j = int(np.argmin(sse))
    step = float(g[1] - g[0])
    return np.linspace(g[j] - step, g[j] + step, refine)


def _final_starts(a: _Arrays, p2: float, t_min: float):
    """The `(T0, vis_decay)` starts of the free fit released at `p2`, in order."""
    T0s = _profile_T0s(a, np.array([p2]), t_min)
    return [(t0, v0) for t0 in (float(T0s[0]), 0.0) for v0 in VDEC_STARTS]


def _final_fit(tr: SpecTrace, p2: float, t0: float, v0: float, anchor: str) -> SpecFit:
    """One start of the free fit. Top-level so a process pool can run it."""
    return _fit_spectrum(_arrays(tr), p2, t0, seed_vdec=v0, anchor=anchor)


def _pick_best(fits) -> SpecFit | None:
    """The lowest-cost successful fit; ties go to the earliest start."""
    best = None
    for f in fits:
        if f.ok and (best is None or f.sse < best.sse):
            best = f
    return best


def best_fit(tr: SpecTrace, psi2_grid: Sequence[GDD], anchor: str = "T0",
             refine: int = 9, t_min: float = 0.05):
    """Profile `psi2` over the grid, then release it from the best point.

    `psi2_grid` should be coarse (the profile is a full refit per point); the
    minimum is then refined on a grid one step wide before the final free fit.
    Nothing is seeded from any across-trace trend -- each setpoint stands alone.
    Returns ``(fit, (psi2_grid, sse))``.
    """
    f, (g, sse) = _best_fit(tr, _ps2(psi2_grid), anchor, refine, t_min)
    return f, (_gdd_list(g), sse)


def _best_fit(tr: SpecTrace, psi2_grid: np.ndarray, anchor: str = "T0",
              refine: int = 9, t_min: float = 0.05):
    """:func:`best_fit` on a numpy grid in ps^2; the profile comes back numpy."""
    g, sse = _profile_psi2(tr, psi2_grid, anchor, t_min)
    g2, sse2 = _profile_psi2(tr, _fine_grid(g, sse, refine), anchor, t_min)
    p2 = float(g2[int(np.argmin(sse2))])
    a = _arrays(tr)
    best = _pick_best(_fit_spectrum(a, p2, t0, seed_vdec=v0, anchor=anchor)
                      for t0, v0 in _final_starts(a, p2, t_min))
    return best, (g, sse)


# ------------------------------------------------------------ the whole dataset

#: `psi2` search grids, ps^2. Coarse: every point is a full refit with `psi2` pinned.
GRID = {"scan_L": np.linspace(-0.05, 0.55, 61),
        "scan_d": np.linspace(-0.05, 0.05, 61)}

#: Which coefficient carries the known sign. `scan_d` has a known-positive arm delay;
#: `scan_L` sits at `dt = 0` and has none, so it measures |psi2| (see above).
ANCHOR = {"scan_L": "psi2", "scan_d": "T0"}

#: Half-width, in ps^2, and point count of the narrow `psi2` profile used when a
#: previous run's `spec_fits.csv` is available as a prior (`seed_from`).
#:
#: The blind 61-point profile costs ~86 s per trace on the 790-814 nm band -- 24 min
#: for the run, too slow to iterate on. The physics did not change when the band
#: widened or when the averaging was realigned, so the previous answer is a
#: legitimate warm start: three points spanning +-0.004 ps^2 around it, which is
#: ~40x the fitted `psi2` sigma, so the bracket is wide enough to catch a real shift
#: and narrow enough to cost three fits instead of 244. Whole run: ~30 s.
#:
#: **The blind profile stays the default.** A prior is only as good as the run that
#: made it, so `seed_from` must never be the first thing a new dataset sees, and a
#: fit that lands on either end of the bracket is reported as `pinned` and must be
#: rechecked blind.
SEED_HALFWIDTH = 0.004
SEED_N = 3


def _prior_grid(path: Path):
    """{(run, setpoint): psi2 in ps^2} from a previous `spec_fits.csv`, or None."""
    if not path.exists():
        return None
    return {(r["run"], int(r["setpoint"])): float(r["psi2_ps2"])
            for r in csv.DictReader(open(path, encoding="utf8"))}


#: Environment variable that sets the number of worker processes for :func:`fit_all`;
#: ``1`` runs the original serial loop in this process (for debugging).
WORKERS_ENV = "SHAPED_USCFG_SPECTRA_WORKERS"

#: BLAS/OpenMP thread caps given to the workers, so that N processes do not each
#: start N BLAS threads. They must be in the environment *before* a worker imports
#: numpy -- which happens as soon as it unpickles its first task -- so they are set
#: in this process's environment for the pool's lifetime and inherited at spawn.
_WORKER_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1"}


def _n_workers(workers: int | None) -> int:
    if workers is None:
        workers = int(os.environ.get(WORKERS_ENV, 0)) or (os.cpu_count() or 1)
    return max(1, int(workers))


@contextmanager
def _pool(workers: int):
    saved = {k: os.environ.get(k) for k in _WORKER_ENV}
    os.environ.update(_WORKER_ENV)
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            yield ex
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# Each pool task carries its `SpecTrace` (typed) and converts it once, like any fit.
# Pickling the typed trace costs 1.8 ms per task against 0.01 ms for bare arrays, which
# is 1.05x on the whole pool (42.9 s typed, 40.6 s numpy payload), so the tasks stay typed.

def _profiles_parallel(ex, items, t_min: float):
    """:func:`_profile_psi2` for every `(trace, grid, anchor)` of `items` at once.

    One task per grid point, so the long blind profiles of `scan_d` and the short
    warm-started brackets of `scan_L` share the workers evenly. The `T0` seeds are
    computed here exactly as the serial profile computes them, and the costs are
    gathered back in grid order, so the result is the serial one.
    """
    futs = []
    for tr, grid, anchor in items:
        T0s = _profile_T0s(_arrays(tr), grid, t_min)
        futs.append([ex.submit(_profile_point, tr, p2, float(T0s[i]), anchor)
                     for i, p2 in enumerate(grid)])
    out = []
    for (_, grid, _), fl in zip(items, futs):
        sse = np.empty(len(grid))
        for i, fu in enumerate(fl):
            sse[i] = fu.result()
        out.append((np.asarray(grid, float), sse))
    return out


def _best_fits_parallel(jobs, workers: int, refine: int = 9, t_min: float = 0.05):
    """:func:`_best_fit` for every `(tag, trace, grid)` of `jobs`, on a process pool.

    The same three steps as :func:`_best_fit` -- coarse profile, fine profile around
    its minimum, free fit from the fine minimum -- each spread over all traces at
    once, with the same selection at every step. Results come back in `jobs` order.
    """
    with _pool(workers) as ex:
        coarse = _profiles_parallel(
            ex, [(tr, grid, ANCHOR[tag]) for tag, tr, grid in jobs], t_min)
        fine = _profiles_parallel(
            ex, [(tr, _fine_grid(g, sse, refine), ANCHOR[tag])
                 for (tag, tr, _), (g, sse) in zip(jobs, coarse)], t_min)
        finals = []
        for (tag, tr, _), (g2, sse2) in zip(jobs, fine):
            p2 = float(g2[int(np.argmin(sse2))])
            finals.append([ex.submit(_final_fit, tr, p2, t0, v0, ANCHOR[tag])
                           for t0, v0 in _final_starts(_arrays(tr), p2, t_min)])
        return [(_pick_best(fu.result() for fu in fl), prof)
                for fl, prof in zip(finals, coarse)]


def _report(tag, tr, f, prior) -> None:
    x = tr.dt.value(Prefix.PICO) if tag == "scan_d" else tr.L.value(Prefix.MILLI)
    psi2 = f.psi2.value(Prefix.PICO)
    pin = ""
    if prior is not None and (tag, tr.setpoint) in prior:
        if abs(psi2 - prior[(tag, tr.setpoint)]) > 0.95 * SEED_HALFWIDTH:
            pin = "   PINNED AT BRACKET EDGE — recheck blind"
    print(f"  {tag} {x:8.3f}  T0 {f.T0.value(Prefix.PICO):+8.4f}  psi2 {psi2:+9.5f}"
          f"  rms {f.rms:.4f}  gain {tr.align_gain:.3f}{pin}", flush=True)


def fit_all(prior=None, workers: int | None = None):
    """Fit every setpoint of both runs. -> ``{run: [(trace, fit, profile), ...]}``,
    ``profile`` being ``(psi2 grid as list[GDD], sse)``.

    ``prior`` is ``{(run, setpoint): psi2 in ps^2}`` (see `_prior_grid`). ``workers``
    processes share the fits (default: the ``SHAPED_USCFG_SPECTRA_WORKERS``
    environment variable, else one per CPU); ``1`` runs them serially in this
    process. Both paths make the same fits and the same choices, in the same order.
    """
    jobs = []
    for tag, fn in RUNS.items():
        for tr in load_traces(fn):
            grid = GRID[tag]
            if prior is not None and (tag, tr.setpoint) in prior:
                p0 = prior[(tag, tr.setpoint)]
                grid = np.linspace(p0 - SEED_HALFWIDTH, p0 + SEED_HALFWIDTH, SEED_N)
            jobs.append((tag, tr, grid))

    workers = _n_workers(workers)
    out = {tag: [] for tag in RUNS}
    if workers == 1:
        results = [_best_fit(tr, grid, anchor=ANCHOR[tag]) for tag, tr, grid in jobs]
    else:
        results = _best_fits_parallel(jobs, workers)
    for (tag, tr, _), (f, (g, sse)) in zip(jobs, results):
        out[tag].append((tr, f, (_gdd_list(g), sse)))
        _report(tag, tr, f, prior)
    return out


def write_csv(res, path: Path):
    """``spec_fits.csv``: one row per setpoint."""
    ps, ps2 = (lambda v: v.value(Prefix.PICO)), (lambda v: v.value(Prefix.PICO))
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "setpoint", "group", "L_mm", "dt_ps", "n_spectra",
                    "T0_ps", "T0_sigma_ps", "psi2_ps2", "psi2_sigma_ps2",
                    "psi3_ps3", "visibility", "vis_decay", "rms_frac", "sse"])
        for tag, rows in res.items():
            for tr, f, _ in rows:
                w.writerow([tag, tr.setpoint, tr.name, f"{tr.L.value(Prefix.MILLI):.4g}",
                            f"{tr.dt.value(Prefix.PICO):.6g}", tr.n_spectra,
                            f"{ps(f.T0):.6g}", f"{ps(f.T0_err):.3g}",
                            f"{ps2(f.psi2):.6g}", f"{ps2(f.psi2_err):.3g}",
                            f"{f.psi3:.4g}", f"{f.vis:.4g}", f"{f.vis_decay:.4g}",
                            f"{f.rms:.4g}", f"{f.sse:.6g}"])


def run(out: Path, seed_from: Path | None = None, workers: int | None = None) -> None:
    """Fit every setpoint of both runs and write ``out/spec_fits.csv``.

    ``seed_from`` warm-starts `psi2` from a previous `spec_fits.csv` (see
    :data:`SEED_HALFWIDTH`); ``None`` runs the blind profile. The averaged, aligned
    spectra are cached in ``config.TEMP_DIR`` (:func:`cache_path`) and reused on the
    next run unless the band has changed. ``workers`` is passed to :func:`fit_all`.
    """
    out.mkdir(parents=True, exist_ok=True)
    res = fit_all(prior=_prior_grid(seed_from) if seed_from is not None else None,
                  workers=workers)
    write_csv(res, out / "spec_fits.csv")
