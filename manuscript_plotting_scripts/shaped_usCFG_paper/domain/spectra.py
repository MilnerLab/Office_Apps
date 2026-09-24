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

Pure numpy/scipy/h5py. No plotting here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

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


@dataclass
class SpecTrace:
    """One setpoint's averaged spectrum, on the common uniform-omega grid."""

    name: str
    setpoint: int
    L_mm: float
    dt_ps: float
    n_spectra: int
    w: np.ndarray          # rad/ps, uniform, ascending
    s: np.ndarray          # mean counts, pedestal removed
    s_err: np.ndarray      # standard error of that mean
    align_gain: float = 1.0   # aligned / unaligned fringe amplitude; see align_phi0

    @property
    def u(self) -> np.ndarray:
        return self.w - W0_REF


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


def align_phi0(N: np.ndarray, du: float, iters: int = ALIGN_ITERS):
    """Strip each exposure's own `phi0`, then average. -> `(mean, sem, phi0, gain)`.

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
    np.savez_compressed(
        dest, w=traces[0].w,
        s=np.array([t.s for t in traces]), s_err=np.array([t.s_err for t in traces]),
        name=np.array([t.name for t in traces]),
        setpoint=np.array([t.setpoint for t in traces]),
        L_mm=np.array([t.L_mm for t in traces]),
        dt_ps=np.array([t.dt_ps for t in traces]),
        n_spectra=np.array([t.n_spectra for t in traces]),
        align_gain=np.array([t.align_gain for t in traces]),
        band=np.array([W_LO, W_HI, N_GRID]))


def restore_traces(src) -> list[SpecTrace] | None:
    """Inverse of :func:`save_traces`; None if absent or built for another band."""
    import pathlib as _pl
    src = _pl.Path(src)
    if not src.exists():
        return None
    z = np.load(src, allow_pickle=False)
    if not np.allclose(z["band"], [W_LO, W_HI, N_GRID]):
        return None                      # band changed: the cache is stale, refuse it
    return [SpecTrace(name=str(z["name"][i]), setpoint=int(z["setpoint"][i]),
                      L_mm=float(z["L_mm"][i]), dt_ps=float(z["dt_ps"][i]),
                      n_spectra=int(z["n_spectra"][i]), w=z["w"],
                      s=z["s"][i], s_err=z["s_err"][i],
                      align_gain=float(z["align_gain"][i]))
            for i in range(len(z["setpoint"]))]


def load_traces(path, grating_zero_mm: float = 30.0, align: bool = True,
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
    scans = {s.setpoint: s for s in load_scans(path, grating_zero_mm)}

    w_raw = 2.0 * np.pi * C_NM_PER_PS / lam
    order = np.argsort(w_raw)
    w_sorted = w_raw[order]
    wg = np.linspace(W_LO, W_HI, N_GRID)

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
            fr, fr_e, _, gain = align_phi0(G / env - 1.0, du)
            s_g, e_g = env * (1.0 + fr), env * fr_e
        out.append(SpecTrace(
            name=sc.name, setpoint=k, L_mm=sc.L_mm, dt_ps=sc.dt_ps,
            n_spectra=len(block), w=wg, s=s_g, s_err=e_g, align_gain=float(gain),
        ))
    if cache:
        save_traces(out, cache_path(path, align))
    return out


# ------------------------------------------------------------------ seed search

def _smooth_env(s: np.ndarray, win: int) -> np.ndarray:
    win = int(win) | 1
    return savgol_filter(s, min(win, len(s) - 1), 2)


def chirp_matched_filter(u: np.ndarray, y: np.ndarray, psi2_grid: np.ndarray,
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

@dataclass
class SpecFit:
    ok: bool
    reason: str
    T0: float               # ps      group delay at W0_REF = the arm delay
    psi2: float             # ps^2    d(group delay)/dw = the arms' chirp mismatch
    psi3: float             # ps^3
    phi0: float             # rad
    vis: float              # fringe visibility at W0_REF
    vis_decay: float        # ps^-2, visibility roll-off in local group delay
    env: np.ndarray         # log-envelope polynomial coefficients (numpy order)
    sse: float
    rms: float              # rms residual as a fraction of the envelope
    sigma: dict             # 1-sigma from the covariance, per named parameter
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


def fit_spectrum(tr: SpecTrace, seed_psi2: float, seed_T0: float,
                 seed_vdec: float = 0.02, anchor: str = "T0",
                 fix_psi2: bool = False, max_nfev: int = 20000) -> SpecFit:
    """Full nonlinear fit of the interferogram, envelope and fringe together.

    The envelope is `exp(poly_4(u))` -- positive by construction and flexible enough
    for the ~13 nm band -- fitted *simultaneously* with the fringe, so no smoothing
    step can eat fringe amplitude or manufacture structure. The visibility is allowed
    to fall off with the local group delay, which is the spectrometer's finite
    resolution: a fringe of period `2 pi / T` washes out as `T` approaches the
    instrument's limit. Without it the wings of the `L` scan, where the fringe runs
    fastest, would pull the envelope instead.
    """
    u, s = tr.u, tr.s
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
        return (_spec_model(th, u)[0] - s) / np.maximum(tr.s_err, 1e-9)

    try:
        r = least_squares(resid, th0, bounds=(lo, hi), x_scale=scale,
                          max_nfev=max_nfev)
    except Exception as exc:                                   # pragma: no cover
        return SpecFit(False, f"solver: {exc}", *([np.nan] * 6), np.array([]),
                       np.nan, np.nan, {}, np.array([]))

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
    return SpecFit(True, "ok", float(T0), float(psi2), float(psi3), float(phi0),
                   float(vis), float(vdec), ec, sse,
                   float(np.std((mod - s) / env)), sig, mod)


#: Visibility-roll-off starts to try. The roll-off and the envelope trade against
#: each other, so a single start can land in a basin where the fringe is explained as
#: envelope structure; these three settle it.
VDEC_STARTS = (0.0, 0.03)

#: Iteration cap inside the profile. The profile only has to rank basins; the winner
#: is refitted without a cap afterwards.
PROFILE_NFEV = 2000


def profile_psi2(tr: SpecTrace, psi2_grid: np.ndarray, anchor: str = "T0",
                 t_min: float = 0.05):
    """Cost of the best fit with `psi2` pinned, at each point of the grid.

    This is the honest seed search and the honest identifiability statement in one.
    The matched filter (:func:`chirp_matched_filter`) is a linear statistic and can be fooled
    where the fringe is slow -- on the `L` scan below |L| ~ 70 mm the fringe period
    approaches the width of the whole band, and the filter's peak lands at
    `psi2 = 0`, where the "fringe" is absorbed into the envelope. Refitting
    everything else at each pinned `psi2` cannot be fooled that way: it asks the only
    question that matters, which `psi2` explains the raw spectrum best.

    `T0` is seeded from the matched filter at that same `psi2` -- from the data, not
    from the commanded stage position, which is itself one of the things under test.

    Returns `(psi2_grid, sse)`.
    """
    _, T0s, _ = chirp_matched_filter(
        tr.u, tr.s / np.maximum(_smooth_env(tr.s, 201), 1e-9) - 1.0,
        psi2_grid, t_min)
    sse = np.empty(len(psi2_grid))
    for i, p2 in enumerate(psi2_grid):
        best = np.inf
        for t0 in (float(T0s[i]), 0.0):
            for v0 in VDEC_STARTS:
                f = fit_spectrum(tr, p2, t0, seed_vdec=v0, anchor=anchor,
                                 fix_psi2=True, max_nfev=PROFILE_NFEV)
                if f.ok:
                    best = min(best, f.sse)
        sse[i] = best
    return np.asarray(psi2_grid, float), sse


def best_fit(tr: SpecTrace, psi2_grid: np.ndarray, anchor: str = "T0",
             refine: int = 9, t_min: float = 0.05):
    """Profile `psi2` over the grid, then release it from the best point.

    `psi2_grid` should be coarse (the profile is a full refit per point); the
    minimum is then refined on a grid one step wide before the final free fit.
    Nothing is seeded from any across-trace trend -- each setpoint stands alone.
    """
    g, sse = profile_psi2(tr, psi2_grid, anchor, t_min)
    j = int(np.argmin(sse))
    step = float(g[1] - g[0])
    fine = np.linspace(g[j] - step, g[j] + step, refine)
    g2, sse2 = profile_psi2(tr, fine, anchor, t_min)
    p2 = float(g2[int(np.argmin(sse2))])

    _, T0s, _ = chirp_matched_filter(
        tr.u, tr.s / np.maximum(_smooth_env(tr.s, 201), 1e-9) - 1.0,
        np.array([p2]), t_min)
    best = None
    for t0 in (float(T0s[0]), 0.0):
        for v0 in VDEC_STARTS:
            f = fit_spectrum(tr, p2, t0, seed_vdec=v0, anchor=anchor)
            if f.ok and (best is None or f.sse < best.sse):
                best = f
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
    """{(run, setpoint): psi2} from a previous `spec_fits.csv`, or None."""
    if not path.exists():
        return None
    return {(r["run"], int(r["setpoint"])): float(r["psi2_ps2"])
            for r in csv.DictReader(open(path, encoding="utf8"))}


def fit_all(prior=None):
    out = {}
    for tag, fn in RUNS.items():
        rows = []
        for tr in load_traces(fn):
            grid = GRID[tag]
            if prior is not None and (tag, tr.setpoint) in prior:
                p0 = prior[(tag, tr.setpoint)]
                grid = np.linspace(p0 - SEED_HALFWIDTH, p0 + SEED_HALFWIDTH, SEED_N)
            f, prof = best_fit(tr, grid, anchor=ANCHOR[tag])
            rows.append((tr, f, prof))
            x = tr.dt_ps if tag == "scan_d" else tr.L_mm
            pin = ""
            if prior is not None and (tag, tr.setpoint) in prior:
                if abs(f.psi2 - prior[(tag, tr.setpoint)]) > 0.95 * SEED_HALFWIDTH:
                    pin = "   PINNED AT BRACKET EDGE — recheck blind"
            print(f"  {tag} {x:8.3f}  T0 {f.T0:+8.4f}  psi2 {f.psi2:+9.5f}"
                  f"  rms {f.rms:.4f}  gain {tr.align_gain:.3f}{pin}", flush=True)
        out[tag] = rows
    return out


def write_csv(res, path: Path):
    """``spec_fits.csv``: one row per setpoint."""
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "setpoint", "group", "L_mm", "dt_ps", "n_spectra",
                    "T0_ps", "T0_sigma_ps", "psi2_ps2", "psi2_sigma_ps2",
                    "psi3_ps3", "visibility", "vis_decay", "rms_frac", "sse"])
        for tag, rows in res.items():
            for tr, f, _ in rows:
                w.writerow([tag, tr.setpoint, tr.name, f"{tr.L_mm:.4g}",
                            f"{tr.dt_ps:.6g}", tr.n_spectra,
                            f"{f.T0:.6g}", f"{f.sigma['T0']:.3g}",
                            f"{f.psi2:.6g}", f"{f.sigma['psi2']:.3g}",
                            f"{f.psi3:.4g}", f"{f.vis:.4g}", f"{f.vis_decay:.4g}",
                            f"{f.rms:.4g}", f"{f.sse:.6g}"])


def run(out: Path, seed_from: Path | None = None) -> None:
    """Fit every setpoint of both runs and write ``out/spec_fits.csv``.

    ``seed_from`` warm-starts `psi2` from a previous `spec_fits.csv` (see
    :data:`SEED_HALFWIDTH`); ``None`` runs the blind profile. The averaged, aligned
    spectra are cached in ``config.TEMP_DIR`` (:func:`cache_path`) and reused on the
    next run unless the band has changed.
    """
    out.mkdir(parents=True, exist_ok=True)
    res = fit_all(prior=_prior_grid(seed_from) if seed_from is not None else None)
    write_csv(res, out / "spec_fits.csv")
