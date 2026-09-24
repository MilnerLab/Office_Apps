"""Per-trace XCORR fringe analysis for the 2026-08-25 two-scan dataset.

Independent of the lab app's `fringe_fit`. Written so every step is inspectable and
plottable on its own, because the point of this round is to *show* the process:

    1. envelope      — crude sliding-window envelope, enough to normalise
    2. STFT ridge    — local |f| from a short-time Fourier transform, several windows
                       (2026-09-22: replaced the Hilbert seeds, see :func:`full_fit`)
    3. folded seed   — fit  |p(u)|  to that |f|   (NOT a plain quadratic; see below)
    4. full fit      — 9-parameter nonlinear fit to the RAW signal

**The folded seed is the whole point of step 3.** A spectrogram ridge, like the
Hilbert transform of a real signal, returns a non-negative frequency, so on the L scan — where the
true fringe frequency passes through zero at the pulse centre — what comes back is
|f(t)|, a V. Fitting a plain quadratic to that V produces a strong, entirely spurious
curvature, and the recovered phase is wrong on one side of the vertex. The correct
form is the *weakly* quadratic absolute value |p0 + p1 u + p2 u²| with p2 small: a
folded straight line. `fit_folded` enumerates the vertex position, which is the only
non-convex part, and is exact once the vertex is fixed.

Conventions, all fixed by Kevin 2026-08-30:

* `t = 2 (probe_mm - probe_offset_mm) / c`     probe stage -> delay, double pass
* `dt = 2 delay_base_mm / c`                   arm delay
* `L  = grating_mm - grating_zero_mm`
* **t = 0 is the fitted envelope centre.** Not the Omega = 0 crossing: the L scan's
  `dt = 0` is an approximation, so Omega is only approximately zero there and cannot
  define the origin.
* **The detector sees the headless vertical polarization projection of the
  corkscrew**, so its cos^2 doubles the rotation rate and the observed fringe sits at
  `2 f_usCFG`. Every readout halves it. This is the ONLY place the factor enters.
* `tau = 306 ps` for `Delta f = f(tau/2) - f(-tau/2)`. **Measured** 2026-08-31 from
  the delay-arm-only cross-correlation, superseding the provisional 320. It cancels
  out of `dDelta beta/d|L|` (`Delta f` is proportional to tau and is then divided by
  it), so changing it rescales the reported `Delta f` and nothing else.

Pure numpy/scipy/h5py. No plotting here.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares

from manuscript_plotting_scripts.shaped_usCFG_paper import config

#: The cut scan is 08-25; the delay scan is the 08-31 re-take at the true zero. The
#: 08-25 delay scan sat at 30.00 mm -- L = +1.9 mm -- so it was never an L = 0 scan,
#: and it is not used anywhere (Kevin, 2026-08-31).
RUNS = {"scan_L": config.XCORR_SCAN_L,
        "scan_d": config.XCORR_SCAN_D}

#: Speed of light, mm/ps.
C_MM_PER_PS = 0.299792458

#: Chirped pulse duration entering Delta f, ps.
#:
#: MEASURED 2026-08-31: 306 +- 3 ps, from a Gaussian fit to the delay-arm-only
#: cross-correlation -- FWHM 305.64 +- 2.95 ps. Was 320 (provisional).
#:
#: No grating-stage transfer is applied. tau is set by the stretcher and does not
#: depend on the compressor/shaper stage position (Kevin, 2026-08-31); the data agree,
#: envelope FWHM vs stage across scan_L being 0.054 +- 0.155 ps/mm over 80 mm, i.e.
#: flat. So the measured width at stage -20.0 mm IS the width at the operating point,
#: and the error bar is the fit's own +-3 ps, not +-8. See NOTES-2026-08-31.md 8.1.
TAU_PS = 306.0

#: The detector's cos^2 doubling: f_observed = FRINGE_PER_USCFG * f_usCFG.
FRINGE_PER_USCFG = 2.0

#: Half-width of the frequency-extraction window, in envelope sigmas.
#:
#: Raised from 1.5 to 2.6 on 2026-08-30 at Kevin's instruction, after the wings
#: turned out to be far cleaner than assumed: the point-to-point noise measured
#: beyond 2.8 sigma (second differences, where no fringe survives) is ~1 mV, so the
#: *normalised* fringe still carries SNR ~20 at 2.6 sigma. Cutting at 1.5 sigma threw
#: away the fastest, best-resolved fringes and left `Delta f` at +-tau/2 = +-154 ps
#: anchored only by the window edge. 2.6 sigma ~ +-330 ps of the +-395 ps record.
KEEP_SIGMA = 2.6

#: Grating-stage position at which the two arms are matched, mm.
#:
#: MEASURED 2026-08-31 at 28.12 +- 0.10, from the 08-31 `dt` scan's own psi2
#: intercept read back along the `scan_L` psi2-vs-L line. The 08-25 analysis assumed
#: 30.0 and was 1.9 mm out. `L = grating_mm - GRATING_ZERO_MM`.
#: (Was quoted as 28.10 while the value was averaged with a second anchor from the
#: 08-25 `dt` scan; that scan is dropped -- it sat at L = +1.9 mm and was never an
#: L = 0 scan -- so the single 08-31 anchor stands alone. The two agreed to 0.03 mm,
#: a third of the error bar, so nothing downstream moves.)
#: `spectra.py` deliberately still works in `stage - 30.0`: that module *derives*
#: the zero and cannot reference its axis to its own answer.
GRATING_ZERO_MM = 28.12

#: |L| above which a fringe-frequency zero crossing is allowed in the seed fit, mm.
#:
#: The fold is physics (Kevin, 2026-08-30): f_usCFG can only reach zero once the
#: shaper cut has given the arms different chirps. Raised from 1.0 to 3.0 when the
#: zero moved off 30.0, so that the classification is unchanged. The 08-31 dt scan
#: sits 0.03 mm from the zero, so it never folds; nothing about which sweeps fold
#: depends on the re-referencing.
FOLD_L_MM = 3.0


# --------------------------------------------------------------------------- io

@dataclass
class Scan:
    """One probe sweep, in the units the analysis works in."""

    run_id: str
    name: str
    setpoint: int
    L_mm: float
    dt_ps: float
    t_ps: np.ndarray          # probe delay, zero at the scan's own window start
    y: np.ndarray             # v_mean_pos
    y_err: np.ndarray         # v_std
    probe_mm: np.ndarray
    probe_offset_mm: float

    @property
    def dt_sample_ps(self) -> float:
        return float(np.median(np.diff(self.t_ps)))

    @property
    def nyquist_ghz(self) -> float:
        return 0.5e3 / self.dt_sample_ps


def load_scans(path, grating_zero_mm: float = GRATING_ZERO_MM) -> list[Scan]:
    """Every scan group of one run file, in setpoint order."""
    out: list[Scan] = []
    with h5py.File(str(path), "r") as f:
        run_id = str(f.attrs["run_id"])
        for i, name in enumerate(sorted(f["scans"].keys())):
            g = f["scans"][name]
            a = g.attrs
            probe = np.asarray(g["probe_mm"][:], float)
            off = float(a["probe_offset_mm"])
            out.append(Scan(
                run_id=run_id,
                name=name,
                setpoint=i,
                L_mm=float(a["grating_mm"]) - grating_zero_mm,
                dt_ps=2.0 * float(a["delay_base_mm"]) / C_MM_PER_PS,
                t_ps=2.0 * (probe - off) / C_MM_PER_PS,
                y=np.asarray(g["v_mean_pos"][:], float),
                y_err=np.asarray(g["v_std"][:], float),
                probe_mm=probe,
                probe_offset_mm=off,
            ))
    return out


def load_spectra(path):
    """The spectrometer block: (wavelength_nm, counts, setpoint_index)."""
    with h5py.File(str(path), "r") as f:
        sp = f["spectra"]
        return (np.asarray(sp["wavelength_nm"][:], float),
                np.asarray(sp["counts"][:], float),
                np.asarray(sp["setpoint_index"][:], int))


# ------------------------------------------------------------- step 1: envelope

def sliding_envelope(y: np.ndarray, half_width: int):
    """Crude upper/lower envelope by sliding max/min. Seeds everything downstream."""
    n = len(y)
    w = max(2, int(half_width))
    up = np.empty(n)
    lo = np.empty(n)
    for i in range(n):
        s = slice(max(0, i - w), min(n, i + w + 1))
        up[i] = y[s].max()
        lo[i] = y[s].min()
    return up, lo


def gaussian_envelope_seed(t: np.ndarray, y: np.ndarray):
    """(base, amp, mu, sigma) for the Gaussian under the fringe crests."""
    base = float(np.percentile(y, 3))
    w = max(3, len(y) // 40)
    up, _ = sliding_envelope(y, w)
    p = np.clip(up - base, 0.0, None)
    tot = p.sum()
    mu = float((t * p).sum() / tot)
    sigma = float(np.sqrt(((t - mu) ** 2 * p).sum() / tot))
    return base, float(p.max()), mu, sigma


# ---------------------------------------------------- step 3: folded (|.|) seed

def fit_folded(u: np.ndarray, f_abs: np.ndarray, deg: int = 1,
               n_vertex: int = 121, irls: int = 4, w: np.ndarray | None = None,
               fold: bool = True):
    """Fit ``|p(u)|`` to a non-negative frequency curve. Returns signed ``p``.

    **This is the step Kevin flagged.** On the L scan the true fringe frequency
    passes through zero at the pulse centre, so the folded curve is a V. Fitting a
    plain quadratic to a V returns a large spurious curvature and a phase that is
    wrong on one side of the vertex. The right form is the *weakly* quadratic
    absolute value ``|p0 + p1 u + p2 u²|`` — a folded straight line with a small
    correction — and that is what this returns.

    ``deg`` is the degree of ``p``. The fold is the only non-convex part: for each
    candidate vertex the sign pattern is fixed, ``|p| = s·p`` is linear, and the fit
    is an ordinary (Tukey-reweighted) least squares. A candidate vertex outside the
    data range reproduces the unfolded fit, so "no fold" is included rather than
    special-cased.

    ``fold=False`` keeps *only* those two outside-range candidates, i.e. forbids a
    zero crossing inside the record. See :func:`full_fit` for when that applies.
    """
    u = np.asarray(u, float)
    f_abs = np.asarray(f_abs, float)
    w0 = np.ones_like(u) if w is None else np.asarray(w, float)
    lo, hi = u.min(), u.max()
    span = hi - lo
    cands = (np.array([lo - span, hi + span]) if not fold else
             np.concatenate([[lo - span], np.linspace(lo, hi, n_vertex), [hi + span]]))
    V = np.vander(u, deg + 1)
    best = None
    for uv in cands:
        for lead in (+1.0, -1.0):
            s = np.where(u >= uv, lead, -lead)
            z = s * f_abs
            w = w0.copy()
            for _ in range(irls):
                p, *_ = np.linalg.lstsq(V * w[:, None], z * w, rcond=None)
                r = np.abs(V @ p) - f_abs
                mad = float(np.median(np.abs(r - np.median(r)))) or 1.0
                x = np.clip(r / (4.685 * 1.4826 * mad), -1.0, 1.0)
                w = w0 * (1.0 - x ** 2) ** 2
            r = np.abs(V @ p) - f_abs
            cost = float(np.sum(np.minimum(r ** 2, (3.0 * 1.4826 *
                         (np.median(np.abs(r - np.median(r))) or 1.0)) ** 2)))
            if best is None or cost < best[0]:
                best = (cost, p, uv)
    cost, p, uv = best
    return p, uv, cost


def seed_phase_coeffs(p_ghz: np.ndarray) -> np.ndarray:
    """Signed frequency polynomial (GHz, ascending powers of u) -> phase c1..c3.

    ``f(u) = p0 + p1 u + p2 u²`` in GHz and ``Phi(u) = c0 + c1 u + c2 u² + c3 u³`` in
    rad give ``c1 = 2 pi p0``, ``c2 = pi p1``, ``c3 = 2 pi p2 / 3`` once GHz is turned
    into cycles/ps.
    """
    p = np.zeros(3)
    p[:len(p_ghz)] = np.asarray(p_ghz, float)[::-1] / 1e3   # -> cycles/ps
    return np.array([2.0 * np.pi * p[0], np.pi * p[1], 2.0 * np.pi * p[2] / 3.0])


# ------------------------------------------------------- step 4: full raw fit

#: Parameter order for :func:`full_fit`.
PARAMS = ("base", "amp", "mu", "sigma", "vis", "c0", "c1", "c2", "c3")


def _model(th, t):
    base, amp, mu, sigma, vis, c0, c1, c2, c3 = th
    u = t - mu
    env = amp * np.exp(-u ** 2 / (2.0 * sigma ** 2))
    phi = c0 + c1 * u + c2 * u ** 2 + c3 * u ** 3
    return base + env * (1.0 + vis * np.cos(phi))


def core_fft_peak(t, y, base, amp, mu, sigma, keep=1.5):
    """Crude fringe frequency of the normalised core, GHz. Sets the SavGol window."""
    m = np.abs(t - mu) <= keep * sigma
    tc, yc = t[m], y[m]
    env = amp * np.exp(-(tc - mu) ** 2 / (2.0 * sigma ** 2))
    n = (yc - base - env) / env
    n = (n - n.mean()) * np.hanning(len(n))
    F = np.abs(np.fft.rfft(n, n=8 * len(n)))
    fr = np.fft.rfftfreq(8 * len(n), float(np.median(np.diff(tc)))) * 1e3
    k = fr > 1.0
    return float(fr[k][np.argmax(F[k])])


@dataclass
class FitResult:
    ok: bool
    status: str
    theta: np.ndarray = field(default_factory=lambda: np.zeros(9))
    cov: np.ndarray = field(default_factory=lambda: np.full((9, 9), np.nan))
    sse: float = float("nan")
    rho2: float = float("nan")
    seed_theta: np.ndarray = field(default_factory=lambda: np.zeros(9))
    #: every STFT seed that competed, as (name, p) with p in GHz, descending powers
    p_seeds: list = field(default_factory=list)
    seed_name: str = ""
    #: True when the no-zero-crossing constraint had to be imposed (L = 0 only).
    constrained: bool = False
    f_hint_ghz: float = float("nan")
    resid: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __getitem__(self, key):
        return self.theta[PARAMS.index(key)]


#: STFT window lengths, as fractions of the record, that compete as seeds.
STFT_FRACS = (0.08, 0.12, 0.16, 0.24)


def stft_ridge_n(u, n, frac: float, nseg: int = 31):
    """Peak fringe frequency of an already-normalised fringe ``n(u)``. → ``(u, f_ghz)``.

    Works on unevenly sampled ``u``: the peak is found in cycles per sample and
    converted with that window's own ps-per-sample. The spectrometer needs this.
    Its pixels are even in omega, so after the lambda -> t map the spacing in t
    varies across the record, and on the delay scan that variation carries the
    whole chirp.
    """
    N = len(n)
    W = max(8, int(frac * N))
    out_u, out_f = [], []
    for c in np.linspace(W // 2, N - W // 2 - 1, nseg).astype(int):
        sl = slice(c - W // 2, c + W // 2)
        seg = n[sl]
        seg = (seg - seg.mean()) * np.hanning(len(seg))
        F = np.abs(np.fft.rfft(seg, n=16 * len(seg)))
        fpx = np.fft.rfftfreq(16 * len(seg), 1.0)
        dt_px = (u[sl][-1] - u[sl][0]) / (len(seg) - 1)
        out_u.append(u[c])
        out_f.append(float(fpx[1:][np.argmax(F[1:])]) / dt_px * 1e3)
    return np.asarray(out_u), np.asarray(out_f)


def stft_seeds(u, n, sigma, fold: bool):
    """The STFT seed set: every window length, each fitted with a V and a quadratic.

    ``|f|`` is read straight off the spectrogram, so there is no phase to unwrap and
    nothing to slip. The V (``deg = 1``, ``|p0 + p1 u|``) is essential on the slow
    L sweeps: there a quadratic ``|f|`` that never reaches zero mimics the V over the
    record and wins on the ridge while sitting in the wrong basin of the raw fit.
    The window length trades frequency resolution against smearing by the chirp, so
    each length competes. → list of ``(name, p)``.
    """
    w_all = np.exp(-u ** 2 / (2.0 * sigma ** 2))
    out = []
    for frac in STFT_FRACS:
        us, fs = stft_ridge_n(u, n, frac)
        w = np.interp(us, u, w_all)
        for deg in (1, 2):
            p, _, _ = fit_folded(us, fs, deg=deg, irls=2, w=w, fold=fold)
            out.append((f"stft {frac:.2f} deg{deg}", p))
    return out


#: Strength of the no-zero-crossing constraint, in units of "envelope amplitudes per
#: GHz of violation". Large enough to be a hard constraint in practice (a 0.1 GHz dip
#: below zero over a few grid points already outweighs any fringe improvement), small
#: enough that the Jacobian stays conditioned.
PEN_NO_ZERO = 4.0


def _no_zero_resid(c1, c2, c3, ug, scale):
    """Residuals penalising any sign change of ``f(u)`` on the fixed grid ``ug``.

    ``f`` may not reach zero inside the record when both arms carry the same chirp
    (``L = 0``); see :func:`full_fit`. The constraint is written as "keep the sign
    ``f`` has at the envelope centre", i.e. the sign of ``c1``, which is smooth in the
    parameters and is exactly the physical statement — the rotation never stops and so
    never reverses.

    ``ug`` is built once from the *seed* sigma and never rebuilt. Deriving it from the
    live sigma lets the optimizer satisfy the constraint by shrinking the envelope
    until the grid no longer covers the crossing, which is what happened the first
    time this was tried.
    """
    fr = (c1 + 2.0 * c2 * ug + 3.0 * c3 * ug ** 2) / (2.0 * np.pi) * 1e3   # GHz
    s = 1.0 if c1 >= 0 else -1.0
    return scale * np.minimum(s * fr, 0.0)


def _fit_phase_fixed_env(t, y, env_th, c_seed, f_hint, ug=None, pen_scale=0.0):
    """Fit c0..c3 against the raw counts with the envelope held fixed. → OptimizeResult."""
    base, amp, mu, sigma = env_th
    u = t - mu
    env = amp * np.exp(-u ** 2 / (2.0 * sigma ** 2))

    def model(c):
        return base + env * (1.0 + c[0] * np.cos(c[1] + c[2] * u + c[3] * u ** 2
                                                 + c[4] * u ** 3))

    def resid(c):
        r = model(c) - y
        if ug is None or pen_scale <= 0.0:
            return r
        return np.concatenate([r, _no_zero_resid(c[2], c[3], c[4], ug, pen_scale)])

    span = max(3.0 * sigma, 1.0)
    w1 = 2.0 * np.pi * max(f_hint, 1.0) / 1e3
    scale = np.array([1.0, 1.0, w1, w1 / span, w1 / span ** 2])
    best = None
    for c0_try in (0.0, np.pi):
        for sgn in (+1.0, -1.0):
            g = np.array([0.5, c0_try, sgn * c_seed[0], sgn * c_seed[1],
                          sgn * c_seed[2]])
            try:
                r = least_squares(resid, g, x_scale=scale, max_nfev=8000)
            except Exception:                                  # pragma: no cover
                continue
            if best is None or r.cost < best.cost:
                best = r
    return best


def full_fit(scan: Scan, keep: float = KEEP_SIGMA,
             fold: bool | None = None) -> FitResult:
    """The whole per-trace pipeline.

    Staged deliberately. Freeing the envelope and the phase together lets the
    optimizer explain the data by shrinking the fringe visibility and reshaping the
    envelope instead of matching the fringe — on this dataset that cost roughly half
    the sweeps their basin. So: fix the envelope, fit the phase against the raw
    counts from competing seeds, then polish everything together from the winner.

    **The fold is physics, not a fitting convenience** (Kevin, 2026-08-30). The
    fringe frequency can only reach zero if the two arms carry *different* chirps,
    which happens only when the shaper cut moves them apart — i.e. at ``L != 0``. On
    the ``L = 0`` sweeps of the ``dt`` scan both arms are identical, the rotation
    never stops, and ``f_usCFG`` has **no zero crossing anywhere in the record**.
    Allowing one there is pure extra freedom, and the low-``dt`` sweeps — low
    frequency, few fringes, little information — are exactly the ones that spend it
    on nonsense. So ``fold`` is switched off at ``L = 0`` and left on for the
    ``dt = 0`` scan, whose wings carry high enough frequencies to pin the V properly.
    ``fold=None`` picks by ``|L|``; pass a bool to override.

    **Seeds are STFT only** (Kevin, 2026-09-22). The phase fit starts from every entry
    of :func:`stft_seeds`. The earlier competition of the Hilbert phase, the Hilbert
    frequency and one STFT ridge is gone: that Hilbert-phase seed alone lands in the
    wrong basin on most of the L scan, and it slips cycles wherever the fringe is
    off-centre in its envelope. Hilbert is the right tool when the phase itself is
    wanted fast, e.g. for active stabilization, not for seeding an offline fit.
    """
    if fold is None:
        fold = abs(scan.L_mm) > FOLD_L_MM
    t, y = scan.t_ps, scan.y
    base, amp, mu, sigma = gaussian_envelope_seed(t, y)
    try:
        f_hint = core_fft_peak(t, y, base, amp, mu, sigma)
        m = np.abs(t - mu) <= keep * sigma
        env = amp * np.exp(-(t[m] - mu) ** 2 / (2.0 * sigma ** 2))
        n_c = np.clip((y[m] - base - env) / env, -3.0, 3.0)
        cands = stft_seeds(t[m] - mu, n_c, sigma, fold)
    except Exception as exc:                                   # pragma: no cover
        return FitResult(False, f"seed failed: {exc}")
    extra = dict(p_seeds=cands, f_hint_ghz=f_hint)

    env_th = (base, amp, mu, sigma)
    span = max(3.0 * sigma, 1.0)
    w1 = 2.0 * np.pi * max(f_hint, 1.0) / 1e3
    scale = np.array([max(abs(base), 1e-3), amp, sigma, sigma, 1.0, 1.0,
                      w1, w1 / span, w1 / span ** 2])
    # The no-zero-crossing grid, fixed here from the *seed* envelope and used unchanged
    # by both the phase stage and the polish. See :func:`_no_zero_resid`.
    ug = np.linspace(-keep * sigma, keep * sigma, 81)

    def run(pen_scale):
        """Seed competition + polish, optionally with the no-zero-crossing penalty."""
        g = ug if pen_scale > 0 else None
        stage1, which = None, ""
        for name, pp in cands:
            r = _fit_phase_fixed_env(t, y, env_th, seed_phase_coeffs(pp), f_hint,
                                     ug=g, pen_scale=pen_scale)
            if r is not None and (stage1 is None or r.cost < stage1.cost):
                stage1, which = r, name
        if stage1 is None:                                     # pragma: no cover
            return None, None, ""
        vis0, c0, c1, c2, c3 = stage1.x
        th0 = np.array([base, amp, mu, sigma, vis0, c0, c1, c2, c3])

        # Plain least squares. A robust loss must NOT be used here: with f_scale of
        # order the noise the fringe crests are themselves treated as outliers, the
        # objective flattens, and the fit never locks onto the fringe at all. Outlier
        # handling belongs across traces, not inside one.
        def polish_resid(th):
            r = _model(th, t) - y
            if g is None:
                return r
            return np.concatenate([r, _no_zero_resid(th[6], th[7], th[8], g,
                                                     pen_scale)])

        try:
            return (least_squares(polish_resid, th0, x_scale=scale, max_nfev=20000),
                    th0, which)
        except Exception:                                      # pragma: no cover
            return None, th0, which

    def crosses(th) -> bool:
        fr = th[6] + 2.0 * th[7] * ug + 3.0 * th[8] * ug ** 2
        return bool(np.any(np.diff(np.sign(fr)) != 0))

    res, th0, which = run(0.0)
    # The constraint binds only where it is violated. Fit free first; if the answer
    # already has no zero crossing it *is* the constrained optimum, and forcing the
    # penalty on regardless only risks trading a good basin for a worse one — which is
    # exactly what it did to the dt = 2.00 ps sweep.
    constrained = False
    if not fold and res is not None and crosses(res.x):
        res2, th0_2, which2 = run(PEN_NO_ZERO * amp)
        if res2 is not None and not crosses(res2.x):
            res, th0, which, constrained = res2, th0_2, which2, True
    if res is None:
        return FitResult(False, "fit failed", seed_theta=np.zeros(9), **extra)
    extra["seed_name"] = which

    th = res.x
    # cos(-phi) = cos(phi): the whole phase polynomial may come back with either sign,
    # and least_squares picks one arbitrarily per trace. That is harmless within a
    # trace but destroys any comparison ACROSS traces — the chirp slope would flip
    # sign at random down the sweep. Canonicalise on f(0) > 0, i.e. c1 > 0.
    if th[PARAMS.index("c1")] < 0:
        th = th.copy()
        for key in ("c0", "c1", "c2", "c3"):
            th[PARAMS.index(key)] *= -1.0
        res.x = th
    r = _model(th, t) - y
    dof = max(len(t) - len(th), 1)
    sse = float(np.sum(r ** 2))
    try:
        J = res.jac
        cov = np.linalg.inv(J.T @ J) * sse / dof
    except np.linalg.LinAlgError:                              # pragma: no cover
        cov = np.full((9, 9), np.nan)

    # fringe correlation: envelope-stripped, scale invariant
    envf = th[1] * np.exp(-(t - th[2]) ** 2 / (2.0 * th[3] ** 2))
    dd = y - th[0] - envf
    mm = envf * th[4] * np.cos(th[5] + th[6] * (t - th[2]) + th[7] * (t - th[2]) ** 2
                               + th[8] * (t - th[2]) ** 3)
    den = float(np.sum(dd ** 2) * np.sum(mm ** 2))
    rho2 = float(np.sum(dd * mm) ** 2 / den) if den > 0 else float("nan")

    return FitResult(True, "ok", th, cov, sse, rho2, th0, resid=r,
                     constrained=constrained, **extra)


# ------------------------------------------------------------------- readouts

def f_uscfg_ghz(fit: FitResult, u):
    """usCFG frequency at ``u = t - mu``, GHz. Halves the detector's cos^2 doubling."""
    _, _, _, _, _, _, c1, c2, c3 = fit.theta
    u = np.asarray(u, float)
    f_obs = (c1 + 2.0 * c2 * u + 3.0 * c3 * u ** 2) / (2.0 * np.pi) * 1e3
    return f_obs / FRINGE_PER_USCFG


def readouts(fit: FitResult, tau_ps: float = TAU_PS):
    """``(f0, sigma_f0, dfus, sigma_dfus)`` in GHz, at the envelope centre.

    ``f0 = f_usCFG(0)`` and ``dfus = f_usCFG(tau/2) - f_usCFG(-tau/2)``. Both are
    linear in the phase coefficients, so their sigmas come straight from the fit
    covariance — the cubic term cancels out of ``dfus`` exactly, which is why it is
    the better-determined of the two.
    """
    k = 1e3 / (2.0 * np.pi) / FRINGE_PER_USCFG
    i1, i2 = PARAMS.index("c1"), PARAMS.index("c2")
    g0 = np.zeros(9); g0[i1] = k
    gd = np.zeros(9); gd[i2] = 2.0 * k * tau_ps
    f0 = float(g0 @ fit.theta)
    df = float(gd @ fit.theta)
    def sig(g):
        v = float(g @ fit.cov @ g)
        return float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("nan")
    return f0, sig(g0), df, sig(gd)


# ------------------------------------------------------------ the whole dataset

def fit_all():
    """Fit every sweep of both runs. → ``{run: [(scan, fit, f0, sf0, df, sdf), ...]}``.

    Per-trace only. There is deliberately **no** across-trace ("global") stage: no
    model-seeded refit, no robust re-weighting, no outlier correction.
    """
    out = {}
    for tag, fn in RUNS.items():
        scans = load_scans(fn)
        rows = []
        for s in scans:
            r = full_fit(s)
            f0, sf0, df, sdf = readouts(r) if r.ok else (np.nan,) * 4
            rows.append((s, r, f0, sf0, df, sdf))
        out[tag] = rows
    return out


def write_csv(res, path: Path):
    """``fits.csv``: one row per sweep."""
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "setpoint", "group", "L_mm", "dt_ps", "n_points",
                    "ok", "seed", "rho2", "f0_uscfg_ghz", "f0_sigma_ghz",
                    "dfus_uscfg_ghz", "dfus_sigma_ghz", "slope_ghz_per_ps",
                    "t_mu_ps", "sigma_env_ps", "fwhm_env_ps", "visibility",
                    "baseline_v", "amp_v", "nyquist_ghz", "f_obs_at_mu_ghz"])
        for tag, rows in res.items():
            for s, r, f0, sf0, df, sdf in rows:
                w.writerow([tag, s.setpoint, s.name, f"{s.L_mm:.4g}",
                            f"{s.dt_ps:.6g}", len(s.t_ps), r.ok, r.seed_name,
                            f"{r.rho2:.4f}", f"{f0:.4f}", f"{sf0:.4f}",
                            f"{df:.4f}", f"{sdf:.4f}", f"{df / TAU_PS:.6g}",
                            f"{r['mu']:.3f}", f"{r['sigma']:.3f}",
                            f"{2.3548 * r['sigma']:.3f}", f"{r['vis']:.4f}",
                            f"{r['base']:.5f}", f"{r['amp']:.5f}",
                            f"{s.nyquist_ghz:.2f}",
                            f"{f0 * FRINGE_PER_USCFG:.4f}"])


def _n_fringes(r) -> float:
    """Observed fringe cycles inside the +-KEEP_SIGMA window. The identifiability number.

    A sweep with only a handful of cycles cannot separate its envelope from its phase:
    both are smooth on the same timescale, so the fit can move a lobe by widening the
    Gaussian or by bending the phase, and least squares has no way to prefer one.
    """
    return abs(r["c1"]) / (2.0 * np.pi) * 2.0 * KEEP_SIGMA * r["sigma"]
