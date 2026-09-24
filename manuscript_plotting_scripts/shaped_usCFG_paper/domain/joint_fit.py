"""The joint pinball-envelope fit behind the characterization, and the seed fits it starts from.

Two stages, run in this order:

:func:`run_seeds` -> ``seeds.npy`` (ported from the 2026-09-15 scratchpad normfit.py, later
the paper repo's ``analysis/xcorr/joint/seeds.py``).

    Step 4 refit on the NORMALISED fringe, then denormalised. All 17 sweeps, same way.

    The stock polish minimises (model - y) in volts, so a wing point with 1/50th the
    envelope contributes 1/2500th the cost: the fit is blind to the wings, which is where
    the chirp's lever arm is. The Hilbert path doesn't have this problem because it divides
    by the envelope first.

    So: divide by the envelope, fit, multiply back. In least squares that is exactly a
    weight w = 1/env on the residual. The envelope used for w is the one already fitted
    (base, amp, mu, sigma) and is HELD FIXED while weighting, so this is a pure change of
    metric, not a new free parameter.

    Outside the read window env -> 0 and 1/env explodes, amplifying pure baseline noise, so
    w is floored at its value at KEEP_SIGMA. Inside the window: equal FRACTIONAL weight,
    which is the right metric for noise that is 24% of the local envelope. Outside: equal
    absolute weight, which still pins the baseline.

    Identical treatment for every sweep. No sweep is seeded, pinned or excluded by hand.

:func:`run_joint` -> ``joint_fit.json`` and ``fit_demo.npz`` (ported from the 2026-09-17
scratchpad pingal3.py, later the paper repo's ``analysis/xcorr/joint/joint_fit.py``). Reads
``seeds.npy``, ``spec_fits.csv`` and the two ``_spec_aligned_*.npz`` spectrometer caches from
``config.TEMP_DIR``, and the frozen inputs in ``domain/inputs/``.

    Per-sweep fit as in the earlier per-trace round (pingal2.py), with ONE change: the pinball
    envelopes are fitted for all traces of a scan SIMULTANEOUSLY, their centres tied by
    Kevin's geometry.

        scan_d (L = 0) : mu_i = mu0 - dt_i/2
        scan_L (dt = 0): mu_i = mu0 + (2/c)(0.00486) L_i       (probe tracks the grating 1:1)

    mu0 is shared per scan; amplitude, width and offset stay per trace. Only the UPPER Gaussian
    is centred at mu_i; the gap Gaussian's centre is free per trace. mu0 minimises the summed
    pinball loss of all traces (each trace's loss divided by its own signal span so every trace
    counts equally). The phase origin u = t - mu_i and the fit window follow the same centre.
    Everything else -- the phase fit, the lambda -> t map, the frozen overlay, the free
    spectrometer fit -- is the per-trace round unchanged.

Knobs read from the environment exist only for experiments, and their defaults are the
published run: ``TAUS`` (0.95), ``DGAMMA`` (1), ``DBAR`` (0).
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, curve_fit, minimize, minimize_scalar

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import xcorr_fit as P
from manuscript_plotting_scripts.shaped_usCFG_paper.domain.xcorr_fit import RUNS, fit_all, _n_fringes

K = 1e6 / (2.0 * np.pi)


# ================================================================== seeds
def weights(theta, t):
    """w = 1/env, floored at the envelope's value at KEEP_SIGMA. Fixed during the fit."""
    base, amp, mu, sigma = theta[:4]
    env = amp * np.exp(-(t - mu) ** 2 / (2.0 * sigma ** 2))
    floor = amp * np.exp(-P.KEEP_SIGMA ** 2 / 2.0)
    return 1.0 / np.maximum(env, floor)


def refit_normalised(scan, r):
    """Re-polish on the normalised fringe. Multi-start, exactly as full_fit does."""
    t, y = scan.t_ps, scan.y
    w = weights(r.theta, t)

    def resid(th):
        return (P._model(th, t) - y) * w

    span = max(3.0 * r["sigma"], 1.0)
    w1 = 2.0 * np.pi * max(r.f_hint_ghz, 1.0) / 1e3
    scale = np.array([1.0, 1.0, span, span, 1.0, 1.0, w1, w1 / span, w1 / span ** 2])

    starts = [r.theta.copy()]
    # the same STFT seeds the stock fit chooses between, so no basin is privileged
    for _, pp in r.p_seeds:
        try:
            c = P.seed_phase_coeffs(pp)
            th = r.theta.copy()
            th[6], th[7], th[8] = c
            starts.append(th)
        except Exception:
            pass

    best = None
    for th0 in starts:
        try:
            rr = least_squares(resid, th0, x_scale=scale, max_nfev=20000)
        except Exception:
            continue
        if best is None or rr.cost < best.cost:
            best = rr
    th = best.x.copy()
    if th[P.PARAMS.index("c1")] < 0:                 # same sign convention as xcorr_fit
        for key in ("c0", "c1", "c2", "c3"):         # phi -> -phi; cos is even, so
            th[P.PARAMS.index(key)] *= -1.0          # the model is exactly unchanged
    return th, 2.0 * best.cost


def origin(x, y):
    return float(np.sum(x * y) / np.sum(x * x))


def run_seeds(out_dir: Path) -> None:
    """Refit every sweep on the envelope-normalised fringe and write ``seeds.npy``."""
    res = fit_all()
    rows = []
    for tag in ("scan_d", "scan_L"):
        for idx in range(len(res[tag])):
            scan, r = res[tag][idx][0], res[tag][idx][1]
            th, _ = refit_normalised(scan, r)
            # readouts, same definitions as xcorr_fit.readouts
            k = 1e3 / (2.0 * np.pi) / P.FRINGE_PER_USCFG
            f0_new = th[6] * k
            df_new = 2.0 * k * P.TAU_PS * th[7]
            f0_old, _, df_old, _ = P.readouts(r)
            rows.append(dict(tag=tag, idx=idx, dt=scan.dt_ps, L=scan.L_mm,
                             nf=_n_fringes(r),
                             chirp_old=r["c2"] * K, chirp_new=th[7] * K,
                             f0_old=f0_old, f0_new=f0_new,
                             df_old=df_old, df_new=df_new,
                             c3_old=r["c3"], c3_new=th[8],
                             theta_old=r.theta.copy(), theta_new=th,
                             mu=r["mu"], sigma=r["sigma"]))

    print(f"{'sweep':>18} {'nfr':>4} | {'chirp old':>9} {'chirp new':>9} | "
          f"{'f0 old':>8} {'f0 new':>8} | {'df old':>8} {'df new':>8}")
    for d in rows:
        name = (f"scan_d Δt={d['dt']:5.2f}" if d["tag"] == "scan_d"
                else f"scan_L L={d['L']:7.2f}")
        print(f"{name:>18} {d['nf']:4.0f} | {d['chirp_old']:9.2f} {d['chirp_new']:9.2f} | "
              f"{d['f0_old']:8.3f} {d['f0_new']:8.3f} | {d['df_old']:8.2f} {d['df_new']:8.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "seeds.npy", rows, allow_pickle=True)

    # ---- the three relationships, old vs new
    d = [r for r in rows if r["tag"] == "scan_d"]
    L = [r for r in rows if r["tag"] == "scan_L"]
    dt = np.array([r["dt"] for r in d])
    Lm = np.abs(np.array([r["L"] for r in L]))

    print("\n--- beta0 from f0 vs dt (GHz/ps) ---")
    for k_ in ("f0_old", "f0_new"):
        v = np.array([r[k_] for r in d]); s = origin(dt, v)
        print(f"  {k_}:  {s:.4f}   resid rms {np.std(v - s*dt, ddof=1):.4f} GHz")

    print("--- gamma0 from chirp vs dt (MHz/ps^2) ---")
    for k_ in ("chirp_old", "chirp_new"):
        v = np.array([r[k_] for r in d]); s = origin(dt, v)
        p = np.polyfit(dt, v, 1)
        print(f"  {k_}:  origin {s:.4f}   free {p[0]:.4f}·Δt {p[1]:+.3f}   "
              f"resid rms {np.std(v - s*dt, ddof=1):.3f} MHz/ps")

    print("--- dbeta from df vs |L| (GHz/mm) ---")
    for k_ in ("df_old", "df_new"):
        v = np.abs(np.array([r[k_] for r in L])); s = origin(Lm, v)
        p = np.polyfit(Lm, v, 1)
        print(f"  {k_}:  origin {s:.4f}   free zero at L = {-p[1]/p[0]:+.2f} mm   "
              f"resid rms {np.std(v - s*Lm, ddof=1):.3f} GHz")


# ================================================================== joint fit
KF = 1e3 / (2*np.pi) / P.FRINGE_PER_USCFG
S_GHZ_PER_MM = 0.6996        # dbeta slope for the map, 2026-09-15 forced-zero fit
KL = 2.0 * 0.00486 / P.C_MM_PER_PS            # ps of centre per mm of L
TAU = 10.0 / 11.0          # cross-correlation
TAU_S = float(os.environ.get("TAUS", "0.95"))   # spectrometer: less noisy
USE_DGAMMA = os.environ.get("DGAMMA", "1") != "0"

# common fourth-order term of wbar(t) = w0 + 2 b t + 3 g t^2 + 4 d t^3. Off (0) in the published
# run. The earlier map diagnosis (2026-09-15, the paper repo's analysis/xcorr/out/gallery/dgG.json)
# read dbar = 3.09e-9 from the scan_d c3-vs-dt slope; nothing here reads that file.
DBAR = float(os.environ.get("DBAR", "0")) or 0.0

#: the paper's fit-demonstration panel (fig_char_fitdemo.pdf): one spectrometer sweep
DEMO_L = -43.12


def _gauss(x, a, mu, sig, off):
    return a*np.exp(-0.5*((x-mu)/sig)**2) + off


def phase_fit(u, n, mask, seed, tag):
    def resid(c):
        return (np.cos(c[0] + c[1]*u + c[2]*u**2 + c[3]*u**3) - n)[mask]
    best = None
    for sgn in (1.0, -1.0):
        for c0 in (0.0, np.pi/2, np.pi, -np.pi/2):
            p0 = np.array([c0, sgn*seed[0], sgn*seed[1], sgn*seed[2]])
            rr = least_squares(resid, p0, max_nfev=40000)
            if best is None or rr.cost < best.cost:
                best = rr
    c = best.x.copy()
    # overall sign: scan_d by f(0) > 0 (c1 > 0); scan_L, where f(0) ~ 0, by down-chirp (c2 < 0)
    if (tag == "scan_L" and c[2] > 0) or (tag != "scan_L" and c[1] < 0):
        c = -c
    return c, best


# ------------------------------------------------------------------ joint pinball envelopes
def pin_loss(r, tau=TAU):
    return float(np.sum(np.where(r > 0, tau*r, (tau - 1.0)*r)))


def upper_pinned(x, y, mu, sig0, p0=None, tau=TAU):
    g = lambda xx, a, s, off: _gauss(xx, a, mu, s, off)
    if p0 is None:
        off0 = float(np.median(y)); i = int(np.argmax(y))
        p0 = [y[i] - off0, sig0, off0]
        try:
            p0, _ = curve_fit(g, x, y, p0=p0, maxfev=10000)
        except RuntimeError:
            pass
    r = minimize(lambda p: pin_loss(y - g(x, *p), tau), p0, method="Nelder-Mead",
                 options=dict(maxiter=20000, maxfev=20000, xatol=1e-4, fatol=1e-4))
    return r.x, r.fun


def upper_free(x, y, sig0, p0=None, tau=TAU):
    """Gap Gaussian: no constraint on its centre."""
    if p0 is None:
        off0 = float(np.median(y)); i = int(np.argmax(y))
        p0 = [y[i] - off0, x[i], sig0, off0]
        try:
            p0, _ = curve_fit(_gauss, x, y, p0=p0, maxfev=10000)
        except RuntimeError:
            pass
    r = minimize(lambda p: pin_loss(y - _gauss(x, *p), tau), p0, method="Nelder-Mead",
                 options=dict(maxiter=20000, maxfev=20000, xatol=1e-4, fatol=1e-4))
    return r.x, r.fun


def env_at(tr, mu, warm=None, kx="t", ky="y"):
    t, y, sig0 = tr[kx], tr[ky], tr["sig0"]
    wU, wL = (warm or (None, None))
    tau = TAU if ky == "y" else TAU_S
    pU, lU = upper_pinned(t, y, mu, sig0, wU, tau)
    res = -(y - _gauss(t, pU[0], mu, pU[1], pU[2]))
    pL, lL = upper_free(t, res, sig0, wL, tau)
    return pU, pL, (lU + lL) / tr["span" if ky == "y" else "span_s"]


def reexpand3(c1, c2, c3, d):
    return c1 + 2*c2*d + 3*c3*d*d, c2 + 3*c3*d, c3


def label(x):
    return (f"Δt = {x['dt']:.2f} ps" if x["tag"] == "scan_d" else f"L = {x['L']:.2f} mm")


def run_joint(out_dir: Path) -> None:
    """Joint envelopes and cubic-phase fits; write ``joint_fit.json`` and ``fit_demo.npz``."""
    # beta0, gamma0 for the lambda -> t map: the 2026-09-15 fits (pinrel.json), frozen.
    REL = json.load(open(config.INPUTS_DIR / "map_calibration.json"))
    BETA0 = 2*np.pi * REL["beta0"]["free"] * 1e-3
    GAMMA0 = 2*np.pi * REL["gamma0"]["free"] * 1e-3 * 1e-3 / 3.0
    DF0 = BETA0 * P.TAU_PS / (2*np.pi) * 1e3
    NF = np.load(out_dir / "seeds.npy", allow_pickle=True)
    # previous per-trace envelopes (pingal2.json): the joint fit's mu0 start
    PREV = {(r["tag"], r["idx"]): r for r in json.load(open(config.INPUTS_DIR / "prev_envelopes.json"))}

    def dbeta_of_L(L_mm):
        lin = S_GHZ_PER_MM * L_mm
        return 2*np.pi * (lin / (1.0 - lin/DF0)) * 1e-3 / P.TAU_PS

    # Dgamma(L) from the spectrometer's psi3 slope (the law of the earlier round's mapcore.py):
    #   psi3(L) = psi3(0) + (dpsi3/dL) L,  Dgamma = -psi3/(6 psi2^3) - gamma0,  psi2 = 1/(2(beta0+Dbeta))
    PSI20 = 1.0/(2*BETA0)
    PSI30 = -6.0*GAMMA0*PSI20**3
    DPSI3_DL = None          # set by calibrate_dgamma() once the scan_L L values are known

    def dgamma_of_L(L_mm):
        if not USE_DGAMMA:
            return 0.0
        b = BETA0 + dbeta_of_L(L_mm)
        psi2 = 1.0/(2*b)
        psi3 = PSI30 + DPSI3_DL*L_mm
        return -psi3/(6*psi2**3) - GAMMA0

    def calibrate_dgamma(Ls):
        nonlocal DPSI3_DL
        rows = sorted([r for r in csv.DictReader(open(config.TEMP_DIR/"spec_fits.csv", encoding="utf8"))
                       if r["run"] == "scan_L"], key=lambda r: int(r["setpoint"]))
        p3 = np.array([float(r["psi3_ps3"]) for r in rows])
        A = np.vstack([np.asarray(Ls, float), np.ones(len(p3))]).T
        DPSI3_DL = float(np.linalg.lstsq(A, p3, rcond=None)[0][0])
        return DPSI3_DL

    def t_of_w(w, w0, bbar, gbar):
        dw = np.asarray(w, float) - w0
        t = dw / (bbar + np.sqrt(np.maximum(bbar**2 + 3*gbar*dw, 1e-12)))
        d = DBAR
        if d:
            for _ in range(60):
                f = 2*bbar*t + 3*gbar*t**2 + 4*d*t**3 - dw
                fp = 2*bbar + 6*gbar*t + 12*d*t**2
                t = t - np.clip(f/np.where(np.abs(fp) < 1e-12, 1e-12, fp), -20.0, 20.0)
        return t

    NPZ = {"scan_d": np.load(config.TEMP_DIR/"_spec_aligned_XCORR_scan_d_20260831_131421.npz"),
           "scan_L": np.load(config.TEMP_DIR/"_spec_aligned_XCORR_scan_L_20260825_200235.npz")}
    _allw = NPZ["scan_d"]["w"]
    _smean = np.mean(np.vstack([NPZ["scan_d"]["s"], NPZ["scan_L"]["s"]]), axis=0)
    _p, _ = curve_fit(_gauss, _allw, _smean,
                      p0=[_smean.max(), _allw[np.argmax(_smean)], 10.0, np.percentile(_smean, 5)],
                      maxfev=20000)
    W0 = float(_p[1])

    def joint_envelopes(TR, g, start=None, kx="t", ky="y"):
        warm = [None]*len(TR)
        cache = {}

        def J(mu0):
            key = round(float(mu0), 4)
            if key in cache:
                return cache[key][0]
            tot, fits = 0.0, []
            for i, tr in enumerate(TR):
                pU, pL, l = env_at(tr, mu0 + g[i], warm[i], kx, ky)
                warm[i] = (pU, pL); tot += l; fits.append((pU, pL))
            cache[key] = (tot, fits)
            return tot

        if start is None:
            start = float(np.median([PREV[(tr["tag"], tr["idx"])]["pU"][1] - gi for tr, gi in zip(TR, g)]))
        grid = np.arange(start - 10.0, start + 10.01, 1.0)
        vals = [J(m) for m in grid]
        m0 = float(grid[int(np.argmin(vals))])
        r = minimize_scalar(J, bounds=(m0 - 1.0, m0 + 1.0), method="bounded", options=dict(xatol=0.02))
        mu0 = float(r.x); J(mu0)
        return mu0, cache[round(mu0, 4)][1], (grid, np.array(vals))

    # ------------------------------------------------------------------ load
    traces = {"scan_d": [], "scan_L": []}
    for tag in traces:
        scans = P.load_scans(RUNS[tag])
        for sc, r in zip(scans, [q for q in NF if q["tag"] == tag]):
            y = sc.y
            traces[tag].append(dict(tag=tag, idx=sc.setpoint, scan=sc, t=sc.t_ps, y=y, nf_row=r,
                                    sig0=float(r["theta_new"][3]),
                                    span=float(np.percentile(y, 99.5) - np.percentile(y, 0.5)),
                                    dt=float(sc.dt_ps), L=float(sc.L_mm)))

    print("dpsi3/dL = %.4g ps^3/mm" % calibrate_dgamma([tr["L"] for tr in traces["scan_L"]]), flush=True)

    JOINT = {}
    for tag, TR in traces.items():
        g = np.array([(-0.5*tr["dt"]) if tag == "scan_d" else (KL*tr["L"]) for tr in TR])
        mu0, fits, prof = joint_envelopes(TR, g)
        JOINT[tag] = dict(mu0=mu0, g=g.tolist(), prof=[prof[0].tolist(), prof[1].tolist()])
        for tr, gi, (pU, pL) in zip(TR, g, fits):
            tr["mu"] = mu0 + gi; tr["pU"] = pU; tr["pL"] = pL
        print(f"{tag}: mu0 = {mu0:.3f} ps", flush=True)

    # ---------------------------------------------- the same envelopes for the spectrometer
    # The mapped spectrum gets the identical treatment: pinball Gaussian upper envelope with one
    # centre shared by every trace of a scan (the spectrometer centre does not move with the
    # stages), gap Gaussian free per trace, fitted jointly.
    for tag, TR in traces.items():
        npz = NPZ[tag]
        for tr in TR:
            j = int(np.where(npz["setpoint"] == tr["idx"])[0][0])
            w, s = npz["w"], npz["s"][j]
            bbar = BETA0 + 0.5*dbeta_of_L(tr["L"])
            gbar = GAMMA0 + 0.5*dgamma_of_L(tr["L"])
            ts = t_of_w(w, W0, bbar, gbar)
            o = np.argsort(ts); ts, s = ts[o], s[o]
            ms = np.abs(ts) <= P.KEEP_SIGMA*tr["sig0"]
            tr["ts"], tr["ss"] = ts[ms], s[ms]
            tr["span_s"] = float(np.percentile(tr["ss"], 99.5) - np.percentile(tr["ss"], 0.5))
        mu0s, fits, _ = joint_envelopes(TR, np.zeros(len(TR)), start=0.0, kx="ts", ky="ss")
        JOINT[tag]["mu0_spec"] = mu0s
        for tr, (pU, pL) in zip(TR, fits):
            tr["mu_s"] = mu0s; tr["spU"] = pU; tr["spL"] = pL
        print(f"{tag}: spectrometer mu0 = {mu0s:.3f} ps", flush=True)

    def do_one(tr):
        tag, idx, scan = tr["tag"], tr["idx"], tr["scan"]
        th = np.array(tr["nf_row"]["theta_new"], float)
        th_old = np.array(tr["nf_row"]["theta_old"], float)
        t, y = tr["t"], tr["y"]
        mu, sigma = tr["mu"], th[3]
        u = t - mu
        m = np.abs(u) <= P.KEEP_SIGMA * sigma

        pU, pL = tr["pU"], tr["pL"]
        Ud = _gauss(t, pU[0], mu, pU[1], pU[2])
        G = _gauss(t, pL[0], pL[1], pL[2], pL[3])
        Ld = Ud - G; mid = (Ud + Ld)/2.0; half = G/2.0
        hs = np.maximum(half, 1e-3*float(half.max()))
        n = (y - mid)/hs
        c, best = phase_fit(u, n, m, reexpand3(th[6], th[7], th[8], mu - th[2]), tag)
        phi = c[0] + c[1]*u + c[2]*u**2 + c[3]*u**3
        dd, mm = (y-mid)[m], (half*np.cos(phi))[m]
        den = float(np.sum(dd**2)*np.sum(mm**2))
        rho2 = float(np.sum(dd*mm)**2/den) if den > 0 else float("nan")
        J = best.jac; res = best.fun
        cov = np.linalg.inv(J.T@J) * float(np.sum(res**2))/max(len(res)-4, 1)

        ts, s = tr["ts"], tr["ss"]
        spU, spL = tr["spU"], tr["spL"]; mu_s = tr["mu_s"]
        sUd = _gauss(ts, spU[0], mu_s, spU[1], spU[2])
        Gs = _gauss(ts, spL[0], spL[1], spL[2], spL[3])
        sLd = sUd - Gs
        smid, shalf = (sUd + sLd)/2.0, Gs/2.0
        mid_at, half_at = np.interp(ts, u, mid), np.interp(ts, u, half)
        Ud_at = np.interp(ts, u, Ud)
        A = np.vstack([sUd, np.ones_like(sUd)]).T
        a_s, b_s = [float(v) for v in np.linalg.lstsq(A, Ud_at, rcond=None)[0]]
        if a_s <= 0:
            a_s = float(np.ptp(Ud_at) / max(np.ptp(sUd), 1e-12))
            b_s = float(np.mean(Ud_at) - a_s*np.mean(sUd))
        s2 = a_s*s + b_s
        sUd, sLd = a_s*sUd + b_s, a_s*sLd + b_s

        us = ts
        def r_c0(p):
            return mid_at + half_at*np.cos(p[0] + c[1]*us + c[2]*us**2 + c[3]*us**3) - s2
        b0 = None
        for gg in np.linspace(-np.pi, np.pi, 9):
            q = least_squares(r_c0, [gg], max_nfev=4000)
            if b0 is None or q.cost < b0.cost:
                b0 = q
        c0_s = float(b0.x[0])
        ddF, mmF = s2 - mid_at, half_at*np.cos(c0_s + c[1]*us + c[2]*us**2 + c[3]*us**3)
        denF = float(np.sum(ddF**2)*np.sum(mmF**2))
        rho2_frozen = float(np.sum(ddF*mmF)**2/denF) if denF > 0 else float("nan")

        shs = np.maximum(shalf, 1e-3*float(np.max(shalf)))
        ns = (s - smid)/shs
        # Seeded from the spectrometer's own STFT ridge, exactly as the cross-correlation:
        # every window length, V and quadratic |f|, no zero crossing on the delay scan.
        # The best fit to this trace wins. Nothing is taken from the cross-correlation.
        cs, bs = None, None
        for _, pp in P.stft_seeds(us, np.clip(ns, -3, 3), sigma, fold=(tag == "scan_L")):
            cc, bb = phase_fit(us, ns, np.ones_like(us, bool), tuple(P.seed_phase_coeffs(pp)), tag)
            if bs is None or bb.cost < bs.cost:
                cs, bs = cc, bb
        # same standard errors as the cross-correlation: (J^T J)^-1 * RSS/(N-4)
        covs = np.linalg.inv(bs.jac.T@bs.jac) * float(np.sum(bs.fun**2))/max(len(bs.fun)-4, 1)

        nf = int(abs(th_old[6]) / (2.0*np.pi) * 2.0 * P.KEEP_SIGMA * th_old[3])
        return dict(tag=tag, idx=idx, dt=float(scan.dt_ps), L=float(scan.L_mm), nf=nf,
                    mu=float(mu), sigma=float(sigma), rho2=rho2,
                    f0=float(c[1]*KF), sf0=float(np.sqrt(cov[1, 1])*KF),
                    chirp=float(c[2]*K), schirp=float(np.sqrt(cov[2, 2])*K),
                    df=float(2*KF*P.TAU_PS*c[2]), sdf=float(2*KF*P.TAU_PS*np.sqrt(cov[2, 2])),
                    c3=float(c[3]), ts=ts, s2=s2, sUd=sUd, sLd=sLd, c0_s=c0_s,
                    rho2_frozen=rho2_frozen, a_s=a_s, b_s=b_s, cs=cs,
                    f0_s=float(cs[1]*KF), chirp_s=float(cs[2]*K), c3_s=float(cs[3]),
                    sf0_s=float(np.sqrt(covs[1, 1])*KF), schirp_s=float(np.sqrt(covs[2, 2])*K),
                    dgamma=float(dgamma_of_L(tr["L"])),
                    pU=[float(pU[0]), float(mu), float(pU[1]), float(pU[2])],
                    spU=[float(spU[0]), float(mu_s), float(spU[1]), float(spU[2])],
                    spLn=[float(v) for v in spL],
                    pLn=[float(pL[0]), float(pL[1]), float(pL[2]), float(pL[3])])

    rows = [do_one(tr) for tag in ("scan_d", "scan_L") for tr in traces[tag]]

    out_dir.mkdir(parents=True, exist_ok=True)
    _x = min((x for x in rows if x["tag"] == "scan_L"), key=lambda x: abs(x["L"] - DEMO_L))
    np.savez(out_dir/"fit_demo.npz", L=_x["L"], ts=_x["ts"], s=(_x["s2"] - _x["b_s"])/_x["a_s"],
             sUd=(_x["sUd"] - _x["b_s"])/_x["a_s"], sLd=(_x["sLd"] - _x["b_s"])/_x["a_s"], cs=_x["cs"],
             sigma=_x["sigma"])

    json.dump(dict(joint=JOINT, rows=[{k: v for k, v in x.items()
                if k in ("tag", "idx", "dt", "L", "nf", "rho2", "rho2_frozen", "c0_s",
                         "f0", "sf0", "chirp", "schirp", "df", "sdf", "c3",
                         "f0_s", "chirp_s", "c3_s", "sf0_s", "schirp_s", "dgamma", "a_s", "b_s", "pU", "pLn", "mu", "sigma", "spU", "spLn")}
               for x in rows]), open(out_dir/"joint_fit.json", "w"), indent=1, default=float)

    print("\n%-20s | %13s | %15s | %8s %8s | %8s %8s" %
          ("sweep", "rho2 xc", "rho2 frozen", "f0 xc", "f0 spec", "chirp xc", "chirp sp"))
    print("   (curv = 3 c3 KF; Dgamma %s, dbar = %.3g)" % ("ON" if USE_DGAMMA else "OFF", DBAR))
    for x in rows:
        pv = PREV[(x["tag"], x["idx"])]
        print("%-20s | %5.3f -> %5.3f | %6.3f -> %6.3f | %8.2f %8.2f | %8.1f %8.1f | curv %8.4f %8.4f" %
              (f"{x['tag']} {label(x)}", pv["rho2"], x["rho2"], pv["rho2_frozen"], x["rho2_frozen"],
               x["f0"], x["f0_s"], x["chirp"], x["chirp_s"], 3*x["c3"]*KF*1e3, 3*x["c3_s"]*KF*1e3))
    for tg in ("scan_d", "scan_L"):
        sub = [x for x in rows if x["tag"] == tg]
        print(f"{tg}: mean rho2 xcorr {np.mean([PREV[(tg,x['idx'])]['rho2'] for x in sub]):.3f} -> "
              f"{np.mean([x['rho2'] for x in sub]):.3f};  mean rho2 frozen "
              f"{np.mean([PREV[(tg,x['idx'])]['rho2_frozen'] for x in sub]):.3f} -> "
              f"{np.mean([x['rho2_frozen'] for x in sub]):.3f}")
