"""Truncation characterization (2026-09-18 data): cut wavelength, edge widths, terminal
frequency (ported from the paper repo's ``analysis/truncation/run_truncation.py``).

Three datasets under ``config.TRUNCATION_ROOT``, all at the jet oscillation scan's
centrifuge settings (GA = -75 mm, i.e. L = -103.12 mm; DA = 20.42 mm, i.e. dt = 9.78 ps):

  spectrometer/<x>.csv
      the shaper arm's spectrum with the truncation prism at x mm, 141 positions from
      10.00 to 17.00 mm in 0.05 mm steps. Columns wavelength_nm, intensity. The blocked
      side of the spectrum goes to the dark pedestal, so these are single-arm spectra
      (or two-arm spectra with the delay arm absent).
  xcorr/Scan8_CFG_GA_Trunc1, Scan9_CFG_GA_Trunc2
      cross-correlation across the temporal edge with the prism at 12.86 and 14.50 mm,
      LabVIEW format (first column probe stage mm, then 20 repeats), probe stage zero at
      64.500 mm double passed (the jet convention). After the edge the signal falls to
      the noise floor, so the delay arm is absent or strongly attenuated in these scans.
  XCORR_20260903_jet_accompany_scan.h5
      the untruncated cross-correlation of the same centrifuge, fitted with
      ``domain/xcorr_fit.py`` unchanged; it supplies f_CFG(u) with u referenced to
      its envelope centre.

Chain:
  1. Every spectrum is fitted as  scale * S_open(lam) * (1/2) erfc((lam_cut - lam)/(sqrt2 sigma_lam)) + off
     with S_open the Gaussian fitted to the fully open spectrum (prism at 16.75-17.00 mm).
     Gives lam_cut(x). sigma_lam is a nuisance parameter: at a ~7 mm input beam the
     Fourier-plane spot is ~8 pm, so the fitted width is the spectrometer's resolution.
  2. lam_cut(x) is fitted with a line over the positions where the cut lies within
     +-2 sigma of the spectral envelope. That is the calibration curve.
  3. lam_cut -> omega_cut -> t' with the SHAPER ARM's map, omega(t') = omega0 + 2 beta_s t' + 3 gamma_s t'^2,
     beta_s = beta0 + Delta beta(L), then t_cut = t' - dt/2 (Eq. tcut_inv of the paper).
     omega0 is the open spectrum's centre. beta0, gamma0 and the Delta beta(L) law are the
     paper's Sec. IV B values.
  4. The two truncated cross-correlations are fitted with a plain erfc edge (the delay arm
     is absent, so the ripples before the cut are the chirp-limited edge's Fresnel fringes,
     not a two-arm beat; they are left in the residuals). They give t_cut directly on the jet axis
     for two prism positions. The spectrometer-derived t_cut for the same two positions
     differ from them by one constant (the map's zero is the spectral centre, the
     cross-correlation's zero is its envelope centre), which is fitted as ONE offset. The
     two positions fix it to ~1 ps, which is the test of the map's scale.
  5. f_trunc(x) = f_CFG(u_cut(x)) from the accompanying fit.

`calibrate` runs the chain and returns everything, including what only the figure
(``fig_char_truncation``) draws; `run` writes ``truncation_calibration.json`` and
``lamcut.csv``.

The public functions (`load_xc`, `fit_edge`, `calibrate`) take and return base_core
quantities and convert to numpy once, at their boundary. The underscored helpers are
the chain's numpy steps: the spectra are a 2-D block (141 spectra over the band), and
the dispersion and Fresnel quantities (rad/ps^2, ps^2) have no base_core type.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc

from base_core.lab_specifics.base_models import Measurement, ScanDataBase
from base_core.lab_specifics.helpers import calculate_time_delay
from base_core.math.functions import erfc as step, gaussian
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Length, Time
from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import xcorr_fit as P
from manuscript_plotting_scripts.shaped_usCFG_paper.domain.jet import TypedLaw

DATA = config.TRUNCATION_ROOT
SPEC_DIR = DATA / "spectrometer"
XC = {12.86: DATA / "xcorr" / "Scan8_CFG_GA_Trunc1" / "20260918532_.csv",
      14.50: DATA / "xcorr" / "Scan9_CFG_GA_Trunc2" / "20260918537_.csv"}
H5 = DATA / "XCORR_20260903_jet_accompany_scan.h5"

C_NM_PS = 299792.458
JET_STAGE_ZERO_MM = 64.500          # LabVIEW probe stage zero, double passed (domain/jet.py)
JET_OFFSET_PS = 27.7                # jet axis = xcorr envelope-centre axis + 27.7 ps (Fig. 9)
PRISM_JET = (13.5, 14.5, 15.0)      # the three release positions of Fig. 10

# Sec. IV B calibration constants (cross-correlation column)
BETA0_GHZ_PS = 6.399
GAMMA0_3_MHZ_PS2 = 2.952            # 3 gamma0 / 2 pi
DF_SLOPE_GHZ_MM = 0.7021            # d Delta f / dL at 0
DF0_GHZ = 1958.1                    # saturation scale = tau beta0 / 2 pi
BETA0 = 2 * np.pi * BETA0_GHZ_PS * 1e-3                 # rad/ps^2
GAMMA0 = 2 * np.pi * GAMMA0_3_MHZ_PS2 * 1e-6 / 3.0       # rad/ps^3


@dataclass(frozen=True)
class SpectrumCut:
    """The erfc-edge fit to one spectrometer spectrum (step 1). NaN where the prism
    blocks the whole spectrum."""
    prism: Length          # truncation prism position
    lam_cut: Length        # fitted cut wavelength
    sigma_lam: Length      # fitted edge width: the spectrometer's resolution, a nuisance
    lam_cut_err: Length    # standard error of lam_cut
    scale: float           # amplitude against the open spectrum


@dataclass(frozen=True)
class EdgeFit:
    """The plain erfc fit to one truncated cross-correlation (step 4, `fit_edge`)."""
    tc: Time               # the cut, on the scan's delay axis
    tc_err: Time
    sigma: Time            # erfc width
    sigma_err: Time
    w1090: Time            # 10-90 width, 2.563 sigma
    chi2_dof: float
    pre: float             # signal level before the cut
    post: float            # signal level after the cut
    model: TypedLaw        # delay -> signal (float), the fitted curve


@dataclass(frozen=True)
class EdgeScan(ScanDataBase):
    """A truncated cross-correlation on the jet delay axis, Measurement(mean, standard
    error of the 20 repeats), with the prism position and its erfc fit."""
    prism: Length | None = None
    fit: EdgeFit | None = None


def _dbeta_of_L(L_mm):
    lin = DF_SLOPE_GHZ_MM * L_mm
    return 2 * np.pi * (lin / (1.0 - lin / DF0_GHZ)) * 1e-3 / P.TAU_PS


# ----------------------------------------------------------------------------- spectra

def _load_spec(f):
    a = np.loadtxt(f, delimiter=",", skiprows=1)
    return a[:, 0], a[:, 1]


def _fit_spectra():
    files = sorted(glob.glob(str(SPEC_DIR / "*.csv")))
    lam, _ = _load_spec(files[0])
    band = (lam > 780) & (lam < 825)
    l = lam[band]

    def spec(f):
        _, I = _load_spec(f)
        return I[band] - np.percentile(I, 1)          # dark pedestal: 1st percentile

    ref = np.mean([spec(f) for f in files[-6:]], 0)  # prism 16.75 .. 17.00 mm: open
    pg = least_squares(lambda p: gaussian(l, *p) - ref, [ref.max(), 802, 4, 0]).x
    rows = []
    for f in files:
        x = float(os.path.basename(f)[:-4])
        s = spec(f)
        if s.max() < 20:
            rows.append((x, np.nan, np.nan, np.nan, np.nan))
            continue
        m = lambda p: p[0] * gaussian(l, *pg) * 0.5 * erfc((p[1] - l) / (np.sqrt(2) * p[2])) + p[3]
        lc0 = l[np.where(s > 0.5 * s.max())[0][0]]
        best = None
        for lc in (lc0 - 1.0, lc0 - 0.3, lc0, lc0 + 0.5):
            r = least_squares(lambda p: m(p) - s, [1.0, lc, 0.3, 0.0],
                              bounds=([0.2, 780, 0.02, -20], [3, 825, 5, 20]), max_nfev=3000)
            if best is None or r.cost < best.cost:
                best = r
        p = best.x
        J = best.jac
        try:
            cov = np.linalg.inv(J.T @ J) * (2 * best.cost / (len(l) - 4))
            sd = np.sqrt(np.abs(np.diag(cov)))
        except np.linalg.LinAlgError:
            sd = np.full(4, np.nan)
        rows.append((x, p[1], p[2], sd[1], p[0]))
    rows = np.array(rows)
    return l, pg, rows, files, spec


def _calibration_line(rows, pg, nsig=2.0):
    ok = np.isfinite(rows[:, 1]) & (rows[:, 1] > pg[1] - nsig * pg[2]) & (rows[:, 1] < pg[1] + nsig * pg[2])
    A = np.vstack([rows[ok, 0], np.ones(ok.sum())]).T
    c, *_ = np.linalg.lstsq(A, rows[ok, 1], rcond=None)
    resid = rows[ok, 1] - A @ c
    cov = np.linalg.inv(A.T @ A) * resid.var(ddof=2)
    return ok, c, np.sqrt(np.diag(cov)), resid


# ------------------------------------------------------------------ cross-correlations

def load_xc(path) -> ScanDataBase:
    """One truncated cross-correlation on the jet delay axis: Measurement(mean, standard
    error) of the 20 repeats per delay. ``run_id`` is the file's acquisition number."""
    a = np.loadtxt(path)
    x, Y = a[:, 0], a[:, 1:]
    zero = Length(JET_STAGE_ZERO_MM, Prefix.MILLI)
    delays = [calculate_time_delay(Length(xi, Prefix.MILLI), zero) for xi in x]
    y, e = Y.mean(1), Y.std(1, ddof=1) / np.sqrt(Y.shape[1])
    return ScanDataBase(delays=delays,
                        measured_values=[Measurement(float(yi), float(ei)) for yi, ei in zip(y, e)],
                        run_id=int(Path(path).stem.rstrip("_")))


def fit_edge(scan: ScanDataBase) -> EdgeFit:
    """Plain erfc edge (a Gaussian-blurred step). The ripples before the cut (the Fresnel
    fringes of the chirp-limited edge, Sec. II C) are left in the residuals; the errors are
    scaled by chi2/dof."""
    t = np.array([d.value(Prefix.PICO) for d in scan.delays])
    y = np.array([m.value for m in scan.measured_values])
    e = np.array([m.error for m in scan.measured_values])

    def model(p, t=t):
        b, A, tc, s = p
        # base_core's erfc is the rising step 0.5 A erf(z) + offset; amplitude -A and
        # offset b + A/2 make it b + A erfc(z)/2 exactly, in the same parameters.
        return step(t, -A, tc, abs(s), b + 0.5 * A)

    half = t[np.argmin(np.abs(y - 0.5 * (y[:20].mean() + y[-5:].mean())))]
    p0 = [y[-5:].mean(), y[:20].mean() - y[-5:].mean(), half, 3.0]
    best = least_squares(lambda p: (model(p) - y) / e, p0, max_nfev=4000)
    p = best.x
    dof = len(t) - len(p)
    cov = np.linalg.inv(best.jac.T @ best.jac) * (2 * best.cost / dof)
    sd = np.sqrt(np.diag(cov))
    ps = lambda v: Time(v, Prefix.PICO)
    return EdgeFit(tc=ps(p[2]), tc_err=ps(sd[2]), sigma=ps(abs(p[3])), sigma_err=ps(sd[3]),
                   w1090=ps(2.563 * abs(p[3])), chi2_dof=float(2 * best.cost / dof),
                   pre=float(p[0] + p[1]), post=float(p[0]),
                   model=TypedLaw(lambda tq: model(p, tq), Prefix.PICO, None))


# --------------------------------------------------------------------------- the map

def _fresnel_edge(gdd_ps2, sigma_rad_ps, tau):
    """Intensity |(1/2) erfc(tau/Lambda)|^2, Lambda^2 = 2 GDD^2 sigma^2 + 2i GDD, normalised
    far before the cut. Eq. (edge_exact) of the paper."""
    Lam = np.sqrt(2 * gdd_ps2 ** 2 * sigma_rad_ps ** 2 + 2j * gdd_ps2)
    if Lam.imag < 0:
        Lam = -Lam
    I = np.abs(0.5 * erfc(tau / Lam)) ** 2
    return I / I[0], abs(Lam)


def _fresnel_1090(gdd_ps2, sigma_rad_ps):
    """10-90 width of the intensity edge, and its width read the way the data are read:
    a Gaussian-blurred step fitted to the same curve (sigma of that erfc)."""
    tau = np.linspace(-60, 60, 240001)
    I, Lam = _fresnel_edge(gdd_ps2, sigma_rad_ps, tau)
    i90 = np.where(I > 0.9)[0][-1]
    i10 = np.where(I < 0.1)[0][0]
    m = np.abs(tau) < 30
    r = least_squares(lambda p: step(tau[m], -p[1], p[2], abs(p[3]), p[0] + 0.5 * p[1]) - I[m],
                      [0.0, 1.0, 0.0, 3.0])
    return float(tau[i10] - tau[i90]), float(Lam), float(abs(r.x[3])), float(r.x[2])


def calibrate() -> dict:
    """Run steps 1-5. Returns, keyed by name:

    ``l``          list[Length]: the fitted band's wavelengths (panel (a)'s image axis)
    ``files``      list[str]: the spectrometer files, in prism order
    ``spec``       spec(file) -> numpy array over ``l``: counts above the dark pedestal.
                   The spectra stay numpy: stacked they are panel (a)'s 2-D image.
    ``rows``       list[SpectrumCut]: the per-spectrum fits (``lamcut.csv``)
    ``ok``         list[bool] over ``rows``: the subset the calibration line is fitted to
    ``lam_of_x``   TypedLaw, prism Length -> Length: the calibration line lam_cut(x)
    ``f_of_u``     TypedLaw, Time -> Frequency: f_CFG(u), u about the xcorr envelope centre
    ``u_cut``      TypedLaw, prism Length -> Time: the cut on the envelope-centre axis
    ``t_jet_cut``  TypedLaw, prism Length -> Time: the cut on the jet axis
    ``edges``      list[EdgeScan]: the two measured edges (on the jet axis), each with
                   its prism position and `EdgeFit` (panel (b))
    ``edge_theory``, ``res``  the JSON record (``truncation_calibration.json``), plain
                   floats in the units the key names state; several quantities in it
                   (rad/ps^2, ps^2, ratios) have no base_core type.
    Every TypedLaw takes a sequence and returns a list; ``.numpy`` is the same law on
    floats (mm, nm, ps, GHz) for dense curves.
    """
    # 1-2. spectra
    l, pg, rows, files, spec = _fit_spectra()
    ok, c, c_err, resid = _calibration_line(rows, pg)
    lam_of_x = lambda x: c[0] * x + c[1]

    # accompanying cross-correlation -> f_CFG(u)
    sc = P.load_scans(H5)[0]
    fit = P.full_fit(sc)
    if not fit.ok:
        raise RuntimeError(fit.status)
    L, dt = sc.L.value(Prefix.MILLI), sc.dt.value(Prefix.PICO)
    f_of_u = fit.f_uscfg.numpy

    # 3. map: shaper arm's frame
    lam0 = pg[1]
    w0 = 2 * np.pi * C_NM_PS / lam0
    bs = BETA0 + _dbeta_of_L(L)
    gs = GAMMA0                                 # Delta gamma(L) is ~0 in the joint fit

    def t_prime(lam):
        dw = 2 * np.pi * C_NM_PS / lam - w0
        return dw / (bs + np.sqrt(bs ** 2 + 3 * gs * dw))

    u_map = lambda x: t_prime(lam_of_x(x)) - dt / 2.0     # xcorr-centre frame, before anchoring

    # 4. the two measured edges
    edges = []
    for x, path in XC.items():
        sc_x = load_xc(path)
        edges.append(EdgeScan(delays=sc_x.delays, measured_values=sc_x.measured_values,
                              run_id=sc_x.run_id, prism=Length(x, Prefix.MILLI), fit=fit_edge(sc_x)))
    tc = {x: E.fit.tc.value(Prefix.PICO) for x, E in zip(XC, edges)}
    # one offset between the map's zero and the cross-correlation axis
    offs = np.array([tc[x] - JET_OFFSET_PS - u_map(x) for x in XC])
    anchor = float(offs.mean())
    anchor_spread = float(np.ptp(offs))
    u_cut = lambda x: u_map(x) + anchor
    t_jet_cut = lambda x: u_cut(x) + JET_OFFSET_PS

    # 5. the chirp-limited floor for a perfectly sharp cut, at each cut's local GDD
    #    GDD = 1/(2 beta_s + 6 gamma_s t'), the reciprocal of the shaper arm's chirp slope at the cut
    edge_theory = {}
    for x in XC:
        tp = float(t_prime(lam_of_x(x)))
        gdd = 1.0 / (2 * bs + 6 * gs * tp)
        w_fres, _, s_fres, _ = _fresnel_1090(gdd, 0.0)
        edge_theory[str(x)] = dict(t_prime_ps=tp, gdd_ps2=gdd, dt_F_ps=float(np.sqrt(2 * gdd)),
                                   w1090_fresnel_ps=w_fres, sigma_fit_fresnel_ps=s_fres)

    def edge_record(F: EdgeFit) -> dict:
        """The fit's JSON entry: Time fields in ps, the rest as they are, no model."""
        out = {}
        for f in fields(F):
            v = getattr(F, f.name)
            if f.name != "model":
                out[f.name] = v.value(Prefix.PICO) if isinstance(v, Time) else v
        return out

    res = dict(
        L_mm=L, dt_ps=dt, lam0_nm=lam0, sigma_open_nm=pg[2],
        calib=dict(slope_nm_per_mm=c[0], slope_err=c_err[0], intercept_nm=c[1], intercept_err=c_err[1],
                   rms_resid_nm=float(resid.std()), n=int(ok.sum()),
                   x_range=[float(rows[ok, 0].min()), float(rows[ok, 0].max())]),
        map=dict(beta_s_over_beta0=bs / BETA0, dbeta_rad_ps2=_dbeta_of_L(L), dt_per_mm_ps=float((u_map(14.5) - u_map(13.5))),
                 anchor_ps=anchor, anchor_spread_ps=anchor_spread, jet_offset_ps=JET_OFFSET_PS),
        edges={str(x): edge_record(E.fit) for x, E in zip(XC, edges)},
        edge_theory=edge_theory,
        f_cfg=dict(f0_ghz=float(f_of_u(0.0)), slope_ghz_ps=float((f_of_u(1.0) - f_of_u(-1.0)) / 2),
                   mu_ps=float(fit["mu"]), sigma_ps=float(fit["sigma"])),
        releases={str(x): dict(lam_cut_nm=float(lam_of_x(x)), u_cut_ps=float(u_cut(x)),
                               t_jet_ps=float(t_jet_cut(x)), f_trunc_ghz=float(f_of_u(u_cut(x))),
                               f_trunc_pm5ps=[float(f_of_u(u_cut(x) - 5)), float(f_of_u(u_cut(x) + 5))])
                  for x in list(XC) + list(PRISM_JET)},
    )
    for x in XC:
        res["releases"][str(x)]["t_jet_measured_ps"] = tc[x]

    nm = lambda v: Length(v, Prefix.NANO)
    cuts = [SpectrumCut(prism=Length(r[0], Prefix.MILLI), lam_cut=nm(r[1]), sigma_lam=nm(r[2]),
                        lam_cut_err=nm(r[3]), scale=float(r[4])) for r in rows]
    return dict(l=[nm(v) for v in l], files=files, spec=spec, rows=cuts, ok=[bool(k) for k in ok],
                lam_of_x=TypedLaw(lam_of_x, Prefix.MILLI, Length, Prefix.NANO),
                f_of_u=TypedLaw(f_of_u, Prefix.PICO, Frequency, Prefix.GIGA),
                u_cut=TypedLaw(u_cut, Prefix.MILLI, Time, Prefix.PICO),
                t_jet_cut=TypedLaw(t_jet_cut, Prefix.MILLI, Time, Prefix.PICO),
                edges=edges, edge_theory=edge_theory, res=res)


def run(out_dir: Path) -> None:
    """Run the calibration; write ``lamcut.csv`` and ``truncation_calibration.json``."""
    k = calibrate()
    edge_theory, res = k["edge_theory"], k["res"]
    lam0, L, dt = res["lam0_nm"], res["L_mm"], res["dt_ps"]
    anchor, anchor_spread = res["map"]["anchor_ps"], res["map"]["anchor_spread_ps"]
    cal = res["calib"]
    rows = np.array([(r.prism.value(Prefix.MILLI), r.lam_cut.value(Prefix.NANO),
                      r.sigma_lam.value(Prefix.NANO), r.lam_cut_err.value(Prefix.NANO), r.scale)
                     for r in k["rows"]])

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "lamcut.csv", rows, delimiter=",",
               header="prism_mm,lam_cut_nm,sigma_lam_nm,lam_cut_err_nm,scale")
    json.dump(res, open(out_dir / "truncation_calibration.json", "w"), indent=1)

    print(f"open spectrum: centre {lam0:.3f} nm, sigma {res['sigma_open_nm']:.3f} nm; L = {L:.2f} mm, dt = {dt:.3f} ps")
    print(f"calibration: lam_cut = {cal['slope_nm_per_mm']:.4f} x + {cal['intercept_nm']:.3f}  "
          f"(rms {cal['rms_resid_nm']:.4f} nm over {cal['n']} points)")
    print(f"map: beta_s/beta0 = {res['map']['beta_s_over_beta0']:.4f}; {res['map']['dt_per_mm_ps']:.1f} ps per mm of prism")
    for x in XC:
        E = res["edges"][str(x)]
        print(f"xcorr edge, prism {x}: t_cut = {E['tc']:.2f} +- {E['tc_err']:.2f} ps (jet axis), "
              f"sigma {E['sigma']:.2f} +- {E['sigma_err']:.2f} ps, 10-90 {E['w1090']:.2f} ps; "
              f"post/pre {E['post']/E['pre']:.3f}, chi2/dof {E['chi2_dof']:.0f}")
    print(f"anchor: map zero sits {anchor:.1f} ps after the xcorr envelope centre; the two edges agree to {anchor_spread:.1f} ps")
    for x in XC:
        T, E = edge_theory[str(x)], res["edges"][str(x)]
        print(f"edge theory, prism {x}: t' {T['t_prime_ps']:.0f} ps, local GDD {T['gdd_ps2']:.2f} ps^2, "
              f"dt_F {T['dt_F_ps']:.2f} ps, sharp-cut floor read as a blurred step: sigma {T['sigma_fit_fresnel_ps']:.2f} ps "
              f"(measured {E['sigma']:.2f} +- {E['sigma_err']:.2f})")
    for x in list(XC) + list(PRISM_JET):
        r = res["releases"][str(x)]
        print(f"prism {x}: lam_cut {r['lam_cut_nm']:.2f} nm -> t_cut {r['t_jet_ps']:.0f} ps (jet axis) -> f_trunc {r['f_trunc_ghz']:.1f} GHz")
