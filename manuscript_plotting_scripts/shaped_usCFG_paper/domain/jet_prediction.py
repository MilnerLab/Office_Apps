"""The jet-accompanying cross-correlation (Fig. 9c) predicted from the calibration constants
(ported from the paper repo's ``analysis/xcorr/joint/predict_jet.py``).

Nothing is fitted to the prediction. It is Eq. (8) evaluated with the Fig. 7 constants in
``scan_fits.json``, Dbeta linear in L (Eq. 6), Dgamma(L) from the joint fit's spectrometer
law (``joint_fit.json``), and the scan's own stage readings about the fitted origins.

**Sign of dt.** A positive delay_base shortens the delay arm, so the delay arm arrives
first. Eq. (8) puts the shaper first for dt > 0, so the model's dt is -2 delay_base / c.
The cross-correlation cannot see the sign of f, so both are folded to f(0) > 0, as
``xcorr_fit.full_fit`` reports them.

Writes ``jet_prediction.json``: the measured and predicted (f0, chirp, curvature), and the
R^2 of the raw trace under the predicted fringe (envelope from the free fit, one overall
phase fitted).
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from manuscript_plotting_scripts.shaped_usCFG_paper import config
from manuscript_plotting_scripts.shaped_usCFG_paper.domain import xcorr_fit as P

H5 = config.JET_ROOT / "Jet Truncation" / "20260904" / "XCORR_20260903_jet_accompany_scan.h5"


def run(out_dir: Path) -> None:
    """Predict the jet-run cross-correlation from the stage fits; write ``jet_prediction.json``."""
    # ---------------------------------------------------------------- measurement
    sc = P.load_scans(H5)[0]
    fit = P.full_fit(sc)
    th = fit.theta
    kk = 1e3/(2*np.pi)/P.FRINGE_PER_USCFG
    f0m, sf0m, _, _ = P.readouts(fit)
    meas = np.array([f0m, (P.f_uscfg_ghz(fit, 1) - P.f_uscfg_ghz(fit, -1))/2, 3*th[8]*kk])

    # ---------------------------------------------------------------- prediction
    F = json.load(open(out_dir/"scan_fits.json"))
    rows = json.load(open(out_dir/"joint_fit.json"))["rows"]
    B0 = F["f0"]["x"]["a"]                         # beta0/2pi, GHz/ps
    G0 = F["chirp"]["x"]["a"]*1e-3/3               # gamma0/2pi, GHz/ps^2
    L = sc.L_mm - F["zero_L"]
    dt = -(sc.dt_ps - F["zero_dt"])                # delay arm first -> dt < 0 in Eq. (8)
    DB = F["df"]["x"]["a"]*L/P.TAU_PS               # Dbeta/2pi, GHz/ps, Eq. (6) linear in L
    row = min((r for r in rows if r["tag"] == "scan_L"), key=lambda r: abs(r["L"] - sc.L_mm))
    DG = row["dgamma"]/(2*np.pi)*1e3*(L/(row["L"] - F["zero_L"]))  # Dgamma/2pi, GHz/ps^3, Delta gamma ∝ L about the fitted zero
    pred = np.array([(B0 + DB/2)*dt + 3/8*DG*dt**2, DB + 3*(G0 + DG/2)*dt, 1.5*DG])
    if pred[0] < 0:
        pred = -pred                               # fold as the fit does
    for n, m, p in zip(("f0 (GHz)", "chirp (GHz/ps)", "curv (GHz/ps^2)"), meas, pred):
        print(f"{n:16s} measured {m: .5g}  predicted {p: .5g}  ({(m-p)/p*100:+.1f} %)")

    # ---------------------------------------------------------------- the trace
    t, y = sc.t_ps, sc.y
    base, amp, mu, sigma, vis = th[:5]
    v = t - mu
    k = 2*np.pi*1e-3*P.FRINGE_PER_USCFG            # observed fringe runs at 2 f_CFG
    m = np.abs(v) <= P.KEEP_SIGMA*sigma

    def model(c0, p=pred):
        phi = c0 + k*(p[0]*v + p[1]*v*v/2 + p[2]*v**3/3)
        return base + amp*np.exp(-v**2/(2*sigma**2))*(1 + vis*np.cos(phi))

    best = min((least_squares(lambda c: (model(c[0]) - y)[m], [c0])
                for c0 in np.linspace(-np.pi, np.pi, 9)), key=lambda r: r.cost)
    yp = model(best.x[0])
    r2 = 1 - np.sum((y - yp)[m]**2)/np.sum((y[m] - y[m].mean())**2)
    print(f"trace: one phase fitted, R^2 = {r2:.3f} over ±{P.KEEP_SIGMA} sigma")

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dict(measured=meas.tolist(), s_f0=sf0m, predicted=pred.tolist(), L_eff=L, dt_model=dt,
                   dbeta=DB, dgamma=DG, trace_r2=float(r2)),
              open(out_dir/"jet_prediction.json", "w"), indent=1)
