"""The stage fits quoted in the paper: beta0, gamma0, dbeta slope and both stage origins
(ported from the 2026-09-17 scratchpad agg2.py, later the paper repo's
``analysis/xcorr/joint/scan_fits.py``). Reads ``joint_fit.json``; writes ``scan_fits.json``.

f0 and the chirp are fitted against dt, and Df against L with the exact reciprocal law.
Every fit has a free zero (slope and origin both free), errors are scaled by
sqrt(chi2/nu), and the delay axis is referenced to the f0 zero. Both instruments, the
cross-correlation (``x``) and the mapped spectrometer (``s``), are fitted the same way,
each weighted by its per-sweep fit standard errors.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from manuscript_plotting_scripts.shaped_usCFG_paper.domain import xcorr_fit as P


def linfit(x, y, s=None):
    """y = a (x - z), both free. chi2-scaled errors when s is given."""
    w = np.ones_like(y) if s is None else 1/np.asarray(s)**2
    A = np.vstack([x, np.ones_like(x)]).T
    W = np.diag(w)
    C = np.linalg.inv(A.T@W@A); p = C @ (A.T@W@y)
    r = y - A@p; dof = len(x) - 2; chi2 = float(np.sum(w*r*r)); C = C*(chi2/dof)
    z = -p[1]/p[0]; g = np.array([p[1]/p[0]**2, -1/p[0]])
    return dict(a=float(p[0]), sa=float(np.sqrt(C[0, 0])), z=float(z),
                sz=float(np.sqrt(g@C@g)), rms=float(np.std(r, ddof=2)),
                chi2=chi2, dof=dof)


def run(out_dir: Path) -> None:
    """Fit the three stage relationships and write ``scan_fits.json``."""
    rows = json.load(open(out_dir/"joint_fit.json"))["rows"]
    d = [x for x in rows if x["tag"] == "scan_d"]
    L = [x for x in rows if x["tag"] == "scan_L"]

    # ---- central frequency, delay scan
    dt = np.array([x["dt"] for x in d])
    f0 = np.array([x["f0"] for x in d]); sf0 = np.array([x["sf0"] for x in d])
    f0s = np.array([x["f0_s"] for x in d]); sf0s = np.array([x["sf0_s"] for x in d])
    Bx = linfit(dt, f0, sf0); Bs = linfit(dt, f0s, sf0s)
    Z = Bx["z"]                                  # the recalibrated delay origin

    # ---- chirp, delay scan (gamma0), same recalibrated axis
    ch = np.array([x["chirp"] for x in d]); sch = np.array([x["schirp"] for x in d])
    chs = np.array([x["chirp_s"] for x in d]); schs = np.array([x["schirp_s"] for x in d])
    Gx = linfit(dt, ch, sch); Gs = linfit(dt, chs, schs)

    # ---- bandwidth, grating scan: exact reciprocal law, slope and origin both free
    Lm = np.array([x["L"] for x in L])
    df = np.array([x["df"] for x in L]); sdf = np.array([x["sdf"] for x in L])
    dfs = np.array([P.TAU_PS*x["chirp_s"]*1e-3 for x in L])
    sdfs = np.array([P.TAU_PS*x["schirp_s"]*1e-3 for x in L])
    DF0 = Bx["a"]*P.TAU_PS

    def lawfit(y, s=None):
        def res(p):
            x = p[0]*(Lm - p[1])
            m = x/(1.0 - x/DF0)
            return (m - y) if s is None else (m - y)/s
        r = least_squares(res, [0.7, 0.0])
        chi2 = float(np.sum(r.fun**2)); dof = len(Lm) - 2
        C = np.linalg.inv(r.jac.T@r.jac)*(chi2/dof)
        return dict(a=float(r.x[0]), sa=float(np.sqrt(C[0, 0])), z=float(r.x[1]),
                    sz=float(np.sqrt(C[1, 1])), chi2=chi2, dof=dof,
                    rms=float(np.std(r.fun*(1 if s is None else s), ddof=2)))

    Ex = lawfit(df, sdf); Es = lawfit(dfs, sdfs)
    ZL = Ex["z"]

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dict(f0=dict(x=Bx, s=Bs), chirp=dict(x=Gx, s=Gs), df=dict(x=Ex, s=Es),
                   zero_dt=Z, zero_L=ZL, df0=DF0),
              open(out_dir/"scan_fits.json", "w"), indent=1, default=float)
    for k, v in (("f0", (Bx, Bs)), ("chirp", (Gx, Gs)), ("df", (Ex, Es))):
        print(k, "xcorr a=%.4f+-%.4f z=%+.4f" % (v[0]["a"], v[0]["sa"], v[0]["z"]),
              " spec a=%.4f z=%+.4f" % (v[1]["a"], v[1]["z"]))
