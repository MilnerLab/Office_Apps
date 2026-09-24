"""The two centrifuge arms and the corkscrew they make.

Theory schematic, no data. Four columns, each a pair of stacked panels on one time axis:

  top     the time-frequency density of the two interferometer arms, each a chirped
          Gaussian about its instantaneous frequency
              omega(t) = 2*beta*(t -+ dt/2) + 3*gamma*(t -+ dt/2)**2      (Eq. chirpphase)
          with a thin iso-intensity contour around it
  bottom  the corkscrew field, turning at
              Omega(t) = [omega_s(t) - omega_d(t)] / 2                     (Eq. Omega_def)
          i.e. half the vertical gap between the two ridges drawn above; the twist phase is
          its running integral, so the rows are registered by construction

  (a) small dt                  (b) large dt
  (c) (a) plus extra shaper chirp dbeta
  (d) (a) truncated at t_cut (a frequency cut at omega_cut in the Fourier plane)

Model parameters (illustrative, dimensionless): base chirp BETA0, base TOD GAMMA0,
envelope width ENV_W, per-column delay dt / extra chirp dbeta / extra TOD dgamma / t_cut,
instantaneous-frequency spread SPREAD, and TWIST_SCALE tying Omega to the ribbon twist.

Load-bearing choices (keep them):
* The arms differ in LUMINANCE (delay L* ~40, shaper L* ~65) so a greyscale print still
  tells them apart; column (d) is unreadable otherwise.
* The cut is masked in FREQUENCY, not time: a horizontal cut, and an edge in time blurred
  over the arm's own frequency spread.
* The ribbon lives only where both arms are above the contour level, so a larger dt
  shortens it; peak amplitudes are normalized column to column (the caption says so).
* The bold ribbon edge follows the arrow rule (an edge is bold where its tangential arrow
  points down the page), with the sense frozen once per column; it is what shows the
  direction of rotation and column (c)'s reversal.
* In column (c) the shaper arm's duration goes as 1/beta, so both arms span the same
  frequency range: a grating pair re-times light, it cannot add bandwidth.
* Past t_cut only the delay arm survives (a circular field, no torque); column (d) draws
  nothing there on purpose.
Every axis is in arbitrary units; lobe widths, delays and turn counts are exaggerated for
legibility and are not calibrations.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from base_core.plotting.enums import PlotColor
from manuscript_plotting_scripts.shaped_usCFG_paper import config

NAME = "cfg_arms_truncation"

# --- Model (illustrative, dimensionless units) -------------------------------
T_MIN, T_MAX = -0.72, 0.72   # plotted time window, set by the longest arm contour
NT = 4000                    # samples along time (raise if the ribbon twist aliases)

BETA0 = 1.00                 # base linear chirp rate of the input pulse
GAMMA0 = 0.10                # base TOD: enough that the ridges are visibly quadratic
ENV_W = 0.34                 # Gaussian 1/e half-width of the pulse envelope

# per column: delay dt, extra shaper chirp dbeta, extra shaper TOD dgamma, cut time t_cut
COLUMNS = [
    dict(label="(a)", dt=0.11, dbeta=0.0, dgamma=0.0, t_cut=None, show=()),
    # (b) has the most widely separated arms and so sets the frame for every column
    dict(label="(b)", dt=0.31, dbeta=0.0, dgamma=0.0, t_cut=None, show=("omega", "dt")),
    # (c) keeps (a)'s delay, so it differs from (a) ONLY in the grating separation; the
    # zero of Omega then sits off centre and the corkscrew turns, slows, stops and returns
    dict(label="(c)", dt=0.11, dbeta=0.45, dgamma=0.0, t_cut=None, show=("dbeta",)),
    # (d) cuts well inside the shaper lobe, mid-corkscrew
    dict(label="(d)", dt=0.11, dbeta=0.0, dgamma=0.0, t_cut=0.13, show=()),
]

SPREAD = 0.062               # instantaneous-frequency spread of each arm (exaggerated)
NW = 520                     # frequency samples for the density image
CONTOUR_LEVEL = 0.09         # iso-intensity level of the outline around each arm
CONTOUR_LW = 0.6
W_LIVE = 0.06                # intensity below which an arm no longer earns plotting room
W_PAD = 2.2                  # frequency headroom past that, in units of SPREAD
DENS_GAMMA = 0.50            # display gamma on the density (raises the faint wings)

TWIST_SCALE = 135.0          # radians of polarization turn per unit of Omega*t
RIBBON_AMP = 1.0             # peak ribbon half-width
NS = 26                      # samples across the ribbon width
ISO_ELEV = 34.0              # rotation of the view about the propagation axis (deg)
LIGHT_OFFSET = 90.0          # shadow band offset from the edge-on pinch angle (deg)
EDGE_LW = 1.15               # width of the traced edge helices where highlighted
EDGE_DIM = 0.22              # weight an edge keeps where its arrow does not point down

# --- Annotation placement ----------------------------------------------------
OMEGA_AT = -0.12             # time at which the 2*Omega split is measured (both arms bright)
DT_AT = 0.21                 # time on the shaper ridge from which the delay is read off;
                             # above the 2*Omega bracket so the two marks do not cross
DBETA_AT = (0.30, 0.50)      # times at which the steep and shallow chirps are labelled
DBETA_S_OFFSET = (-17.0, -11.0)  # stand-off of the steep arm's label, points
DBETA_D_XSHRINK = 0.35       # horizontal share of the shallow arm label's normal stand-off
LABEL_CLEAR = 13.0           # label stand-off along the ridge NORMAL, display points

# --- Colours -----------------------------------------------------------------
INK = PlotColor.BLACK        # labels, marks and the cut: the annotation layer
ARM_D = "#b3261e"            # delay arm (red, L* ~40): survives the cut
ARM_S = "#5b9ec9"            # shaper arm (blue, L* ~65): chirped and cut by the 4f line
RIBBON_LO, RIBBON_HI = "#7a3f10", "#e8b269"   # ribbon shading, dark to bright
EDGE_COLOR = "#5c3008"       # the two traced ribbon-edge helices
AXIS_GRAY = "#c0c0c0"        # the propagation axis the corkscrew winds around
SPINE_GRAY = "#999999"       # light frame: the units are illustrative

ROW_RATIO = (1.45, 1.0)      # row 1 is read quantitatively; row 2 is one large object
DPI = 1200                   # the ribbon and densities are rasterized


def arm_params(col):
    """(beta, gamma, t_offset, env_width) for the delay and shaper arms of one column.

    Shaper advanced by +dt/2, delay arm by -dt/2. The shaper also carries the extra
    chirp/TOD and, with it, a duration shorter by BETA0/beta: same spectrum, re-timed.
    """
    bs = BETA0 + col["dbeta"]
    delay = (BETA0, GAMMA0, -col["dt"] / 2.0, ENV_W)
    shaper = (bs, GAMMA0 + col["dgamma"], +col["dt"] / 2.0, ENV_W * BETA0 / bs)
    return delay, shaper


def omega_of_t(t, beta, gamma, t_off):
    """Instantaneous frequency of one arm, Eq. chirpphase, about omega_0 = 0."""
    s = t + t_off
    return 2.0 * beta * s + 3.0 * gamma * s**2


def envelope(t, t_off, w):
    """Gaussian field envelope of one arm, centred on its own delayed origin."""
    return np.exp(-((t + t_off) ** 2) / (2.0 * w**2))


def arm_density(T, W, beta, gamma, t_off, w_env):
    """Time-frequency density: intensity envelope along the ridge, SPREAD across it."""
    ridge = omega_of_t(T, beta, gamma, t_off)
    return envelope(T, t_off, w_env) ** 2 * np.exp(-((W - ridge) ** 2) / (2.0 * SPREAD**2))


def omega_rot(t, col):
    """Polarization rotation rate Omega(t) = [omega_s - omega_d]/2, Eq. Omega_def."""
    (bd, gd, od, _), (bs, gs, os_, _) = arm_params(col)
    return 0.5 * (omega_of_t(t, bs, gs, os_) - omega_of_t(t, bd, gd, od))


def twist_phase(t, col):
    """Accumulated polarization angle, the running integral of Omega(t)."""
    om = omega_rot(t, col)
    phase = np.concatenate(([0.0], np.cumsum(0.5 * (om[1:] + om[:-1]) * np.diff(t))))
    return TWIST_SCALE * (phase - phase[len(phase) // 2])


def contour_taper(intensity):
    """Window an envelope to reach zero exactly at the drawn contour level.

    Linear in the intensity, so the ribbon closes to a finite-angle tip rather than a
    blunt face (which only the truncated column may have).
    """
    w = np.clip((intensity - CONTOUR_LEVEL) / (1.0 - CONTOUR_LEVEL), 0.0, None)
    return w / max(w.max(), 1e-12)


def ribbon_envelope(t, col):
    """Corkscrew envelope: alive only while BOTH arms are above the contour level."""
    (_, _, od, wed), (_, _, os_, wes) = arm_params(col)
    both = np.minimum(envelope(t, od, wed) ** 2, envelope(t, os_, wes) ** 2)
    return RIBBON_AMP * contour_taper(both)


def snap_cut(t, col):
    """Move t_cut to the nearest instant at which the ribbon is broadside to the viewer,
    so the severed face projects as a full-height vertical edge. Both rows use the value.
    """
    if col["t_cut"] is None:
        return None
    e = np.radians(ISO_ELEV)
    k = np.abs(np.cos(twist_phase(t, col) + e))          # 1 = broadside, 0 = edge-on
    near = np.abs(t - col["t_cut"]) < 0.5 * np.pi / max(
        np.abs(omega_rot(t, col)).max() * TWIST_SCALE, 1e-9)
    if not near.any():
        near = np.ones_like(t, dtype=bool)
    return float(t[np.where(near, k, -1.0).argmax()])


def project(x, y, z):
    """Orthographic view rigidly rotated by ISO_ELEV about the propagation axis.

    Time maps exactly to screen x, which keeps the corkscrew registered with the panel
    above. Returns screen (x, y) and a depth key (larger = nearer the viewer).
    """
    e = np.radians(ISO_ELEV)
    return x, y * np.cos(e) - z * np.sin(e), y * np.sin(e) + z * np.cos(e)


def draw_ribbon(ax, t, col, cmap):
    """Draw the corkscrew as a depth-shaded helicoid twisting at Omega(t)."""
    s = np.linspace(-1.0, 1.0, NS)
    phi = twist_phase(t, col)
    env = ribbon_envelope(t, col)

    T2, S2 = np.meshgrid(t, s)
    P2 = np.broadcast_to(phi, T2.shape)
    E2 = np.broadcast_to(env, T2.shape)
    X, Y, Z = T2, S2 * E2 * np.cos(P2), S2 * E2 * np.sin(P2)

    P2c = P2
    if col["t_cut"] is not None:
        keep = t <= col["t_cut"]
        X, Y, Z, P2c = X[:, keep], Y[:, keep], Z[:, keep], P2[:, keep]

    xs, ys, depth = project(X, Y, Z)

    # Lambert shading from the helicoid normal n = (0, sin phi, -cos phi); the light is
    # placed 90 deg from the edge-on pinch so the shadow falls at each lobe's widest point.
    e = np.radians(ISO_ELEV)
    a = np.arctan2(np.cos(e), np.sin(e)) + np.radians(LIGHT_OFFSET)
    lit = np.abs(np.sin(P2c - a))
    rgb = cmap(np.clip(0.06 + 0.94 * lit, 0.0, 1.0))[..., :3]

    # painter's algorithm: quads back to front; rasterized (vector would be megabytes)
    quad_depth = 0.25 * (depth[:-1, :-1] + depth[:-1, 1:] + depth[1:, :-1] + depth[1:, 1:])
    ii, jj = np.unravel_index(np.argsort(quad_depth.ravel()), quad_depth.shape)
    verts = np.stack([
        np.column_stack([xs[ii, jj], ys[ii, jj]]),
        np.column_stack([xs[ii, jj + 1], ys[ii, jj + 1]]),
        np.column_stack([xs[ii + 1, jj + 1], ys[ii + 1, jj + 1]]),
        np.column_stack([xs[ii + 1, jj], ys[ii + 1, jj]]),
    ], axis=1)
    ax.add_collection(PolyCollection(verts, facecolors=rgb[ii, jj], edgecolors="none",
                                     linewidths=0, antialiased=True, zorder=3,
                                     rasterized=True))

    # Edge helices, bold by the arrow rule: with the stick along u = (0, cos phi, sin phi),
    # the +u edge is bold where sense*sin(phi + e) > 0 and the -u edge where it is < 0.
    # The sense is frozen once per column (envelope-weighted mean of Omega), so column
    # (c)'s reversal shows as the bold edge slowing, stopping and retracing its path.
    # Dim stretches still close the silhouette; outside the overlap window both edges lie
    # on the axis and are not drawn.
    env_edge = env if col["t_cut"] is None else env[t <= col["t_cut"]]
    alive = env_edge > 1e-3 * RIBBON_AMP
    sense = np.sign(np.sum(omega_rot(t, col) * env)) or 1.0
    turn = sense * np.sin(P2c[0] + e)
    for row, sgn in ((-1, +1.0), (0, -1.0)):     # row -1 is the +u edge, row 0 is -u
        ex, ey, ed = xs[row], ys[row], depth[row]
        segs = np.stack([np.column_stack([ex[:-1], ey[:-1]]),
                         np.column_stack([ex[1:], ey[1:]])], axis=1)
        mid = 0.5 * (ed[:-1] + ed[1:])
        down = sgn * 0.5 * (turn[:-1] + turn[1:])  # graded, so highlights fade, not cap
        emph = EDGE_DIM + (1.0 - EDGE_DIM) * np.clip(down, 0.0, 1.0) ** 0.6
        rgba = np.tile(np.array(to_rgba(EDGE_COLOR)), (segs.shape[0], 1))
        rgba[:, 3] = emph
        live = alive[:-1] & alive[1:]
        # split at depth 0 so the surface occludes the far half
        for sel, z in (((mid < 0.0) & live, 2), ((mid >= 0.0) & live, 4)):
            if sel.any():
                ax.add_collection(LineCollection(
                    segs[sel], colors=rgba[sel], zorder=z, capstyle="round",
                    rasterized=True, linewidths=emph[sel] * EDGE_LW))

    # the cut face: truncation is one instant, so the ribbon stops on a straight edge
    if col["t_cut"] is not None:
        ax.plot([xs[0, -1], xs[-1, -1]], [ys[0, -1], ys[-1, -1]], marker="",
                color=INK, lw=1.0, zorder=5, solid_capstyle="butt")

    # the propagation axis the corkscrew winds around
    ax.plot([xs.min(), xs.max()], [0, 0], marker="", color=AXIS_GRAY, lw=0.5, zorder=0)


def label_along_ridge(ax, t, w, at, txt, offset=None, ha="center", va="center",
                      x_shrink=1.0):
    """Name a ridge, standing off along its display-space NORMAL (sign of `at` picks the
    side); `offset` in points overrides the normal.
    """
    k = int(np.argmin(np.abs(t - abs(at))))
    if offset is None:
        k2 = min(k + 40, len(t) - 1)             # a finite step: k+1 is display noise
        d = ax.transData.transform((t[k2], w[k2])) - ax.transData.transform((t[k], w[k]))
        d = d / max(np.hypot(*d), 1e-9)
        n = np.array([-d[1], d[0]]) * np.sign(at) * np.array([x_shrink, 1.0])
        offset = tuple(n * LABEL_CLEAR)
    ax.annotate(txt, xy=(t[k], w[k]), xytext=tuple(offset), textcoords="offset points",
                ha=ha, va=va, color=INK, zorder=7)


def annotate_quantities(ax, t, col, wd, ws):
    """Mark the quantity each column varies: 2*Omega and dt in (b), the chirps in (c)."""
    arrow = dict(arrowstyle="<->,head_width=0.10,head_length=0.28", color=INK,
                 lw=0.6, shrinkA=0, shrinkB=0)

    if "omega" in col["show"]:
        # the vertical split between the arms; the polarization turns at half of it
        i = int(np.argmin(np.abs(t - OMEGA_AT)))
        ax.annotate("", xy=(t[i], wd[i]), xytext=(t[i], ws[i]), arrowprops=arrow, zorder=7)
        ax.annotate(r"$2\Omega$", xy=(t[i], 0.5 * (wd[i] + ws[i])),
                    xytext=(-1.5, -7), textcoords="offset points",
                    ha="right", va="center", color=INK, zorder=7)

    if "dt" in col["show"]:
        # the arm delay, read along one frequency: the two arms reach it dt apart
        w_ref = ws[int(np.argmin(np.abs(t - DT_AT)))]
        td = t[int(np.argmin(np.abs(wd - w_ref)))]
        ts = t[int(np.argmin(np.abs(ws - w_ref)))]
        ax.annotate("", xy=(td, w_ref), xytext=(ts, w_ref), arrowprops=arrow, zorder=7)
        ax.annotate(r"$\Delta t$", xy=(0.5 * (td + ts), w_ref),
                    xytext=(2, 0), textcoords="offset points",
                    ha="center", va="bottom", color=INK, zorder=7)

    if "dbeta" in col["show"]:
        # name the two slopes, no leaders; the gap between the names is Delta beta
        label_along_ridge(ax, t, ws, +DBETA_AT[0], r"$\beta+\Delta\beta$",
                          offset=DBETA_S_OFFSET, ha="right", va="center")
        label_along_ridge(ax, t, wd, -DBETA_AT[1], r"$\beta$", x_shrink=DBETA_D_XSHRINK)


def style_axes(ax):
    """No ticks (units are illustrative), light frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(SPINE_GRAY)
        sp.set_linewidth(0.5)


def density_layer(ax, dens, w_lim, color):
    """One arm's density as a coloured alpha wash, so crossing arms both show."""
    rgba = np.zeros(dens.shape + (4,))
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = np.clip(dens, 0, 1) ** DENS_GAMMA
    ax.imshow(rgba, origin="lower", aspect="auto", extent=[T_MIN, T_MAX, -w_lim, w_lim],
              interpolation="bilinear", zorder=1, rasterized=True)


def main() -> None:
    ribbon_cmap = LinearSegmentedColormap.from_list("ribbon", [RIBBON_LO, RIBBON_HI])
    t = np.linspace(T_MIN, T_MAX, NT)
    columns = [dict(c) for c in COLUMNS]
    for col in columns:                          # both rows use the same snapped cut
        col["t_cut"] = snap_cut(t, col)

    # one frequency range for all columns, sized to where the arms are actually bright
    w_max = 0.0
    for col in columns:
        for beta, gamma, off, w_env in arm_params(col):
            keep = envelope(t, off, w_env) ** 2 > W_LIVE
            if keep.any():
                w_max = max(w_max, np.abs(omega_of_t(t[keep], beta, gamma, off)).max())
    w_lim = w_max + W_PAD * SPREAD

    # Built at the final size, since the ridge-label stand-offs are computed in display
    # space. hspace = 0: each column reads as one panel about one pulse.
    fig = plt.figure(figsize=config.FIGURE_SIZES_IN[NAME], dpi=DPI)
    gs = GridSpec(2, len(columns), figure=fig, height_ratios=list(ROW_RATIO),
                  hspace=0.0, wspace=0.07, left=0.045, right=0.992, top=0.915, bottom=0.105)

    for k, col in enumerate(columns):
        (bd, gd, od, wed), (bs, gs_, os_, wes) = arm_params(col)

        # ---- top: time-frequency density of both arms ----
        axt = fig.add_subplot(gs[0, k])
        w = np.linspace(-w_lim, w_lim, NW)
        T, W = np.meshgrid(t, w)

        w_cut = None
        dens_d = arm_density(T, W, bd, gd, od, wed)
        dens_s_full = arm_density(T, W, bs, gs_, os_, wes)
        dens_s = dens_s_full
        if col["t_cut"] is not None:
            # the prism removes the shaper spectrum beyond omega_cut: mask on FREQUENCY
            w_cut = omega_of_t(np.array([col["t_cut"]]), bs, gs_, os_)[0]
            dens_s = dens_s_full * 0.5 * (1.0 - np.tanh((W - w_cut) / (0.55 * SPREAD)))
        density_layer(axt, dens_s, w_lim, ARM_S)
        density_layer(axt, dens_d, w_lim, ARM_D)

        # Iso-intensity outline of each arm. The truncated arm's outline is contoured on
        # the UNMASKED density, clipped at omega_cut and closed with one straight rule,
        # so the severed side is flat, as the cut is.
        for dens, c in ((dens_s_full, ARM_S), (dens_d, ARM_D)):
            cs = axt.contour(T, W, dens, levels=[CONTOUR_LEVEL], colors=[c],
                             linewidths=CONTOUR_LW, zorder=4)
            if w_cut is not None and c == ARM_S:
                cs.set_clip_path(Rectangle((T_MIN, -w_lim), T_MAX - T_MIN, w_cut + w_lim,
                                           transform=axt.transData))
                live = dens_s_full[int(np.argmin(np.abs(w - w_cut)))] >= CONTOUR_LEVEL
                if live.any():
                    ii = np.where(live)[0]
                    axt.plot([t[ii[0]], t[ii[-1]]], [w_cut, w_cut], marker="", color=c,
                             lw=CONTOUR_LW, zorder=4, solid_capstyle="butt")

        if w_cut is not None:
            axt.axhline(w_cut, marker="", color=INK, lw=0.7, ls=(0, (5, 2.5)), zorder=5)
            axt.plot([col["t_cut"]], [w_cut], ls="", marker="o", ms=2.6, color=INK,
                     zorder=6, mec="none")
            axt.annotate(r"$\omega_{\rm cut}$", xy=(T_MIN, w_cut), xytext=(3, 2.5),
                         textcoords="offset points", color=INK, ha="left", va="bottom")

        axt.set_xlim(T_MIN, T_MAX)
        axt.set_ylim(-w_lim, w_lim)
        style_axes(axt)
        axt.set_title(col["label"], color=INK, pad=3)
        if k == 0:
            axt.set_ylabel("Optical frequency", color=INK, labelpad=2)
        annotate_quantities(axt, t, col, omega_of_t(t, bd, gd, od),
                            omega_of_t(t, bs, gs_, os_))

        # ---- bottom: the corkscrew, same time axis ----
        axb = fig.add_subplot(gs[1, k])
        draw_ribbon(axb, t, col, ribbon_cmap)
        if col["t_cut"] is not None:
            # right of the rule: the last lobe occupies the left side
            axb.annotate(r"$t_{\rm cut}$", xy=(col["t_cut"], RIBBON_AMP), xytext=(3, 0),
                         textcoords="offset points", color=INK, ha="left", va="top")
        axb.set_xlim(T_MIN, T_MAX)
        axb.set_ylim(-1.05 * RIBBON_AMP, 1.05 * RIBBON_AMP)
        style_axes(axb)
        axb.set_xlabel("Time", color=INK, labelpad=1)
        if k == 0:
            axb.set_ylabel("Corkscrew field", color=INK, labelpad=2)

        # the t_cut rule spans both rows: release is one instant on one time axis
        if col["t_cut"] is not None:
            for ax_ in (axt, axb):
                ax_.axvline(col["t_cut"], marker="", color=INK, lw=0.7, ls=(0, (2, 2)),
                            zorder=6, clip_on=False)

    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
