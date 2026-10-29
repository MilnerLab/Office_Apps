"""The two centrifuge arms and the corkscrew they make.

Theory schematic, no data. Two columns, each a pair of stacked panels on one time axis:

  top     the time-frequency density of the two interferometer arms, each a chirped
          Gaussian about its instantaneous frequency
              omega(t) = 2*beta*(t -+ dt/2) + 3*gamma*(t -+ dt/2)**2      (Eq. chirpphase)
          with a thin iso-intensity contour around it
  bottom  the corkscrew field, turning at
              Omega(t) = [omega_s(t) - omega_d(t)] / 2                     (Eq. Omega_def)
          i.e. half the vertical gap between the two ridges drawn above; the twist phase is
          its running integral, so the rows are registered by construction

  (a1) an ultraslow centrifuge that starts near zero frequency and accelerates
  (b1) the same centrifuge truncated at t_cut (a frequency cut in the Fourier plane)
  (a2), (b2) the corkscrews below (a1) and (b1); panel letters sit in each top-left corner
The four-column version (delay, chirp and truncation side by side) is in git history; the
review asked for the truncation alone, on one centrifuge.

Model parameters (illustrative, dimensionless): base chirp BETA0, base TOD GAMMA0,
envelope width ENV_W, per-column delay dt / extra chirp dbeta / extra TOD dgamma / t_cut,
instantaneous-frequency spread SPREAD, and TWIST_SCALE tying Omega to the ribbon twist.

Load-bearing choices (keep them):
* Each arm's density is coloured by optical frequency (a rainbow, red at the bottom), so
  the chirp reads directly. Both arms carry the same plain black outline; the caption
  tells them apart by position (shaper arm left, delay arm right).
* The cut is masked in FREQUENCY, not time: a horizontal cut, and an edge in time blurred
  over the arm's own frequency spread.
* The ribbon and the grey band share FIELD units: the ribbon is the linear part
  2*min(|E_s|, |E_d|), the grey band the delay arm's circular remainder |E_d| - |E_s|, both
  over one peak. So the field drops to about half at t_cut, as it must when one of two
  equal counter-rotating arms vanishes. (An earlier version drew the ribbon as a windowed
  intensity and the grey band as a field, which hid that drop.)
* The bold ribbon edge follows the arrow rule (an edge is bold where its tangential arrow
  points down the page), with the sense frozen once per column; it is what shows the
  direction of rotation and column (c)'s reversal.
* In column (c) the shaper arm's duration goes as 1/beta, so both arms span the same
  frequency range: a grating pair re-times light, it cannot add bandwidth.
* Past t_cut only the delay arm survives (a circular field, no torque). Column (b) draws
  it as a flat grey shaded envelope, distinct from the ribbon; circular remainders
  elsewhere, including column (a)'s tail, are left out as clutter. It is deliberately NOT a helix: a drawn
  winding would read as a corkscrew turning rate, but a circular field turns at the
  optical frequency, orders of magnitude faster than the centrifuge.
Every axis is in arbitrary units; lobe widths, delays and turn counts are exaggerated for
legibility and are not calibrations.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
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
# One ultraslow centrifuge, drawn whole and truncated. dt and dbeta put the zero of Omega
# near the leading edge of the overlap window, so the corkscrew starts slow and accelerates.
US_DT, US_DBETA = 0.13, 0.30
COLUMNS = [
    dict(labels=("(a1)", "(a2)"), dt=US_DT, dbeta=US_DBETA, dgamma=0.0, t_cut=None,
         show=("dt", "dbeta")),
    # (b) cuts well inside the shaper lobe, mid-corkscrew
    dict(labels=("(b1)", "(b2)"), dt=US_DT, dbeta=US_DBETA, dgamma=0.0, t_cut=0.12,
         show=()),
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
CIRC_COLOR = "#8a8a8a"       # the circular field: grey, so it reads apart from the ribbon
CIRC_ALPHA = 0.35            # fill opacity of its envelope
CIRC_LW = 0.45               # its envelope outline
CIRC_END = 0.02              # envelope stops where it drops below this x ribbon peak

# --- Annotation placement ----------------------------------------------------
OMEGA_AT = -0.12             # time at which the 2*Omega split is measured (both arms bright)
DT_AT = 0.21                 # time on the shaper ridge from which the delay is read off;
                             # above the 2*Omega bracket so the two marks do not cross
DBETA_AT = (0.42, 0.50)      # times at which the steep and shallow chirps are labelled
DT_LABEL_OFFSET = (4.0, -4.0)  # Delta t label from the arrow's right end, points
DBETA_S_OFFSET = (-17.0, -11.0)  # stand-off of the steep arm's label, points
DBETA_D_XSHRINK = 0.35       # horizontal share of the shallow arm label's normal stand-off
LABEL_CLEAR = 13.0           # label stand-off along the ridge NORMAL, display points
PANEL_LABEL_AT = (3.0, -3.0)  # panel letter from the axes' top-left corner, points

# --- Colours -----------------------------------------------------------------
INK = PlotColor.BLACK        # labels, marks and the cut: the annotation layer
ARM_OUTLINE = INK            # both arm outlines: plain black
SPECTRUM = ["#d7191c", "#f07c12", "#e3bb00", "#2ca02c", "#1f5fd6", "#6a2fb0"]  # low to high frequency
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


def field_envelopes(t, col):
    """Linear (ribbon) and circular (grey) envelopes of the two counter-rotating arms.

    Linear part 2*min(|E_s|, |E_d|), circular remainder |E_d| - |E_s|, both in field units
    and normalized by the same peak, so the drop at t_cut is drawn to scale. Past t_cut the
    shaper arm is zero. The circular part is kept only past t_cut, where the delay arm is
    left on its own. The remainders elsewhere (including the untruncated column's tail)
    are physical but would only clutter a schematic.
    """
    (_, _, od, wed), (_, _, os_, wes) = arm_params(col)
    e_d, e_s = envelope(t, od, wed), envelope(t, os_, wes)
    if col["t_cut"] is not None:
        e_s = np.where(t <= col["t_cut"], e_s, 0.0)
        alone = t > col["t_cut"]
    else:
        alone = np.zeros_like(t, dtype=bool)
    lin = 2.0 * np.minimum(e_s, e_d)
    peak = lin.max()
    return RIBBON_AMP * lin / peak, np.where(alone, RIBBON_AMP * (e_d - e_s) / peak, 0.0)


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


def draw_ribbon(ax, t, col, cmap, env):
    """Draw the corkscrew as a depth-shaded helicoid twisting at Omega(t)."""
    s = np.linspace(-1.0, 1.0, NS)
    phi = twist_phase(t, col)

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

    # the propagation axis the corkscrew winds around, across the whole window
    ax.plot([t.min(), t.max()], [0, 0], marker="", color=AXIS_GRAY, lw=0.5, zorder=0)


def draw_circular(ax, t, circ):
    """The delay arm alone: a circularly polarized field, drawn as its envelope.

    A shaded band between +-amplitude (a circle seen from any side spans +-a), with no
    winding, since the field turns at the optical frequency. `circ` comes from
    field_envelopes, in the ribbon's units; the band ends below CIRC_END of the ribbon peak.
    """
    sel = circ > CIRC_END * RIBBON_AMP
    if not sel.any():
        return
    a = np.where(sel, circ, np.nan)
    ax.fill_between(t, -a, a, facecolor=to_rgba(CIRC_COLOR, CIRC_ALPHA), edgecolor="none",
                    zorder=2, rasterized=True)
    for sgn in (+1.0, -1.0):
        ax.plot(t, sgn * a, marker="", color=CIRC_COLOR, lw=CIRC_LW, zorder=4)


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
        # the arm delay, read along one frequency: the two arms reach it dt apart. At the
        # common centre frequency the gap is exactly dt even when the chirps differ.
        w_ref = 0.0 if col["dbeta"] else ws[int(np.argmin(np.abs(t - DT_AT)))]
        td = t[int(np.argmin(np.abs(wd - w_ref)))]
        ts = t[int(np.argmin(np.abs(ws - w_ref)))]
        ax.annotate("", xy=(td, w_ref), xytext=(ts, w_ref), arrowprops=arrow, zorder=7)
        # below and right of the arrow's delay-arm end, clear of both arms
        ax.annotate(r"$\Delta t$", xy=(td, w_ref), xytext=DT_LABEL_OFFSET,
                    textcoords="offset points", ha="left", va="top", color=INK, zorder=7)

    if "dbeta" in col["show"]:
        # name the two slopes, no leaders; the gap between the names is Delta beta
        label_along_ridge(ax, t, ws, +DBETA_AT[0], r"$\beta_0+\Delta\beta$",
                          offset=DBETA_S_OFFSET, ha="right", va="center")
        label_along_ridge(ax, t, wd, -DBETA_AT[1], r"$\beta_0$", x_shrink=DBETA_D_XSHRINK)


def panel_label(ax, txt):
    """Panel letter inside the top-left corner, which both rows leave empty."""
    ax.annotate(txt, xy=(0, 1), xycoords="axes fraction", xytext=PANEL_LABEL_AT,
                textcoords="offset points", ha="left", va="top", color=INK, zorder=8)


def style_axes(ax):
    """No ticks (units are illustrative), light frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(SPINE_GRAY)
        sp.set_linewidth(0.5)


def density_layer(ax, dens, w_lim, cmap, w_span):
    """One arm's density as an alpha wash coloured by optical frequency (rows of `dens`).

    The colour scale runs over w_span = (lo, hi), the frequencies inside the drawn outlines,
    so the full spectrum shows within them rather than in the faint wings.
    """
    rgba = np.zeros(dens.shape + (4,))
    w = np.linspace(-w_lim, w_lim, dens.shape[0])
    rgba[..., :3] = cmap(np.clip((w - w_span[0]) / (w_span[1] - w_span[0]), 0.0, 1.0))[:, None, :3]
    rgba[..., 3] = np.clip(dens, 0, 1) ** DENS_GAMMA
    ax.imshow(rgba, origin="lower", aspect="auto", extent=[T_MIN, T_MAX, -w_lim, w_lim],
              interpolation="bilinear", zorder=1, rasterized=True)


def main() -> None:
    ribbon_cmap = LinearSegmentedColormap.from_list("ribbon", [RIBBON_LO, RIBBON_HI])
    spectrum_cmap = LinearSegmentedColormap.from_list("spectrum", SPECTRUM)
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
                  hspace=0.0, wspace=0.07, left=0.06, right=0.992, top=0.985, bottom=0.09)

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
        inside = w[np.maximum(dens_d, dens_s_full).max(axis=1) >= CONTOUR_LEVEL]
        w_span = (inside.min(), inside.max())
        density_layer(axt, dens_s, w_lim, spectrum_cmap, w_span)
        density_layer(axt, dens_d, w_lim, spectrum_cmap, w_span)

        # Iso-intensity outline of each arm. The truncated arm's outline is contoured on
        # the UNMASKED density, clipped at omega_cut and closed with one straight rule,
        # so the severed side is flat, as the cut is.
        for dens, is_shaper in ((dens_s_full, True), (dens_d, False)):
            c = ARM_OUTLINE
            cs = axt.contour(T, W, dens, levels=[CONTOUR_LEVEL], colors=[c],
                             linewidths=CONTOUR_LW, zorder=4)
            if w_cut is not None and is_shaper:
                cs.set_clip_path(Rectangle((T_MIN, -w_lim), T_MAX - T_MIN, w_cut + w_lim,
                                           transform=axt.transData))
                live = dens_s_full[int(np.argmin(np.abs(w - w_cut)))] >= CONTOUR_LEVEL
                if live.any():
                    ii = np.where(live)[0]
                    axt.plot([t[ii[0]], t[ii[-1]]], [w_cut, w_cut], marker="", color=c,
                             lw=CONTOUR_LW, zorder=4, solid_capstyle="butt")

        if w_cut is not None:
            # the cut shows as the shaper arm's flat edge; the dot pins it to t_cut
            axt.plot([col["t_cut"]], [w_cut], ls="", marker="o", ms=2.6, color=INK,
                     zorder=6, mec="none")

        axt.set_xlim(T_MIN, T_MAX)
        axt.set_ylim(-w_lim, w_lim)
        style_axes(axt)
        panel_label(axt, col["labels"][0])
        if k == 0:
            axt.set_ylabel("Optical Frequency", color=INK, labelpad=5)
        annotate_quantities(axt, t, col, omega_of_t(t, bd, gd, od),
                            omega_of_t(t, bs, gs_, os_))

        # ---- bottom: the corkscrew, same time axis ----
        axb = fig.add_subplot(gs[1, k])
        lin, circ = field_envelopes(t, col)
        draw_ribbon(axb, t, col, ribbon_cmap, lin)
        draw_circular(axb, t, circ)
        if col["t_cut"] is not None:
            # right of the rule: the last lobe occupies the left side
            axb.annotate(r"$t_{\rm cut}$", xy=(col["t_cut"], RIBBON_AMP), xytext=(3, 0),
                         textcoords="offset points", color=INK, ha="left", va="top")
        axb.set_xlim(T_MIN, T_MAX)
        axb.set_ylim(-1.05 * RIBBON_AMP, 1.05 * RIBBON_AMP)
        style_axes(axb)
        panel_label(axb, col["labels"][1])
        axb.set_xlabel("Time", color=INK, labelpad=4)
        if k == 0:
            axb.set_ylabel("Centrifuge Field", color=INK, labelpad=5)

        # the t_cut rule spans both rows: release is one instant on one time axis
        if col["t_cut"] is not None:
            for ax_ in (axt, axb):
                ax_.axvline(col["t_cut"], marker="", color=INK, lw=0.7, ls=(0, (2, 2)),
                            zorder=6, clip_on=False)

    config.save_figure(fig, NAME)


if __name__ == "__main__":
    main()
