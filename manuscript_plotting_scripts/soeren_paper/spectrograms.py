#!/usr/bin/env python3
"""
Zeit-Frequenz-Spektrogramme (links) und zeitintegrierte Spektren (rechts)
der konventionellen Zentrifuge (oben) und der Ultraslow-Zentrifuge (unten).

Layout (2x2, Frequenzachse je Zeile geteilt):
    (a) CFG-Spektrogramm      (b) CFG-Spektrum: Gauss, keine Fringes
                                  (die Arme ueberlappen spektral nicht)
    (c) usCFG-Spektrogramm    (d) usCFG-Spektrum: Gauss mit gechirpten Fringes
                                  (beide Arme tragen das volle Spektrum)

Die cfCFG (panel_b) ist im aktuellen Layout nicht enthalten.

Konvention (wie in den Papern):
    phi(t) = phi_0 + w_0*t + beta*t^2   ->   w(t) = w_0 + 2*beta*t
    Die Steigung einer Arm-Linie im Zeit-Frequenz-Diagramm ist also 2*beta.

(a) Konventionelle Zentrifuge (CFG):
    Spektrum wird in der Fourier-Ebene halbiert; blauer Arm beta_+ > 0,
    roter Arm beta_- = -beta_+.  Beide Arme starten bei (t=0, w_0) mit
    maximaler Intensitaet (Schnitt durch das Gauss-Maximum) und klingen
    nur noch ab.  Frequenzdifferenz: dw(t) = 4*beta*t  ->  Omega(t) = 2*beta*t.

(b) Konstant-Frequenz-Zentrifuge (cfCFG):
    Identischer Chirp beta_0 in beiden Armen, Verzoegerung dt
    ->  Omega_cf = beta_0 * dt = const.

(c) Ultraslow-Zentrifuge (usCFG):
    beta_L = beta_R + d_beta  ->  Omega_us(t) ~ beta_0*dt + d_beta*t,
    Winkelbeschleunigung eps = d_beta / (2*pi).

Ausgabe: centrifuge_spectrograms.png / .pdf
"""

import numpy as np
from pathlib import Path as FilePath

import matplotlib
matplotlib.use("Agg")  # file output only; Qt backend aborts on this GNOME setup
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, ScaledTranslation

# ----------------------------------------------------------------------
# Stil
# ----------------------------------------------------------------------
DARK = False                     # True -> dunkler Hintergrund
LANG_X, LANG_Y = "Time", "Optical Frequency"

# Same style as single_spectrum.py / stabilized_vs_unstabilized.py: one journal
# column (3 3/8 in = 8.6 cm), fonts at final print size.
REPO_ROOT = FilePath(__file__).resolve().parents[2]
SAVE = FilePath(__file__).with_name("centrifuge_spectrograms")   # + .png / .pdf
FIGSIZE = (3 + 3 / 8, 3.25)
WIDTH_RATIOS = (1.7, 1.0)        # Spektrogramm : Spektrum
plt.style.use(str(REPO_ROOT / "stylefiles" / "compare_c2t_spectrogram.mplstyle"))
# savefig.bbox "standard": the PDF is exactly FIGSIZE, so it is placed in the column
# without rescaling and the 10 pt fonts stay 10 pt (the style's "tight" crops it narrower,
# and stretching it back to \columnwidth enlarges the text).
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "pdf.fonttype": 42,
                     "savefig.bbox": "standard"})

if DARK:
    BG, FG, FRAME, ACCENT = "#2b3040", "#e8e8ee", "#b8bcc8", "#7fb3d5"
    TAILC, MARKEDGE = "#b8bcc8", "white"
else:
    # schwarz wie die anderen Paper-Abbildungen
    BG, FG, FRAME, ACCENT = "white", "black", "black", "black"
    TAILC, MARKEDGE = "#888888", "#222222"

# Sichtbares Spektrum: rot (niedrige Frequenz) -> violett (hohe Frequenz)
SPECTRAL = LinearSegmentedColormap.from_list("vis", [
    "#8d1b0b", "#c43c07", "#e07b0a", "#e8b400", "#c9cc1a",
    "#4caf28", "#17a58a", "#1f9fc4", "#2f77cf", "#4a54c8",
    "#7b46c9", "#a878dd", "#c9a6e8",
])

# ----------------------------------------------------------------------
# Geometrie-Parameter (alles in willkuerlichen Einheiten der Achsen)
# ----------------------------------------------------------------------
XLIM, YLIM = (0.0, 10.0), (0.0, 10.0)

SLOPE = 1.35        # Steigung dw/dt = 2*beta_0
F_SPAN = 3.6        # halbe dargestellte spektrale Breite eines Arms
WIDTH = 0.30        # halbe Banddicke (senkrecht zur Chirp-Linie)
TAPER = 2.1          # F_SPAN / sigma der Gauss-Einhuellenden
TAIL = 0.60         # Laenge der gestrichelten Verlaengerung (Zeiteinheiten)

DT = 3.20           # Delta t zwischen den Armen (Panels b, c)
DBETA = 0.50        # Steigungsdifferenz 2*d_beta (Panel c)

CFG_T0, CFG_F0 = 2.6, 5.0   # Scheitel der CFG-Arme (Panel a)
US_TX, US_FX = 1.4, 0.7     # gemeinsamer Startpunkt der usCFG-Arme (Panel c)
N_FRINGES = 7               # Spektrale Fringes ueber die usCFG-Bandbreite (Panel d)
SPEC_XLIM = (0.0, 1.15)     # Intensitaetsachse der Spektren


# ----------------------------------------------------------------------
# Zeichenroutinen
# ----------------------------------------------------------------------
def chirped_band(ax, t_c, f_c, slope, f_span=F_SPAN, width=WIDTH,
                 part="full", cnorm=None, taper=TAPER, tail=TAIL,
                 n_seg=400, cmap=SPECTRAL, zorder=3):
    """Zeichnet einen gechirpten Puls als spektral eingefaerbtes Band.

    t_c, f_c : Zentrum des Pulses (bei part='upper'/'lower' der Scheitel)
    slope    : dw/dt = 2*beta
    part     : 'full'  -> volle Gauss-Einhuellende
               'upper' -> nur blaue Haelfte (Maximum am Scheitel)
               'lower' -> nur rote Haelfte  (Maximum am Scheitel)
    cnorm    : (f_min, f_max) fuer die Farbskala
    """
    if part == "full":
        u = np.linspace(-f_span, f_span, n_seg + 1)
    elif part == "upper":
        u = np.linspace(0.0, f_span, n_seg + 1)
    elif part == "lower":
        u = np.linspace(-f_span, 0.0, n_seg + 1)
    else:
        raise ValueError(part)

    sigma = f_span / taper
    env = np.exp(-(u / sigma) ** 2)          # Amplituden-Einhuellende
    f = f_c + u
    t = t_c + u / slope

    # Einheitsnormale zur Chirp-Linie
    L = np.hypot(1.0 / slope, 1.0)
    nx, ny = -1.0 / L, (1.0 / slope) / L
    h = width * env

    xu, yu = t + nx * h, f + ny * h
    xl, yl = t - nx * h, f - ny * h

    verts = np.stack([
        np.column_stack([xu[:-1], yu[:-1]]),
        np.column_stack([xu[1:], yu[1:]]),
        np.column_stack([xl[1:], yl[1:]]),
        np.column_stack([xl[:-1], yl[:-1]]),
    ], axis=1)

    f_mid = 0.5 * (f[:-1] + f[1:])
    v0, v1 = cnorm if cnorm else (f.min(), f.max())
    colors = cmap(np.clip((f_mid - v0) / (v1 - v0), 0, 1))

    ax.add_collection(PolyCollection(verts, facecolors=colors,
                                     edgecolors=colors, linewidths=0.4,
                                     zorder=zorder))

    # gestrichelte Verlaengerung in Richtung wachsender Zeit
    if tail:
        i = int(np.argmax(t))
        p0 = np.array([t[i], f[i]])
        p1 = p0 + np.array([1.0, slope]) * tail   # entlang der Chirp-Linie
        ax.plot(*zip(p0, p1), ls=(0, (3, 2)), lw=0.7, color=TAILC,
                alpha=0.75, zorder=2)
        return p1
    return np.array([t[int(np.argmax(t))], f[int(np.argmax(t))]])


def diamond(ax, t, f, filled=False, size=16):
    ax.scatter([t], [f], marker="D", s=size, zorder=6,
               facecolors=("#e8402a" if filled else BG),
               edgecolors=MARKEDGE, linewidths=0.7)


def circ_arrow(ax, x, y, ccw=True, r=0.07, gap=70):
    """Kreispfeil um (x, y) als Marker fuer die zirkulare Polarisation.

    r ist in Zoll, damit der Kreis unabhaengig vom Achsen-Seitenverhaeltnis
    rund bleibt.  gap: Oeffnungswinkel (Grad) der Luecke, oben zentriert.
    """
    arc = Path.arc(90 + gap / 2, 90 + 360 - gap / 2)          # gegen den Uhrzeigersinn
    if not ccw:
        arc = Path(arc.vertices[::-1], arc.codes)
    trans = (Affine2D().scale(r) + ax.figure.dpi_scale_trans
             + ScaledTranslation(x, y, ax.transData))
    ax.add_patch(FancyArrowPatch(path=arc, transform=trans, arrowstyle="-|>",
                                 mutation_scale=6, color=FG, lw=0.8, zorder=7))


def double_arrow(ax, p0, p1, label, offset=(0.0, 0.0), **kw):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="<->", color=FG, lw=0.8,
                                mutation_scale=7, shrinkA=0, shrinkB=0), zorder=7)
    mid = (0.5 * (p0[0] + p1[0]) + offset[0], 0.5 * (p0[1] + p1[1]) + offset[1])
    ax.text(*mid, label, color=FG, ha="center", va="center", zorder=8,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1.0), **kw)


def style_axes(ax, label, xlabel=LANG_X, xlim=XLIM, label_right=False):
    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():                # Breite aus dem Stylefile
        s.set_color(FRAME)
    ax.set_facecolor(BG)
    if xlabel:
        ax.set_xlabel(xlabel, color=ACCENT, labelpad=3)
    # Panel-Label wie in stabilized_vs_unstabilized.py.
    # \textbf, not fontweight="bold": the style sets text.usetex, which ignores fontweight
    # Abstand zum Rahmen in Punkten, damit er in schmalen und breiten Panels gleich ist
    ax.annotate(rf"\textbf{{{label}}}", xy=(1.0 if label_right else 0.0, 1.0),
                xycoords="axes fraction", textcoords="offset points",
                xytext=(-7 if label_right else 7, -5), color=FG,
                va="top", ha="right" if label_right else "left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="0.85", edgecolor="k", lw=0.6))


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------
def panel_a(ax):
    """Konventionelle Zentrifuge: beta_- = -beta_+, Spektrum halbiert."""
    t0, f0 = CFG_T0, CFG_F0
    cn = (f0 - F_SPAN, f0 + F_SPAN)          # gemeinsame Farbskala (ein Spektrum)

    e_up = chirped_band(ax, t0, f0, +SLOPE, part="upper", cnorm=cn)  # blauer Arm
    e_dn = chirped_band(ax, t0, f0, -SLOPE, part="lower", cnorm=cn)  # roter Arm

    # entgegengesetzte zirkulare Polarisation der beiden Arme
    # neben die Arm-Enden, weit genug vom Rahmen, damit nichts abgeschnitten wird
    circ_arrow(ax, e_up[0] + 0.7, e_up[1] - 1.0, ccw=True)
    circ_arrow(ax, e_dn[0] + 0.7, e_dn[1] + 1.0, ccw=False)
    diamond(ax, t0, f0, filled=False)

    # 2*Omega: vertikaler Abstand der beiden Arme, hier an einer Stelle
    # abseits vom gemeinsamen Ursprung ausgewertet, da der Abstand mit t waechst.
    t_a = t0 + 0.65 * (F_SPAN / SLOPE)
    f_hi = f0 + SLOPE * (t_a - t0)
    f_lo = f0 - SLOPE * (t_a - t0)
    double_arrow(ax, (t_a, f_lo + 0.35), (t_a, f_hi - 0.35), r"$2f_{\mathrm{CFG}}$",
                 offset=(0.95, 0.0))


def panel_b(ax):
    """cfCFG: gleicher Chirp, Verzoegerung dt -> Omega = beta_0*dt."""
    # Shift the pulse pair to the right so the full two-pulse structure,
    # including the dashed tails, is centered in the panel and not clipped.
    t1, f1 = 3.10, 4.6
    cn1 = (f1 - F_SPAN, f1 + F_SPAN)
    e1 = chirped_band(ax, t1, f1, SLOPE, cnorm=cn1)            # frueher Arm
    e2 = chirped_band(ax, t1 + DT, f1, SLOPE, cnorm=cn1)       # spaeterer Arm

    # Marker der Zentralfrequenzen (definieren dt)
    diamond(ax, t1, f1, filled=False)
    diamond(ax, t1 + DT, f1, filled=True)

    # Delta t: horizontaler Abstand der beiden Pulszentren (Rauten)
    double_arrow(ax, (t1 + 0.35, f1), (t1 + DT - 0.35, f1), r"$\Delta t$")

    ax.text(e1[0] - 0.15, e1[1] + 0.45, r"$\beta_0$", color=FG, fontsize=14,
            ha="center")
    ax.text(e2[0] - 0.15, e2[1] + 0.45, r"$\beta_0$", color=FG, fontsize=14,
            ha="center")
    ax.text(0.52, 0.05, r"$\Omega_{\mathrm{cfg}}=\beta_0\Delta t=$ const.",
            transform=ax.transAxes, color=FG, fontsize=12, ha="center")


def panel_c(ax):
    """usCFG: beta_L = beta_R + d_beta -> lineares Anwachsen von Omega."""
    # Beide Arme starten im selben Punkt (t_x, f_x): Omega(t_x) = 0,
    # die Frequenzdifferenz waechst danach linear an.
    s_L, s_R = SLOPE + DBETA, SLOPE
    t_x, f_x = US_TX, US_FX
    f_c = f_x + F_SPAN
    t_L, t_R = t_x + F_SPAN / s_L, t_x + F_SPAN / s_R
    cn1 = (f_x, f_x + 2 * F_SPAN)

    e1 = chirped_band(ax, t_L, f_c, s_L, cnorm=cn1)            # beta_L
    e2 = chirped_band(ax, t_R, f_c, s_R, cnorm=cn1)            # beta_R

    diamond(ax, t_L, f_c, filled=False)
    diamond(ax, t_R, f_c, filled=True)

    # entgegengesetzte zirkulare Polarisation der beiden Arme
    # links neben den steileren, rechts neben den flacheren Arm (nicht am Rahmen)
    circ_arrow(ax, e1[0] - 1.1, e1[1] - 0.6, ccw=True)
    circ_arrow(ax, e2[0] + 0.6, e2[1] - 0.9, ccw=False)


# ----------------------------------------------------------------------
# Zeitintegrierte Spektren (Frequenz vertikal, wie in den Spektrogrammen)
# ----------------------------------------------------------------------
def gauss_spectrum(f, f_c):
    """Spektrale Einhuellende, gleiche Form wie die Banddicke in chirped_band."""
    sigma = F_SPAN / TAPER
    return np.exp(-((f - f_c) / sigma) ** 2)


def spectrum(ax, f, inten, cnorm, cmap=SPECTRAL, envelope=None):
    """Spektrum als nach Frequenz eingefaerbte Flaeche, Intensitaet nach rechts."""
    verts = np.stack([
        np.column_stack([np.zeros(len(f) - 1), f[:-1]]),
        np.column_stack([inten[:-1], f[:-1]]),
        np.column_stack([inten[1:], f[1:]]),
        np.column_stack([np.zeros(len(f) - 1), f[1:]]),
    ], axis=1)
    f_mid = 0.5 * (f[:-1] + f[1:])
    v0, v1 = cnorm
    colors = cmap(np.clip((f_mid - v0) / (v1 - v0), 0, 1))
    ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors=colors,
                                     linewidths=0.4, zorder=3))
    ax.plot(inten, f, color=FG, lw=0.5, zorder=4)
    if envelope is not None:
        ax.plot(envelope, f, ls=(0, (3, 2)), lw=0.6, color=TAILC, zorder=5)


def panel_b_spec(ax):
    """CFG: Die Arme sind disjunkte Haelften eines Spektrums -> keine
    Interferenz, das integrierte Spektrum ist das urspruengliche Gauss-Spektrum."""
    f = np.linspace(*YLIM, 2000)
    spectrum(ax, f, gauss_spectrum(f, CFG_F0),
             cnorm=(CFG_F0 - F_SPAN, CFG_F0 + F_SPAN))


def panel_d_spec(ax):
    """usCFG: Beide Arme tragen das volle Spektrum A(w), mit verschiedenen
    spektralen Phasen.  I(w) ~ |A(w)|^2 * (1 + cos dphi(w)), wobei
    d(dphi)/dw = Laufzeitdifferenz der Arme bei w.  Diese waechst linear vom
    gemeinsamen Startpunkt f_x an -> dphi ~ (w - f_x)^2, d.h. gechirpte Fringes:
    breit bei niedriger, dicht bei hoher Frequenz."""
    f = np.linspace(*YLIM, 4000)
    f_c = US_FX + F_SPAN
    env = gauss_spectrum(f, f_c)
    # Skaliert so, dass ueber die Bandbreite [f_x, f_x + 2*F_SPAN] N_FRINGES liegen.
    dphi = 2 * np.pi * N_FRINGES * ((f - US_FX) / (2 * F_SPAN)) ** 2
    spectrum(ax, f, env * 0.5 * (1 + np.cos(dphi)),
             cnorm=(US_FX, US_FX + 2 * F_SPAN), envelope=env)


# ----------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, layout="constrained",
                             gridspec_kw={"width_ratios": WIDTH_RATIOS})
    # Spektrum teilt die Frequenzachse mit dem Spektrogramm -> fast anliegend
    fig.get_layout_engine().set(w_pad=1 / 72, h_pad=1 / 72, wspace=0.02, hspace=0.02)
    fig.patch.set_facecolor(BG)

    rows = [((panel_a, "(a1)"), (panel_b_spec, "(a2)")),
            ((panel_c, "(b1)"), (panel_d_spec, "(b2)"))]
    for i, ((ax_t, ax_s), ((gram_fn, gram_lab), (spec_fn, spec_lab))) in enumerate(
            zip(axes, rows)):
        bottom = i == len(rows) - 1          # x-Labels nur in der unteren Zeile
        style_axes(ax_t, gram_lab, xlabel=LANG_X if bottom else None)
        gram_fn(ax_t)
        ax_t.set_ylabel(LANG_Y, color=ACCENT, labelpad=3)
        style_axes(ax_s, spec_lab, xlabel="Intensity" if bottom else None,
                   xlim=SPEC_XLIM, label_right=True)
        spec_fn(ax_s)

    for ext in ("png", "pdf"):
        fig.savefig(SAVE.with_suffix(f".{ext}"), dpi=300, facecolor=BG)
    print(f"gespeichert: {SAVE}.png / .pdf")


if __name__ == "__main__":
    main()
