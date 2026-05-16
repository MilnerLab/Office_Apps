"""
Optisches Zentrifugen-Spektrum — Paper-Style Skizze
====================================================
Rendern:
    manim -s -qh spectrum_sketch.py SpectrumSketch
"""

import numpy as np
from manim import *

# ── Physik ────────────────────────────────────────────────────────────────────

C_MMPS   = 2.998e-1
W0       = 2 * np.pi * 375.0
FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def compute_spectrum(
    beta0=3.70e-2, delta_beta=-9.90e-4,   # Chirp umgedreht
    T_fwhm=200.0,                          # etwas breiter
    delta_z_mm=1.5,
    Dphi=0.0, N=2000,
    window_sigma=2.2,                      # Sichtfenster ±window_sigma · sig_w
):
    aR = beta0
    aL = beta0 + delta_beta
    T  = T_fwhm * FWHM2SIG
    Dz = delta_z_mm / C_MMPS

    T2, T4 = T**2, T**4
    DR = 1.0/T4 + aR**2
    DL = 1.0/T4 + aL**2

    sig_w = T * np.sqrt(max(DR, DL))
    ws    = np.linspace(W0 - window_sigma*sig_w,
                        W0 + window_sigma*sig_w, N)
    dws   = ws - W0

    dR = ws - W0
    dL = ws - W0

    IR = (1.0/np.sqrt(DR)) * np.exp(-dR**2 / (T2 * DR))
    IL = (1.0/np.sqrt(DL)) * np.exp(-dL**2 / (T2 * DL))

    phaseR = 0.5 * np.arctan2(-aR, 1.0/T2) - aR * dR**2 / (2.0*DR)
    phaseL = (0.5 * np.arctan2(-aL, 1.0/T2)
              - aL * dL**2 / (2.0*DL)
              - ws * Dz + Dphi)

    theta  = phaseR - phaseL
    Ix     = IR + IL + 2.0 * np.sqrt(IR * IL) * np.cos(theta)

    return dws, Ix / Ix.max()


# ── Scene ─────────────────────────────────────────────────────────────────────

class SpectrumSketch(Scene):

    def construct(self):
        self.camera.background_color = WHITE

        BOX_BG    = "#EBEBEB"
        BOX_EDGE  = "#BBBBBB"
        AXIS_COL  = "#111111"
        CURVE_COL = "#E8735A"
        LABEL_COL = "#111111"

        # ── Kästchen ──────────────────────────────────────────────────────────
        box = RoundedRectangle(
            corner_radius=0.28,
            width=7.6, height=4.8,
            fill_color=BOX_BG, fill_opacity=1.0,
            stroke_color=BOX_EDGE, stroke_width=5.0,
        )

        # ── Achsen ────────────────────────────────────────────────────────────
        axes = Axes(
            x_range=[0, 1, 1],
            y_range=[0, 1, 1],
            x_length=5.6,
            y_length=3.2,
            axis_config={
                "color":         AXIS_COL,
                "stroke_width":  6,
                "include_tip":   True,
                "tip_length":    0.25,
                "tip_width":     0.17,
                "include_ticks": False,
            },
        )
        axes.move_to(box.get_center() + RIGHT * 0.3 + DOWN * 0.3)

        # ── Achsenbeschriftung (nur Namen, keine Zahlen) ───────────────────────
        lbl_x = MathTex(r"\omega", font_size=55, color=LABEL_COL)
        lbl_x.next_to(axes.x_axis.get_end(), RIGHT, buff=0.18)

        lbl_y = MathTex(r"I(\omega)", font_size=55, color=LABEL_COL)
        lbl_y.next_to(axes.y_axis.get_end(), UP, buff=0.14)

        # ── Spektrumskurve ────────────────────────────────────────────────────
        dws, Ix = compute_spectrum()

        dws_norm = (dws - dws[0]) / (dws[-1] - dws[0])

        # Kurve vor dem Pfeil stoppen (x < 0.92)
        step   = 2
        mask   = dws_norm[::step] < 0.92
        xs     = dws_norm[::step][mask]
        ys     = Ix[::step][mask]
        points = [axes.c2p(float(x), float(y)) for x, y in zip(xs, ys)]

        curve = VMobject()
        curve.set_points_as_corners(points)
        curve.make_smooth()
        curve.set_stroke(color=CURVE_COL, width=6.0)

        # ── Zusammensetzen ────────────────────────────────────────────────────
        self.add(box, axes, lbl_x, lbl_y, curve)