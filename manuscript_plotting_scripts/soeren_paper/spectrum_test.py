"""
Optisches Zentrifugen-Spektrum — physikalische Einheiten (Kevin's paper)
========================================================================
Einheiten:  Frequenz in rad/ps,  Zeit in ps,  Chirp in rad/ps²,  Länge in mm

Standardwerte aus Kevin's paper:
  ω₀        = 2π × 375 THz  ≈ 2356 rad/ps   (~ 800 nm)
  T_FWHM    = 300 ps
  β₀        = 3.70e-2  rad/ps²   (Basis-Chirp)
  Δβ        = 5.90e-4  rad/ps²   (Chirp-Mismatch: |αR| − |αL|)
  Δz        = 1.5 mm              (Armlängendifferenz)

  αR = +β₀ + Δβ/2,   αL = −(β₀ − Δβ/2)

Abhängigkeiten:  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ── Konstanten ────────────────────────────────────────────────────────────────

C_MMPS   = 2.998e-1          # Lichtgeschwindigkeit in mm/ps
W0       = 2 * np.pi * 375.0 # rad/ps  (375 THz)
FWHM2SIG = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM → Gauss-σ


# ── Physik ────────────────────────────────────────────────────────────────────

def compute(dwR0, dwL0, beta0, delta_beta, T_fwhm, delta_z_mm, Dphi, N=1200):
    """
    dwR0, dwL0    : Abweichung von ω₀  [rad/ps]
    beta0         : Basis-Chirp β₀     [rad/ps²]
    delta_beta    : Chirp-Mismatch Δβ  [rad/ps²]
    T_fwhm        : Pulsdauer (FWHM)   [ps]
    delta_z_mm    : Armlängendiff. Δz  [mm]
    Dphi          : Phasendiff. φR−φL  [rad]
    """
    wR0 = W0 + dwR0
    wL0 = W0 + dwL0
    aR  = beta0
    aL  = beta0 + delta_beta
    T   = T_fwhm * FWHM2SIG           # Gauss-σ in ps
    Dz  = delta_z_mm / C_MMPS         # Wegunterschied → Zeitverzögerung in ps

    T2, T4 = T**2, T**4
    DR = 1.0/T4 + aR**2
    DL = 1.0/T4 + aL**2

    # Spektrum ±4.5 spektrale Breiten um ω₀
    sig_w = T * np.sqrt(max(DR, DL))
    ws    = np.linspace(W0 - 4.5*sig_w, W0 + 4.5*sig_w, N)
    dws   = ws - W0   # Offset-Achse für Plot

    dR = ws - wR0
    dL = ws - wL0

    # |Ẽ_j(ω)|²  ∝  1/√D_j · exp[−(ω−ω_j0)²/(T²·D_j)]
    IR = (1.0/np.sqrt(DR)) * np.exp(-dR**2 / (T2 * DR))
    IL = (1.0/np.sqrt(DL)) * np.exp(-dL**2 / (T2 * DL))

    # Phase von Ẽ_j(ω)
    phaseR = (  0.5 * np.arctan2(-aR, 1.0/T2)
               - aR * dR**2 / (2.0*DR) )
    phaseL = (  0.5 * np.arctan2(-aL, 1.0/T2)
               - aL * dL**2 / (2.0*DL)
               - ws * Dz
               + Dphi )

    theta = phaseR - phaseL
    cross = 2.0 * np.sqrt(IR * IL) * np.cos(theta)
    Ix    = IR + IL + cross

    return dws, IR, IL, cross, Ix, theta, sig_w


# ── Standardwerte ─────────────────────────────────────────────────────────────

DEF = dict(
    dwR0       =  0.0,
    dwL0       =  0.0,
    beta0      =  3.70e-2,
    delta_beta =  5.90e-4,
    T_fwhm     =  300.0,
    delta_z_mm =  1.5,
    Dphi       =  0.0,
)

# ── Figur-Layout ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(13, 6.5))
fig.patch.set_facecolor('#fafafa')
plt.subplots_adjust(left=0.30, right=0.97, top=0.93, bottom=0.08)

ax = fig.add_axes([0.30, 0.12, 0.67, 0.78])
ax.set_facecolor('white')
ax.grid(True, color='#e0e0e0', linewidth=0.6)
ax.set_xlabel('Δω = ω − ω₀  [rad/ps]', fontsize=11)
ax.set_ylabel('Intensität (normiert)', fontsize=11)
ax.set_title('Optisches Zentrifugen-Spektrum  |  ω₀ = 2π·375 THz ≈ 800 nm',
             fontsize=12, fontweight='normal')

# ── Slider ────────────────────────────────────────────────────────────────────
#  (label,              key,           left,  bot,    min,     max,    step,    fmt)
SLDEFS = [
    ('Δω_R0  [rad/ps]',  'dwR0',       0.02, 0.88,  -3.0,    3.0,    0.05,  '%.2f'),
    ('Δω_L0  [rad/ps]',  'dwL0',       0.02, 0.80,  -3.0,    3.0,    0.05,  '%.2f'),
    ('β₀  [rad/ps²]',    'beta0',      0.02, 0.70,   0.005,  0.10,   0.001, '%.3f'),
    ('Δβ  [rad/ps²]',    'delta_beta', 0.02, 0.62,  -0.005,  0.005,  0.0001,'%.4f'),
    ('T_FWHM  [ps]',     'T_fwhm',     0.02, 0.52,  50.0,    800.0,  5.0,   '%.0f'),
    ('Δz  [mm]',         'delta_z_mm', 0.02, 0.44,  -5.0,    5.0,    0.05,  '%.2f'),
    ('Δφ  [rad]',        'Dphi',       0.02, 0.36,  -3.14,   3.14,   0.05,  '%.2f'),
]

sliders = {}
for label, key, l, b, vmin, vmax, step, fmt in SLDEFS:
    ax_s = fig.add_axes([l, b, 0.22, 0.025])
    sl   = Slider(ax_s, label, vmin, vmax,
                  valinit=DEF[key], valstep=step,
                  valfmt=fmt, color='#534AB7')
    sl.label.set_fontsize(9)
    sl.valtext.set_fontsize(9)
    sliders[key] = sl

# ── Radio-Buttons ─────────────────────────────────────────────────────────────

ax_radio = fig.add_axes([0.02, 0.08, 0.22, 0.22])
ax_radio.set_facecolor('#fafafa')
radio = RadioButtons(ax_radio,
                     ('Gesamtspektrum', 'Terme aufgeteilt', 'Phase Θ(ω)'),
                     active=0, activecolor='#534AB7')
for lbl in radio.labels:
    lbl.set_fontsize(9)

# Abgeleitete Größen + Warnung
info_text = fig.text(0.02, 0.31, '', fontsize=8, color='#444',
                     family='monospace', va='top')
warn_text = fig.text(0.02, 0.26, '', fontsize=8.5, color='#993C1D')

# ── Plot-Linien ───────────────────────────────────────────────────────────────

ln_total,  = ax.plot([], [], color='#534AB7', lw=2,   label='Iₓ(ω)')
ln_IR,     = ax.plot([], [], color='#0F6E56', lw=1.5, ls='--', label='|Ẽ_R|²')
ln_IL,     = ax.plot([], [], color='#BA7517', lw=1.5, ls=':',  label='|Ẽ_L|²')
ln_cross,  = ax.plot([], [], color='#993C1D', lw=1.5, label='2Re[Ẽ_R Ẽ_L*]')
ln_sum,    = ax.plot([], [], color='#534AB7', lw=2,   label='Σ')
ln_phase,  = ax.plot([], [], color='#185FA5', lw=2,   label='Θ(ω)')
fill_obj   = [None]


# ── Update ────────────────────────────────────────────────────────────────────

def update(_=None):
    p = {key: sl.val for key, sl in sliders.items()}
    dws, IR, IL, cross, Ix, theta, _ = compute(**p)

    aR = p['beta0'] + p['delta_beta']/2
    aL = -(p['beta0'] - p['delta_beta']/2)
    T  = p['T_fwhm'] * FWHM2SIG
    Dz = p['delta_z_mm'] / C_MMPS

    info_text.set_text(
        f"αR = {aR:+.5f} rad/ps²\n"
        f"αL = {aL:+.5f} rad/ps²\n"
        f"T  = {T:.1f} ps  (σ)\n"
        f"Δz/c = {Dz:.3f} ps"
    )

    dw = abs(p['dwR0'] - p['dwL0'])
    da = abs(abs(aR) - abs(aL))
    warn_text.set_text('⚠ Amplituden ≠ → kein reines cos²'
                       if (dw > 0.05 or da > 1e-3) else '')

    mode = radio.value_selected
    norm = max(Ix.max(), 1e-30)

    for ln in [ln_total, ln_IR, ln_IL, ln_cross, ln_sum, ln_phase]:
        ln.set_data([], [])
    if fill_obj[0] is not None:
        fill_obj[0].remove(); fill_obj[0] = None

    ax.set_ylabel('Intensität (normiert)', fontsize=11)

    if mode == 'Gesamtspektrum':
        ln_total.set_data(dws, Ix / norm)
        fill_obj[0] = ax.fill_between(dws, 0, Ix/norm,
                                      color='#534AB7', alpha=0.10)
        ax.legend(handles=[ln_total], fontsize=9, loc='upper right')

    elif mode == 'Terme aufgeteilt':
        ln_IR.set_data(dws,    IR    / norm)
        ln_IL.set_data(dws,    IL    / norm)
        ln_cross.set_data(dws, cross / norm)
        ln_sum.set_data(dws,   Ix    / norm)
        ax.legend(handles=[ln_IR, ln_IL, ln_cross, ln_sum],
                  fontsize=9, loc='upper right')

    else:
        ln_phase.set_data(dws, theta)
        ax.set_ylabel('Θ(ω)  [rad]', fontsize=11)
        ax.legend(handles=[ln_phase], fontsize=9, loc='upper right')

    ax.set_xlim(dws[0], dws[-1])
    if mode == 'Phase Θ(ω)':
        m = max(abs(theta).max() * 1.15, 0.5)
        ax.set_ylim(-m, m)
    else:
        ymin = min((cross/norm).min(), 0) * 1.15
        ax.set_ylim(ymin, 1.15)

    fig.canvas.draw_idle()


for sl in sliders.values():
    sl.on_changed(update)
radio.on_clicked(update)

update()
plt.show()