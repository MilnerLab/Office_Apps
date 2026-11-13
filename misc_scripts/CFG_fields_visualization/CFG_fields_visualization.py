#EVERYTHING HERE IS CLAUDE-CODED WITH NO REVIEW.
#
#BEWARE!
"""The two composite fields of a constant-frequency optical centrifuge (cfCFG) and their sum (GUI).

One chirped pulse is split into two arms with the same spectral phase, arm 2 delayed by tau.
Because both arms have the same chirp, their instantaneous frequencies differ by
(chirp rate) * tau at every time. When the arms are oppositely circularly polarized, the linear
polarization of their sum rotates at

    f_rot(t) = (omega_1(t) - omega_2(t)) / (4 pi) ~ tau * (d omega / dt) / (4 pi).

With pure GDD the chirp rate, and so f_rot, is constant. Third-order dispersion makes the chirp
rate drift through the pulse, which is the residual acceleration of the centrifuge.

Neither the GDD nor the TOD is known exactly, so they are solved for here from:
  * the pulse duration: phi2 is chosen so that one arm's intensity FWHM is the set duration;
  * the acceleration: phi3 is chosen so that f_rot rises by the set acceleration between the two
    half-maximum points of the combined intensity, defined as f_rot(end) / f_rot(start) - 1.
Stationary phase gives the starting guesses: phi2 ~ T / d_omega and
phi3 ~ -accel * phi2 / d_omega (negative, so that the rotation speeds up with time).
This solve uses the beat between the arms, which does not depend on their polarizations, so
the dispersion is that of the real (circular) centrifuge whatever polarizations are chosen. It
is also cached: changing only the polarization (or the 3D plot settings) redraws without
solving again. Alternatively the GDD and TOD can be given directly (defaults: the solve's result
for 320 ps / 14 %), which skips the solve, and with it the solve's need for a long time window.

Extra GDD and TOD can be added to one arm only, on top of the shared dispersion. The arms then
have different chirps, so the beat (and the rotation) is no longer near-constant; the solve,
when used, still sets only the shared part.

An arm with only the shared dispersion is generated with a peak field amplitude of 1; both arms
have the same spectrum and so the same energy, so an arm stretched further by extra GDD has a
lower peak. Nothing is renormalized after the arms are combined (through the QWP and polarizer):
the plot and the results show the combination's actual field, e.g. a peak |E| of sqrt 2 for the
normal 3D centrifuge.

Polarization: each arm starts linearly polarized (arm 1 at its angle, arm 2 offset from it).
After they are combined, both can pass through a quarter-wave plate and then a linear polarizer.
Linear at 0 and 90 deg with the plate at 45 deg makes them opposite circular: the usual
centrifuge. Any other choice gives elliptical light whose orientation psi and ellipticity chi are
read off the Stokes parameters of the summed field. With the polarizer in, the output is linear
along its axis, and the rotation appears as intensity beating instead.

Units come from base_core (Length / Time / Angle inputs, SPEED_OF_LIGHT, and Frequency,
AngularFrequency, AngularChirp and GDD for results); internally everything is plain SI floats,
which is what those quantities store. There is no base_core type for third-order dispersion, so
phi3 stays a float in s^3.

Fields are complex envelopes about omega0 on a baseband grid, with field ~ exp(+i Phi(t)) as in
base_core/physics/optical_centrifuge.py. Angles are measured from x towards y.

Run from Office_Apps:  python -m misc_scripts.CFG_fields_visualization.CFG_fields_visualization
"""
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
import json
from pathlib import Path
import sys
import time

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from scipy.optimize import brentq

from base_core.framework.events import EventBus
from base_core.framework.serialization.serialization import from_primitive, to_primitive
from base_core.math.enums import AngleUnit
from base_core.math.models import Angle
from base_core.physics.optical_centrifuge import BETA_0
from base_core.quantities.constants import SPEED_OF_LIGHT
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Length, Time
from base_core.quantities.specific_models import GDD, AngularChirp, AngularFrequency
from base_qt.app.dispatcher import QtDispatcher
from base_qt.ui import LabMainWindow, MessageLevel, Panel, PanelViewModel
from base_qt.ui.apply import install_ui
from base_qt.ui.form import (
    AngleSpec,
    BoolSpec,
    DirtyFormWidget,
    EnumSpec,
    FloatSpec,
    GDDSpec,
    IntSpec,
    LengthSpec,
    TimeSpec,
)


# ============================================================
# Settings (all editable in the window)
# ============================================================
class Arm(Enum):
    ARM_1 = 1
    ARM_2 = 2


@dataclass
class CfgFieldsSettings:
    # ---- Pulse parameters ----
    wavelength: Length = Length(800.0, Prefix.NANO)
    bandwidth: Length = Length(8.2, Prefix.NANO)          # spectral intensity FWHM
    # Dispersion shared by both arms: solved for from duration_fwhm and acceleration_percent, or
    # (dispersion_given, the default) taken directly from gdd and tod_ps3
    dispersion_given: bool = True
    duration_fwhm: Time = Time(320.0, Prefix.PICO)        # intensity FWHM of each (stretched) arm
    acceleration_percent: float = 14.0                    # f_rot(end of FWHM) / f_rot(start) - 1, in %
    gdd: GDD = GDD(13.28, Prefix.PICO)                    # defaults: the solve's result for 320 ps / 14 %
    tod_ps3: float = -0.071573                            # ps^3 (no base_core TOD type)
    # Extra dispersion on one arm only, on top of the shared dispersion
    extra_dispersion_in: bool = False
    extra_arm: Arm = Arm.ARM_2
    extra_gdd: GDD = GDD(0.0, Prefix.PICO)
    extra_tod_ps3: float = 0.0                            # ps^3
    delay: Time = Time(1.0, Prefix.PICO)                  # arm 2 trails arm 1 by this
    relative_phase: Angle = Angle(0.0)                    # extra phase on arm 2: offsets the polarization angle
    # ---- Polarization ----
    # Each arm is linearly polarized before the combiner (angles from x towards y). Defaults: the
    # normal 3D centrifuge (same as that preset)
    arm1_pol: Angle = Angle(0.0, AngleUnit.DEG)           # arm 1 (leading)
    pol_offset: Angle = Angle(90.0, AngleUnit.DEG)        # arm 2 (delayed) is at arm1_pol + pol_offset
    # Quarter-wave plate after the combiner, acting on both arms. Arms at 0 / 90 deg with the
    # plate at 45 deg = opposite circular (standard cfCFG).
    qwp_in: bool = True
    qwp_angle: Angle = Angle(45.0, AngleUnit.DEG)         # fast axis
    # Linear polarizer after the QWP (or after the combiner, if there is no QWP). The output is
    # then always linear along its axis, and the rotation shows up as intensity beating instead.
    polarizer_in: bool = False
    polarizer_angle: Angle = Angle(0.0, AngleUnit.DEG)    # transmission axis
    # ---- Computation ----
    n_points_log2: int = 16                               # grid of 2**n points
    window: Time = Time(4000.0, Prefix.PICO)              # time window; dt = window / 2**n, Nyquist = 1 / (2 dt)
    ribbon_points: int = 300                              # samples along the 3D ribbon (whole pulse)
    ribbon_phase_points: int = 25                         # samples around each polarization ellipse
    ribbon_cutoff: float = 1e-2                           # ribbon drawn where the combined intensity exceeds this x peak


# The fields the GDD/TOD solve depends on; changing anything else reuses the cached solve (and
# with the dispersion given there is no solve at all)
SOLVE_FIELDS = ("wavelength", "bandwidth", "duration_fwhm", "acceleration_percent", "delay",
                "relative_phase", "n_points_log2", "window")

# A user-editable settings snapshot persisted next to this script, written/read by the
# "Save config" / "Load config" buttons. It does not replace CfgFieldsSettings' defaults above.
SIDECAR_CONFIG_PATH = Path(__file__).with_name("CFG_fields_visualization_config.json")
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "_temp" / "CFG_fields_visualization"

PS = Prefix.PICO.value                                    # s per ps, for plotting arrays in ps
SIDE_PANEL_MIN_WIDTH = 420                                # px; widened if the results text needs it
MIN_WINDOW_SIZE = (1600, 800)                             # px; grown to fit the side panel's contents, up to the screen size


def save_settings_to_sidecar(settings: CfgFieldsSettings, path: Path = SIDECAR_CONFIG_PATH) -> None:
    path.write_text(json.dumps(to_primitive(settings), indent=2), encoding="utf-8")


def load_settings_from_sidecar(path: Path = SIDECAR_CONFIG_PATH) -> CfgFieldsSettings:
    return from_primitive(CfgFieldsSettings, json.loads(path.read_text(encoding="utf-8")))


# ============================================================
# Grid and fields
# ============================================================
@dataclass(frozen=True)
class Grid:
    t: np.ndarray        # (N,) s, centred on 0 (index N // 2)
    omega: np.ndarray    # (N,) rad/s, detuning from omega0, in fft order
    dt: float            # s


class Pulse:
    """The chirped pulse the two arms are cut from, and the grid it lives on."""

    def __init__(self, s: CfgFieldsSettings) -> None:
        if s.bandwidth <= 0:
            raise ValueError("Bandwidth must be positive.")
        if s.delay <= 0:
            raise ValueError("Delay must be positive (with no delay there is no rotation).")
        self.omega0 = AngularFrequency.from_frequency(Frequency(SPEED_OF_LIGHT / s.wavelength))
        # bandwidth at the wavelength as an angular-frequency FWHM: d_omega = 2 pi c d_lambda / lambda^2
        self.d_omega = AngularFrequency(2 * np.pi * SPEED_OF_LIGHT * s.bandwidth / s.wavelength**2)
        self.tau = s.delay
        self.relative_phase = s.relative_phase
        n = 2 ** s.n_points_log2
        dt = s.window / n
        t = (np.arange(n) - n // 2) * dt
        self.grid = Grid(t, 2 * np.pi * np.fft.fftfreq(n, dt), dt)

    def spectral_amplitude(self, omega: np.ndarray) -> np.ndarray:
        # Gaussian whose intensity |A|^2 has FWHM d_omega
        return np.exp(-2 * np.log(2) * (omega / self.d_omega) ** 2)

    def arm_field(self, phi2: float, phi3: float, delay: float, extra_phase: float = 0.0) -> np.ndarray:
        """Complex scalar envelope E(t - delay) of one arm, unnormalized (arms() scales it).

        With field ~ exp(+i Omega t), a spectral phase -phi(Omega) gives group delay +phi'(Omega),
        so phi2 > 0 is an up-chirp. Delaying the full field exp(i omega0 t) E(t) by `delay` also
        puts exp(-i omega0 delay) on the envelope.
        """
        omega = self.grid.omega
        phase = phi2 * omega**2 / 2 + phi3 * omega**3 / 6
        spectrum = self.spectral_amplitude(omega) * np.exp(-1j * phase - 1j * omega * delay)
        field = np.fft.fftshift(np.fft.ifft(spectrum))
        return field * np.exp(-1j * self.omega0 * delay + 1j * extra_phase)

    def arms(self, phi2: float, phi3: float, extra: "ExtraDispersion | None" = None) -> tuple[np.ndarray, np.ndarray]:
        """Arm 1 leads by tau/2, arm 2 trails by tau/2, so the centrifuge is centred on t = 0.
        Both have dispersion (phi2, phi3); `extra` adds more to one of them. A pulse with only the
        shared dispersion peaks at |E| = 1; one with extra dispersion has the same energy, so a
        lower peak."""
        d1 = d2 = (0.0, 0.0)
        if extra is not None:
            if extra.arm is Arm.ARM_1:
                d1 = (extra.gdd, extra.tod)
            else:
                d2 = (extra.gdd, extra.tod)
        e_1 = self.arm_field(phi2 + d1[0], phi3 + d1[1], -self.tau / 2)
        e_2 = self.arm_field(phi2 + d2[0], phi3 + d2[1], +self.tau / 2, self.relative_phase)
        # One scale for both arms (they have the same spectrum, so the same energy): the arm with
        # only the shared dispersion peaks at |E| = 1, and an arm stretched further by extra GDD
        # comes out with a correspondingly lower peak
        reference = e_2 if extra is not None and extra.arm is Arm.ARM_1 else e_1
        scale = 1 / np.max(np.abs(reference))
        return e_1 * scale, e_2 * scale


# ============================================================
# Derived quantities
# ============================================================
def half_max_crossings(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    half = y.max() / 2
    above = np.nonzero(y >= half)[0]
    i0, i1 = above[0], above[-1]
    if i0 == 0 or i1 == len(y) - 1:
        raise ValueError("The pulse runs off the edge of the time window: increase the window.")
    t_left = np.interp(half, [y[i0 - 1], y[i0]], [t[i0 - 1], t[i0]])
    t_right = np.interp(half, [y[i1 + 1], y[i1]], [t[i1 + 1], t[i1]])
    return float(t_left), float(t_right)


def phase_difference(e_1: np.ndarray, e_2: np.ndarray) -> np.ndarray:
    """Phi_1 - Phi_2, unwrapped. Taken from E_1 E_2* so the large common chirp phase cancels."""
    return np.unwrap(np.angle(e_1 * np.conj(e_2)))


def beat_rotation_frequency(e_1: np.ndarray, e_2: np.ndarray, dt: float) -> np.ndarray:
    """(omega_1 - omega_2) / (4 pi), Hz: the rotation rate the arms would give if opposite
    circular, whatever polarization is actually chosen."""
    return np.gradient(phase_difference(e_1, e_2), dt) / (4 * np.pi)


def acceleration(grid: Grid, e_1: np.ndarray, e_2: np.ndarray) -> tuple[float, Time, Time, Frequency, Frequency]:
    """(accel, t_start, t_end, f_start, f_end) of the beat rotation over the combined FWHM."""
    total = np.abs(e_1) ** 2 + np.abs(e_2) ** 2
    t0, t1 = half_max_crossings(grid.t, total)
    f = beat_rotation_frequency(e_1, e_2, grid.dt)
    f0, f1 = np.interp([t0, t1], grid.t, f)
    return f1 / f0 - 1, Time(t0), Time(t1), Frequency(f0), Frequency(f1)


# ============================================================
# Dispersion: solved for, or given; and the resulting fields
# ============================================================
@dataclass(frozen=True)
class ExtraDispersion:
    arm: "Arm"
    gdd: float           # s^2
    tod: float           # s^3


def extra_dispersion(s: CfgFieldsSettings) -> ExtraDispersion | None:
    if not s.extra_dispersion_in:
        return None
    return ExtraDispersion(s.extra_arm, float(s.extra_gdd), s.extra_tod_ps3 * PS**3)


@dataclass(frozen=True)
class Solution:
    pulse: Pulse
    phi2: GDD            # dispersion shared by both arms (solved or given)
    phi3: float          # s^3 (base_core has no TOD type)
    extra: ExtraDispersion | None
    e_1: np.ndarray      # complex envelopes of the two arms, before any polarization optics
    e_2: np.ndarray
    accel: float
    t_start: Time        # combined-intensity FWHM edges
    t_end: Time
    f_start: Frequency   # circular-reference rotation rate at those edges and at t = 0
    f_end: Frequency
    f_centre: Frequency
    arm1_fwhm: Time
    arm2_fwhm: Time
    solve_seconds: float | None   # None when the dispersion was given, not solved


def single_arm_fwhm(pulse: Pulse, phi2: float, phi3: float) -> Time:
    t0, t1 = half_max_crossings(pulse.grid.t, np.abs(pulse.arm_field(phi2, phi3, 0.0)) ** 2)
    return Time(t1 - t0)


def fwhm(grid: Grid, field: np.ndarray) -> Time:
    t0, t1 = half_max_crossings(grid.t, np.abs(field) ** 2)
    return Time(t1 - t0)


# The time window must hold the pulse with room to spare: the GDD search tries pulses up to twice
# the target length, and in a window shorter than ~4 FWHM their wings wrap around the grid edges
# and corrupt the measured width
MIN_WINDOW_PER_FWHM = 5.0


def min_window(duration: float, s: CfgFieldsSettings) -> Time:
    return Time(MIN_WINDOW_PER_FWHM * duration + s.delay)


def check_window(duration: float, s: CfgFieldsSettings, what: str) -> None:
    if s.window < min_window(duration, s):
        raise ValueError(
            f"Time window {s.window.value(Prefix.PICO):.0f} ps is too short for {what} "
            f"(~{Time(duration).value(Prefix.PICO):.0f} ps): use at least "
            f"{min_window(duration, s).value(Prefix.PICO):.0f} ps ({MIN_WINDOW_PER_FWHM:g} x duration + delay)."
        )


def find_root(f, a: float, b: float, what: str) -> float:
    """brentq, with an explanation instead of SciPy's "f(a) and f(b) must have different signs"
    when the target is not between the two trial values."""
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb):
        raise ValueError(
            f"Could not reach the target {what}: it is not between the two trial values tried "
            f"(miss {fa:.3g} and {fb:.3g}). Usually the time window is too short, the duration is "
            "too close to the transform limit, or the acceleration is too large for the bandwidth."
        )
    return brentq(f, a, b, xtol=abs(b - a) * 1e-12)


def solve_dispersion(s: CfgFieldsSettings, pulse: Pulse) -> tuple[GDD, float, float]:
    """(phi2, phi3 in s^3, seconds taken): the dispersion shared by both arms that gives each a
    duration_fwhm pulse and the centrifuge acceleration_percent over its FWHM. Any extra
    dispersion on one arm is not part of this; it is added on top afterwards."""
    start = time.perf_counter()
    if s.duration_fwhm <= 0:
        raise ValueError("Duration must be positive.")
    check_window(s.duration_fwhm, s, f"a {s.duration_fwhm.value(Prefix.PICO):g} ps pulse")
    grid = pulse.grid
    target_accel = s.acceleration_percent / 100
    phi2 = s.duration_fwhm / pulse.d_omega
    phi3 = -target_accel * phi2 / pulse.d_omega
    # The two conditions are nearly decoupled (phi2 sets the duration, phi3/phi2 the
    # acceleration), so alternating 1D solves converge in a few rounds
    for _ in range(6):
        phi2 = find_root(lambda p2: (single_arm_fwhm(pulse, p2, phi3) - s.duration_fwhm) / s.duration_fwhm,
                         0.5 * phi2, 2.0 * phi2, f"duration of {s.duration_fwhm.value(Prefix.PICO):g} ps")
        if target_accel == 0:
            phi3 = 0.0
            continue
        g = max(abs(phi3), 0.01 * phi2 / pulse.d_omega)
        phi3 = find_root(lambda p3: acceleration(grid, *pulse.arms(phi2, p3))[0] - target_accel,
                         -4.0 * g, 4.0 * g, f"acceleration of {s.acceleration_percent:g}%")
    accel = acceleration(grid, *pulse.arms(phi2, phi3))[0]
    arm_fwhm = single_arm_fwhm(pulse, phi2, phi3)
    # The alternating solve can settle on a pair that satisfies each condition only in turn (e.g.
    # at large accelerations, where it can collapse the pulse), so check both at the end
    if abs(arm_fwhm / s.duration_fwhm - 1) > 0.01 or abs(accel - target_accel) > 0.005:
        raise ValueError(
            f"The GDD/TOD solve did not converge: got {arm_fwhm.value(Prefix.PICO):.2f} ps and "
            f"{100 * accel:.2f}% for targets {s.duration_fwhm.value(Prefix.PICO):g} ps and "
            f"{s.acceleration_percent:g}%. The acceleration is probably too large for this bandwidth."
        )
    return GDD(phi2), phi3, time.perf_counter() - start


# With the dispersion known (given, or already solved) nothing is searched, so the window only has
# to hold the actual pulses: their intensity at the window edges must be negligible, or the FFT
# wraps their wings around onto the other side. The rotation rate is very sensitive to that (a
# wrapped wing has a very different instantaneous frequency): for 320 ps / 14 %, edges at 2e-5
# still give 14.0 %, while 1e-4 gives 14.6 % and 3e-4 gives 11 %.
MAX_EDGE_INTENSITY = 3e-5   # x the peak


def needed_window(pulse: Pulse, arm_dispersions: list[tuple[float, float]]) -> Time:
    """Shortest window that holds every arm down to MAX_EDGE_INTENSITY. In a strongly chirped
    pulse each frequency arrives at its group delay phi2 W + phi3 W^2 / 2, so the pulse reaches the
    threshold at the delay of the frequency W where the spectrum does; the arms sit at +-tau/2 and
    the window is centred on t = 0."""
    w = pulse.d_omega * np.sqrt(np.log(1 / MAX_EDGE_INTENSITY) / (4 * np.log(2)))
    reach = max(abs(p2 * sign * w + p3 * w**2 / 2) for p2, p3 in arm_dispersions for sign in (1, -1))
    return Time(2 * (reach + pulse.tau / 2))


def check_edges(grid: Grid, e_1: np.ndarray, e_2: np.ndarray, s: CfgFieldsSettings, needed: Time) -> None:
    total = np.abs(e_1) ** 2 + np.abs(e_2) ** 2
    edge = max(total[0], total[-1]) / total.max()
    if edge > MAX_EDGE_INTENSITY:
        raise ValueError(
            f"Time window {s.window.value(Prefix.PICO):.0f} ps is too short for this pulse.\n"
            "The fields are computed with an FFT, which treats the time window as repeating end to\n"
            "end, so any part of the pulse that does not fit inside it wraps around onto the other end.\n"
            f"Here the pulse is still at {edge:.1e} of its peak at the window edges (the limit is "
            f"{MAX_EDGE_INTENSITY:.0e}).\n"
            "Wrapped light comes from the far end of the chirp, several THz away in frequency, so even\n"
            "this little of it makes the computed rotation rate wrong.\n"
            f"This pulse needs a window of about {needed.value(Prefix.PICO):.0f} ps: increase the time "
            "window\n(to make it faster, lower the grid points instead of the window)."
        )


def build_solution(s: CfgFieldsSettings, pulse: Pulse, phi2: GDD, phi3: float, solve_seconds: float | None) -> Solution:
    """The two arms for a known shared dispersion (plus any extra on one arm), and what they make."""
    extra = extra_dispersion(s)
    grid = pulse.grid
    e_1, e_2 = pulse.arms(phi2, phi3, extra)
    arm_dispersions = [(phi2, phi3)]
    if extra is not None:
        arm_dispersions.append((phi2 + extra.gdd, phi3 + extra.tod))
    check_edges(grid, e_1, e_2, s, needed_window(pulse, arm_dispersions))
    accel, t_start, t_end, f_start, f_end = acceleration(grid, e_1, e_2)
    f_centre = Frequency(np.interp(0.0, grid.t, beat_rotation_frequency(e_1, e_2, grid.dt)))
    return Solution(pulse, phi2, phi3, extra, e_1, e_2, accel, t_start, t_end, f_start, f_end, f_centre,
                    fwhm(grid, e_1), fwhm(grid, e_2), solve_seconds)


# ============================================================
# Polarization (Jones calculus)
# ============================================================
def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def quarter_wave_plate(fast_axis: Angle) -> np.ndarray:
    """Jones matrix of a QWP with its fast axis at `fast_axis`; global phase dropped."""
    return rotation(fast_axis) @ np.diag([1.0, 1j]) @ rotation(-fast_axis)


def linear(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)], dtype=complex)


def linear_polarizer(axis: Angle) -> np.ndarray:
    """Jones matrix of an ideal linear polarizer transmitting along `axis`."""
    n = linear(axis)
    return np.outer(n, n.conj())


def arm_jones_vectors(s: CfgFieldsSettings) -> tuple[np.ndarray, np.ndarray]:
    """Jones vectors (x, y) of arms 1 and 2 as they leave the optics. Unit length unless a
    polarizer is in, which leaves each with its transmitted amplitude."""
    j_1 = linear(s.arm1_pol)
    j_2 = linear(s.arm1_pol + s.pol_offset)
    if s.qwp_in:
        qwp = quarter_wave_plate(s.qwp_angle)
        j_1, j_2 = qwp @ j_1, qwp @ j_2
    if s.polarizer_in:
        pol = linear_polarizer(s.polarizer_angle)
        j_1, j_2 = pol @ j_1, pol @ j_2
    return j_1, j_2


def describe(j: np.ndarray) -> str:
    """Kind, orientation, ellipticity and transmission of a single Jones vector."""
    ex, ey = j
    s0, s1 = abs(ex) ** 2 + abs(ey) ** 2, abs(ex) ** 2 - abs(ey) ** 2
    if s0 < 1e-12:
        return "blocked"
    s2, s3 = 2 * np.real(ex * np.conj(ey)), 2 * np.imag(ex * np.conj(ey))
    psi = Angle(0.5 * np.arctan2(s2, s1))
    chi = Angle(0.5 * np.arcsin(np.clip(s3 / s0, -1, 1)))
    if abs(abs(chi.Deg) - 45) < 0.5:
        kind = "circular"
    elif abs(chi.Deg) < 0.5:
        kind = "linear"
    else:
        kind = "elliptical"
    return f"{kind}, psi {psi.Deg:.1f}°, chi {chi.Deg:.1f}°, T {s0:.3f}"


@dataclass(frozen=True)
class Polarization:
    """Polarization state of the summed field at each time."""
    ex: np.ndarray       # complex envelopes
    ey: np.ndarray
    s0: np.ndarray       # total intensity
    dolp: np.ndarray     # degree of linear polarization sqrt(S1^2 + S2^2) / S0
    psi: np.ndarray      # ellipse orientation (rad), unwrapped
    chi: np.ndarray      # ellipticity angle (rad): 0 linear, +-pi/4 circular


def combine(e_1: np.ndarray, e_2: np.ndarray, s: CfgFieldsSettings) -> Polarization:
    j_1, j_2 = arm_jones_vectors(s)
    ex = e_1 * j_1[0] + e_2 * j_2[0]
    ey = e_1 * j_1[1] + e_2 * j_2[1]
    s0 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    s1 = np.abs(ex) ** 2 - np.abs(ey) ** 2
    cross = ex * np.conj(ey)
    s2, s3 = 2 * np.real(cross), 2 * np.imag(cross)
    safe = np.where(s0 > 0, s0, 1.0)
    dolp = np.where(s0 > 0, np.hypot(s1, s2) / safe, 0.0)
    psi = 0.5 * np.unwrap(np.arctan2(s2, s1))
    chi = 0.5 * np.arcsin(np.clip(s3 / safe, -1, 1))
    return Polarization(ex, ey, s0, dolp, psi, chi)


# ============================================================
# Text summary and plot
# ============================================================
def polarization_label(s: CfgFieldsSettings) -> str:
    a2 = Angle(s.arm1_pol + s.pol_offset).Deg
    qwp = f", QWP at {s.qwp_angle.Deg:g}$^\\circ$" if s.qwp_in else ", no QWP"
    pol = f", polarizer at {s.polarizer_angle.Deg:g}$^\\circ$" if s.polarizer_in else ""
    return f"arms linear at {s.arm1_pol.Deg:g}$^\\circ$ / {a2:g}$^\\circ${qwp}{pol}"


def summary(sol: Solution, s: CfgFieldsSettings) -> str:
    pulse = sol.pulse
    beta = AngularChirp(1 / (2 * sol.phi2))   # equivalent time-domain chirp, as base_core uses
    nyquist = Frequency(1 / (2 * pulse.grid.dt))
    j_1, j_2 = arm_jones_vectors(s)
    peak_e = float(np.sqrt(combine(sol.e_1, sol.e_2, s).s0.max()))   # after QWP / polarizer, not renormalized
    source = "given" if sol.solve_seconds is None else "solved"
    extra_lines = []
    if sol.extra is not None:
        name = "arm 1" if sol.extra.arm is Arm.ARM_1 else "arm 2"
        extra_lines = [f"Extra on {name}  {GDD(sol.extra.gdd).value(Prefix.PICO):+.5g} ps², "
                       f"{sol.extra.tod / PS**3:+.5g} ps³"]
    return "\n".join([
        f"Spectral FWHM   {pulse.d_omega.to_frequency().value(Prefix.TERA):.3f} THz",
        f"GDD phi2        {sol.phi2.value(Prefix.PICO):.5g} ps² ({source}, both arms)",
        f"TOD phi3        {sol.phi3 / PS**3:.5g} ps³",
        *extra_lines,
        f"Chirp beta      {beta.value(Prefix.PICO):.4g} rad/ps²",
        f"  (BETA_0       {BETA_0.value(Prefix.PICO):.4g} rad/ps²)",
        f"Arm FWHM        {sol.arm1_fwhm.value(Prefix.PICO):.2f} / {sol.arm2_fwhm.value(Prefix.PICO):.2f} ps (arm 1 / 2)",
        f"Combined FWHM   {Time(sol.t_end - sol.t_start).value(Prefix.PICO):.2f} ps",
        f"Arm peak |E|    {np.abs(sol.e_1).max():.3f} / {np.abs(sol.e_2).max():.3f} (arm 1 / 2, before optics)",
        f"Combined peak   |E| {peak_e:.3f}, |E|² {peak_e**2:.3f}",
        f"Delay           {pulse.tau.value(Prefix.PICO):.3f} ps "
        f"({Length(pulse.tau * SPEED_OF_LIGHT).value(Prefix.MILLI):.3f} mm)",
        "f_rot (circular reference)",
        f"  start/centre/end  {sol.f_start.value(Prefix.GIGA):.2f} / {sol.f_centre.value(Prefix.GIGA):.2f} / "
        f"{sol.f_end.value(Prefix.GIGA):.2f} GHz",
        f"  acceleration      {100 * sol.accel:.2f} %" if sol.f_start * sol.f_end > 0
        else "  acceleration      n/a (rotation reverses within the FWHM)",
        f"Arm 1           {describe(j_1)}",
        f"Arm 2           {describe(j_2)}",
        f"Overlap |<j1|j2>| {abs(np.vdot(j_1, j_2)):.3f}",
        f"Grid            2^{s.n_points_log2}, dt {Time(pulse.grid.dt).value(Prefix.PICO) * 1e3:.1f} fs, "
        f"Nyquist ±{nyquist.value(Prefix.TERA):.1f} THz",
        f"Solve time      {sol.solve_seconds:.2f} s" if sol.solve_seconds is not None else "Solve time      - (GDD/TOD given)",
    ])


def plot_ribbon(fig: Figure, sol: Solution, s: CfgFieldsSettings) -> None:
    """The polarization ellipse of the sum along the pulse, drawn as a tube: at each time, the
    curve the field vector traces in the (Ex, Ey) plane over one optical cycle, Re[E e^{i phi}]
    for phi in [0, 2 pi). For linear light the tube collapses to a flat ribbon (the
    centrifuge ribbon of animation_optical_centrifuge.py); for circular light it is a round tube.
    Unlike the major axis alone, which is undefined at circular polarization and jumps by 90 deg
    there, this varies smoothly whatever the polarization. Coloured by the signed ellipticity, so
    the two handednesses differ. Drawn over the whole pulse, wherever the combined intensity is
    above the ribbon cutoff of its peak.
    """
    grid = sol.pulse.grid
    e_1, e_2 = sol.e_1, sol.e_2
    pol = combine(e_1, e_2, s)
    if pol.s0.max() <= 0:
        raise ValueError("Nothing is transmitted: the polarizer blocks both arms.")
    # Not renormalized: each arm peaks at |E| = 1, and this is the combination's actual field
    peak = float(np.sqrt(pol.s0.max()))

    # Strip the common optical phase (the chirp shared by both arms, which does not change the
    # ellipse) so what is left varies only at the slow beat and can be sampled coarsely.
    # Unwrapped before halving, otherwise each 2 pi wrap flips the sign of the field.
    common = np.exp(-0.5j * (np.unwrap(np.angle(e_1)) + np.unwrap(np.angle(e_2))))
    ex, ey = pol.ex * common, pol.ey * common

    keep = np.nonzero(pol.s0 > s.ribbon_cutoff * pol.s0.max())[0]
    t = np.linspace(grid.t[keep[0]], grid.t[keep[-1]], s.ribbon_points)
    t_ps = t / PS

    def at(values: np.ndarray) -> np.ndarray:
        return np.interp(t, grid.t, values.real) + 1j * np.interp(t, grid.t, values.imag)

    ex_s, ey_s = at(ex), at(ey)
    th = np.interp(t, grid.t, pol.psi)
    chi_deg = np.interp(t, grid.t, pol.chi) / AngleUnit.DEG.value

    cycle = np.exp(1j * np.linspace(0, 2 * np.pi, s.ribbon_phase_points))[:, None]
    x = np.broadcast_to(t_ps, (s.ribbon_phase_points, len(t_ps)))
    y = np.real(ex_s * cycle)
    z = np.real(ey_s * cycle)

    # linear = purple (the centrifuge), the two circular handednesses = blue / red
    cmap = LinearSegmentedColormap.from_list("ellipticity", ["#2a6cff", "#7f3fbf", "#d62728"])
    cnorm = Normalize(-45, 45)
    face = cmap(cnorm(np.broadcast_to(chi_deg, x.shape)))
    face[..., 3] = 0.6

    fig.clear()
    # Placed by hand: tight_layout and a 3D axes' colorbar do not get along. The zoomed 3D box
    # spills past its axes on the right, so its colorbar goes underneath, and the xy projection
    # sits to the right with its own (time) colorbar underneath
    ax = fig.add_axes((0.03, 0.08, 0.66, 0.85), projection="3d")
    cax = fig.add_axes((0.17, 0.07, 0.34, 0.018))
    ax_xy = fig.add_axes((0.75, 0.25, 0.19, 0.55))
    cax_xy = fig.add_axes((0.75, 0.07, 0.19, 0.018))
    plot_xy_projection(ax_xy, cax_xy, t_ps, y, z, peak)
    # rcount/ccount: plot_surface otherwise resamples to at most 50 x 50 facets
    ax.plot_surface(x, y, z, facecolors=face, rcount=x.shape[0], ccount=x.shape[1],
                    linewidth=0, antialiased=False, shade=True)
    ax.plot(t_ps, np.zeros_like(t_ps), np.zeros_like(t_ps), color="k", lw=0.5)
    # combined-FWHM edges, as rings in the field plane
    ring = np.linspace(0, 2 * np.pi, 100)
    for t_edge in (sol.t_start, sol.t_end):
        r = np.interp(t_edge, grid.t, np.sqrt(pol.s0))
        ax.plot(np.full_like(ring, t_edge.value(Prefix.PICO)), r * np.cos(ring), r * np.sin(ring),
                color="k", lw=0.8, ls="--")
    # zoom > 1 spills the box past its axes (it grows with the window height), so keep it modest
    ax.set_box_aspect((5, 1, 1), zoom=1.15)
    ax.set_xlim(t_ps[0], t_ps[-1])
    # Field axes span the combination's actual peak |E| (each arm alone peaks at 1)
    ax.set_ylim(-peak, peak)
    ax.set_zlim(-peak, peak)
    ticks = [-peak, 0.0, peak]
    ax.set_yticks(ticks, [f"{v:.2f}" for v in ticks])
    ax.set_zticks(ticks, [f"{v:.2f}" for v in ticks])
    ax.set_xlabel("Time (ps)", labelpad=12)
    ax.set_ylabel("$E_x$")
    ax.set_zlabel("$E_y$")
    turns = abs(th[-1] - th[0]) / (2 * np.pi)
    ax.set_title(
        f"Polarization ellipse along the pulse ($\\tau$ = {sol.pulse.tau.value(Prefix.PICO):.2f} ps, "
        f"{turns:.1f} turns shown)\n{polarization_label(s)}; dashed rings: combined FWHM\n"
        f"peak |E| of the combination = {peak:.3f} (an arm with only the shared dispersion alone: 1)"
    )
    ax.view_init(elev=20, azim=-70)
    fig.colorbar(ScalarMappable(norm=cnorm, cmap=cmap), cax=cax, orientation="horizontal",
                 label=r"ellipticity $\chi$ (deg): 0 linear, $\pm$45 circular")


def plot_xy_projection(ax, cax, t_ps: np.ndarray, x_field: np.ndarray, y_field: np.ndarray, peak: float) -> None:
    """The tube seen end-on, down the time axis: every sampled polarization ellipse (columns of
    x_field / y_field, one optical cycle each) drawn in the Ex-Ey plane, coloured by time. For a
    centrifuge, whose light is linear, each is a line through the origin and together they form a
    rosette that turns as time goes on."""
    segments = [np.column_stack((x_field[:, k], y_field[:, k])) for k in range(len(t_ps))]
    tnorm = Normalize(t_ps[0], t_ps[-1])
    lines = LineCollection(segments, cmap="viridis", norm=tnorm, linewidths=0.6, alpha=0.7)
    lines.set_array(t_ps)
    ax.add_collection(lines)
    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(peak * np.cos(ring), peak * np.sin(ring), color="k", lw=0.6, ls=":")
    lim = 1.05 * peak
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$E_x$")
    ax.set_ylabel("$E_y$")
    # E_y on the right: the 3D plot's own field axis sits just to the left of this panel
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_title(f"Projection onto the xy plane\n(dotted: peak |E| = {peak:.3f})")
    ax.grid(alpha=0.3)
    fig = ax.figure
    fig.colorbar(ScalarMappable(norm=tnorm, cmap="viridis"), cax=cax, orientation="horizontal", label="Time (ps)")


# ============================================================
# GUI
# ============================================================
# Polarization tab presets: button label -> (tooltip, field values). Fields a preset leaves out
# (e.g. the polarizer angle while the polarizer is out) keep whatever they were.
POLARIZATION_PRESETS: dict[str, tuple[str, dict]] = {
    "Normal 3D Centrifuge": (
        "Regular centrifuge: arms linear at 0° and 90°, QWP at 45° (opposite circular), no polarizer.",
        dict(arm1_pol=Angle(0.0, AngleUnit.DEG), pol_offset=Angle(90.0, AngleUnit.DEG),
             qwp_in=True, qwp_angle=Angle(45.0, AngleUnit.DEG), polarizer_in=False),
    ),
    "2D centrifuge": (
        "Regular centrifuge followed by a linear polarizer at 0°: linear output whose intensity beats.",
        dict(arm1_pol=Angle(0.0, AngleUnit.DEG), pol_offset=Angle(90.0, AngleUnit.DEG),
             qwp_in=True, qwp_angle=Angle(45.0, AngleUnit.DEG),
             polarizer_in=True, polarizer_angle=Angle(0.0, AngleUnit.DEG)),
    ),
    "Linear 45°": (
        "Arms linear 45° apart (-22.5° and +22.5°), no QWP, no polarizer.",
        dict(arm1_pol=Angle(-22.5, AngleUnit.DEG), pol_offset=Angle(45.0, AngleUnit.DEG),
             qwp_in=False, polarizer_in=False),
    ),
}


class CfgFieldsForm(DirtyFormWidget):
    _specs = {
        "wavelength": LengthSpec("Central wavelength", Prefix.NANO, min=100.0, max=10_000.0),
        "bandwidth": LengthSpec("Bandwidth (FWHM)", Prefix.NANO, min=0.01, max=1000.0),
        "dispersion_given": BoolSpec("Give GDD/TOD directly"),
        "duration_fwhm": TimeSpec("Duration (FWHM)", Prefix.PICO, min=0.01, max=100_000.0),
        "acceleration_percent": FloatSpec("Acceleration over FWHM due to TOD",min=-40.0, max=40.0, decimals=2, step=0.5, suffix="%"),
        "gdd": GDDSpec("GDD", Prefix.PICO),
        "tod_ps3": FloatSpec("TOD", min=-1e4, max=1e4, decimals=6, step=0.001, suffix="ps³"),
        "extra_dispersion_in": BoolSpec("Add to one arm"),
        "extra_arm": EnumSpec("Arm", Arm),
        "extra_gdd": GDDSpec("Extra GDD", Prefix.PICO),
        "extra_tod_ps3": FloatSpec("Extra TOD", min=-1e4, max=1e4, decimals=6, step=0.001, suffix="ps³"),
        "delay": TimeSpec("Arm 2 delay", Prefix.PICO, min=0.0001, max=10_000.0),
        "relative_phase": AngleSpec("Arm 2 extra phase"),
        "arm1_pol": AngleSpec("Arm 1 angle"),
        "pol_offset": AngleSpec("Arm 2 offset from arm 1"),
        "qwp_in": BoolSpec("QWP in"),
        "qwp_angle": AngleSpec("QWP fast axis"),
        "polarizer_in": BoolSpec("Polarizer in"),
        "polarizer_angle": AngleSpec("Polarizer axis"),
        "n_points_log2": IntSpec("Grid points (log2)", min=10, max=22),
        "window": TimeSpec("Time window", Prefix.PICO, min=1.0, max=1_000_000.0),
        "ribbon_points": IntSpec("Samples along pulse", min=10, max=20_000, step=50),
        "ribbon_phase_points": IntSpec("Samples around ellipse", min=3, max=200),
        "ribbon_cutoff": FloatSpec("Intensity cutoff", min=1e-6, max=0.5, decimals=4, step=0.005),
    }
    _groups = [
        ("Pulse", ["wavelength", "bandwidth"]),
        ("Dispersion (both arms)", ["dispersion_given", "duration_fwhm", "acceleration_percent", "gdd", "tod_ps3"]),
        ("Extra dispersion (one arm)", ["extra_dispersion_in", "extra_arm", "extra_gdd", "extra_tod_ps3"]),
        ("Arms", ["delay", "relative_phase"]),
        ("Linear polarization of the arms", ["arm1_pol", "pol_offset"]),
        ("Quarter-wave plate (after the combiner)", ["qwp_in", "qwp_angle"]),
        ("Polarizer (after the QWP)", ["polarizer_in", "polarizer_angle"]),
        ("Grid", ["n_points_log2", "window"]),
        ("3D plot", ["ribbon_points", "ribbon_phase_points", "ribbon_cutoff"]),
    ]
    # Tabs group the sections above: (tab title, group titles shown on that tab)
    _tabs = [
        ("Pulse Parameters", ["Pulse", "Dispersion (both arms)", "Extra dispersion (one arm)", "Arms"]),
        ("Polarization", ["Linear polarization of the arms", "Quarter-wave plate (after the combiner)",
                          "Polarizer (after the QWP)"]),
        ("Computation", ["Grid", "3D plot"]),
    ]

    def _build(self) -> None:
        super()._build()
        # super()._build() added one QGroupBox per _groups entry, in order, directly to
        # form_layout; take them back out and sort them into tabs instead.
        containers = {}
        for title, _ in self._groups:
            container = self.form_layout.takeAt(0).widget()
            container.setStyleSheet("QGroupBox { background-color: #e8e8e8; }")
            containers[title] = container

        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { color: red; }")
        for tab_title, titles in self._tabs:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            if tab_title == "Polarization":
                page_layout.addWidget(self._build_presets())
            for title in titles:
                page_layout.addWidget(containers[title])
            page_layout.addStretch(1)
            tabs.addTab(page, tab_title)
        self.form_layout.addWidget(tabs)

    def _build_presets(self) -> QGroupBox:
        box = QGroupBox("Presets")
        box.setStyleSheet("QGroupBox { background-color: #e8e8e8; }")
        row = QHBoxLayout(box)
        for label, (tooltip, values) in POLARIZATION_PRESETS.items():
            button = QPushButton(label)
            button.setToolTip(tooltip + "\nFills in the fields below; press Apply to compute.")
            button.clicked.connect(lambda _=False, v=values: self._load_preset(v))
            row.addWidget(button)
        return box

    def _load_preset(self, values: dict) -> None:
        # Fills the fields like typing into them would: they show as edited (dirty) until Apply
        for name, value in values.items():
            self._specs[name].set_value(self._widgets[name], value)
            self._set_dirty(name)

    # Checkbox -> (fields enabled while it is checked, fields enabled while it is not)
    _toggles = {
        "dispersion_given": (["gdd", "tod_ps3"], ["duration_fwhm", "acceleration_percent"]),
        "extra_dispersion_in": (["extra_arm", "extra_gdd", "extra_tod_ps3"], []),
        "qwp_in": (["qwp_angle"], []),
        "polarizer_in": (["polarizer_angle"], []),
    }

    def __init__(self, settings: CfgFieldsSettings, on_apply, parent: QWidget | None = None) -> None:
        super().__init__(settings, parent)
        self._on_apply = on_apply
        # Grey out the fields a checkbox makes irrelevant; follows the checkbox as edited (before
        # Apply), and when a loaded config sets it
        for name in self._toggles:
            self._widgets[name].checkStateChanged.connect(lambda _, n=name: self._update_enabled(n))
            self._update_enabled(name)

    def _update_enabled(self, toggle: str) -> None:
        checked = self._widgets[toggle].isChecked()
        when_on, when_off = self._toggles[toggle]
        for name in when_on:
            self._widgets[name].setEnabled(checked)
        for name in when_off:
            self._widgets[name].setEnabled(not checked)

    def on_apply(self) -> None:
        self._on_apply()


class CfgFieldsViewModel(PanelViewModel):
    def __init__(self, bus: EventBus, dispatcher: QtDispatcher, settings: CfgFieldsSettings) -> None:
        super().__init__(bus, dispatcher)
        self.settings = settings
        self._solved: tuple[GDD, float, float] | None = None   # (phi2, phi3, seconds)
        self._solved_key: tuple | None = None
        self._warned_grid_key: tuple | None = None

    def solution(self) -> Solution:
        s = self.settings
        pulse = Pulse(s)
        self._warn_if_undersampled(pulse)
        if s.dispersion_given:
            return build_solution(s, pulse, s.gdd, s.tod_ps3 * PS**3, None)
        # The solve takes seconds; polarization, extra-dispersion and plot settings do not affect
        # it, so it is only redone when one of SOLVE_FIELDS changed
        key = tuple(float(getattr(s, name)) for name in SOLVE_FIELDS)
        if key != self._solved_key:
            self._solved = solve_dispersion(s, pulse)
            self._solved_key = key
            self._msg(f"Solved GDD/TOD in {self._solved[2]:.1f} s")
        return build_solution(s, pulse, *self._solved)

    def _warn_if_undersampled(self, pulse: Pulse) -> None:
        # Once per grid/bandwidth combination, not on every Apply
        key = (self.settings.n_points_log2, float(self.settings.window), float(self.settings.bandwidth))
        if key == self._warned_grid_key:
            return
        self._warned_grid_key = key
        nyquist = np.pi / pulse.grid.dt   # rad/s
        edge = float(pulse.spectral_amplitude(np.array([nyquist]))[0])
        if edge > 1e-3:
            self._msg(f"Grid too coarse: spectrum is still at {edge:.1e} of its peak at the Nyquist "
                      "frequency; raise the grid points or shorten the window.", MessageLevel.WARNING)

    # ============================================================
    # Sidecar config
    # ============================================================
    def save_sidecar_config(self) -> None:
        save_settings_to_sidecar(self.settings)
        self._msg(f"Saved config to: {SIDECAR_CONFIG_PATH}")

    def load_sidecar_config(self) -> frozenset[str]:
        loaded = load_settings_from_sidecar()
        names = [f.name for f in dataclass_fields(CfgFieldsSettings)]
        for name in names:
            setattr(self.settings, name, getattr(loaded, name))
        self._msg(f"Loaded config from: {SIDECAR_CONFIG_PATH}")
        return frozenset(names)

    def save(self, fig: Figure) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "CFG_fields_ribbon.png"
        fig.savefig(out_path, dpi=150)
        self._msg(f"Saved plot to: {out_path}")

    def report_error(self, e: Exception) -> None:
        self._msg(f"Error: {e}", MessageLevel.ERROR)


class CfgFieldsPanel(Panel):
    def __init__(self, vm: CfgFieldsViewModel, parent: QWidget | None = None) -> None:
        self._window_sized = False
        super().__init__("CFG fields visualization", vm, parent)

    def setup(self) -> None:
        self._fig = Figure(figsize=(15, 6))
        self._canvas = FigureCanvasQTAgg(self._fig)

        plot_col = QVBoxLayout()
        plot_col.addWidget(NavigationToolbar2QT(self._canvas, self))
        plot_col.addWidget(self._canvas, stretch=1)

        # Edits stay pending (dirty) until Apply, which writes them to the settings and replots
        self._form = CfgFieldsForm(self.vm.settings, self._replot)
        save_config_button = QPushButton("Save config")
        save_config_button.clicked.connect(self._on_save_config)
        load_config_button = QPushButton("Load config")
        load_config_button.clicked.connect(self._on_load_config)
        apply_row = QHBoxLayout()
        apply_row.addStretch(1)
        apply_row.addWidget(save_config_button)
        apply_row.addWidget(load_config_button)
        apply_row.addWidget(self._form.apply_button)

        self._results = QPlainTextEdit()
        self._results.setReadOnly(True)
        self._results.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._results.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))

        self._side_panel = QWidget()
        side_col = QVBoxLayout(self._side_panel)
        side_col.setContentsMargins(0, 0, 0, 0)
        side_col.addWidget(self._form)
        side_col.addLayout(apply_row)
        side_col.addWidget(QLabel("Results"))
        side_col.addWidget(self._results, stretch=1)
        # Fixed, because the spin boxes' huge ranges make their size hints far too wide; widened
        # to fit the results text in _fit_results
        self._side_panel.setFixedWidth(SIDE_PANEL_MIN_WIDTH)

        plot_row = QHBoxLayout()
        plot_row.addLayout(plot_col, stretch=1)
        plot_row.addWidget(self._side_panel)

        save_button = QPushButton("Save PNG")
        save_button.clicked.connect(lambda: self.vm.save(self._fig))
        bottom_row = QHBoxLayout()
        bottom_row.addStretch(1)
        bottom_row.addWidget(save_button)

        self.body_layout.addLayout(plot_row, stretch=1)
        self.body_layout.addLayout(bottom_row)
        self._replot()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # The app stylesheet changes the results font (and resets minimum sizes) only once the
        # panel is shown, so measure the text again now, then grow the window to fit it all
        if self._window_sized:
            return
        self._window_sized = True
        self._fit_results()
        window = self.window()
        screen = window.screen().availableGeometry().size()
        window.resize(window.sizeHint().expandedTo(QSize(*MIN_WINDOW_SIZE)).boundedTo(screen))

    def _fit_results(self) -> None:
        # Size the results box to its text, so every line shows without scrolling, and widen
        # the side panel to match (never narrower than the form needs)
        self._results.ensurePolished()
        lines = self._results.toPlainText().splitlines() or [""]
        metrics = self._results.fontMetrics()
        pad = 2 * (self._results.frameWidth() + int(self._results.document().documentMargin())) + 8
        width = max(metrics.horizontalAdvance(line) for line in lines) + pad
        height = metrics.lineSpacing() * len(lines) + pad
        self._results.setMinimumSize(width, height)
        self._side_panel.setFixedWidth(max(SIDE_PANEL_MIN_WIDTH, width))

    def _show_error(self, e: Exception) -> None:
        self._fig.clear()
        self._fig.text(0.5, 0.5, f"Error: {e}", color="r", ha="center", va="center", wrap=True)
        self._canvas.draw_idle()
        self._results.setPlainText(f"Error: {e}")
        self.vm.report_error(e)

    def _replot(self) -> None:
        settings = self.vm.settings
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            sol = self.vm.solution()
            plot_ribbon(self._fig, sol, settings)
            self._results.setPlainText(summary(sol, settings))
            self._fit_results()
        except Exception as e:
            self._show_error(e)
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._canvas.draw_idle()

    def _on_save_config(self) -> None:
        try:
            self.vm.save_sidecar_config()
        except Exception as e:
            self._show_error(e)

    def _on_load_config(self) -> None:
        try:
            names = self.vm.load_sidecar_config()
        except Exception as e:
            self._show_error(e)
            return
        self._form.refresh_fields(names)
        self._replot()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    install_ui(app)
    bus = EventBus()
    dispatcher = QtDispatcher()

    window = LabMainWindow("CFG fields visualization", bus, dispatcher)
    vm = CfgFieldsViewModel(bus, dispatcher, CfgFieldsSettings())
    window.setCentralWidget(CfgFieldsPanel(vm))
    # Grown further once shown, to fit the results text (see CfgFieldsPanel.showEvent)
    window.resize(*MIN_WINDOW_SIZE)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
