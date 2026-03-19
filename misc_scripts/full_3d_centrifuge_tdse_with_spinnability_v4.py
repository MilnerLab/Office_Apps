from __future__ import annotations

"""
Full 3D rigid-rotor TDSE for an optical centrifuge in the rotating frame.

Model
-----
    H_rot(t) = h [ B J(J+1) - D J^2(J+1)^2 - nu_tdse(t) M ]
               - U0(t) * sin^2(theta) cos^2(phi)

with basis |J, M>.

Conventions used here
---------------------
1. The user-facing centrifuge frequency is the real cfg frequency.
2. The TDSE uses half of that frequency internally:
       nu_tdse(t) = 0.5 * nu_cfg_real(t)
3. The linear chirp spans one pulse FWHM centered on pulse_center_ps.
4. Spinnability is plotted over the full time axis. Beyond the centrifugal wall,
   the effective inertia is clamped to its wall value so the curve remains defined.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply
from sympy.physics.wigner import wigner_3j


# ============================================================
# constants
# ============================================================
H = 6.62607015e-34           # J s
HBAR = 1.054571817e-34       # J s
C = 299792458.0              # m / s
EPS0 = 8.8541878128e-12      # F / m
PI = np.pi
BOLTZMANN_J_K = 1.380649e-23


# ============================================================
# user-facing data classes
# ============================================================
@dataclass(frozen=True)
class Molecule:
    name: str
    B_Hz: float
    D_Hz: float
    delta_alpha_A3: float


@dataclass(frozen=True)
class Pulse:
    intensity_peak_W_cm2: float
    fwhm_ps: float
    center_ps: float
    cfg_freq_start_GHz: float
    cfg_freq_end_GHz: float


@dataclass(frozen=True)
class Simulation:
    J_max: int
    J0: int
    M0: int
    t_min_ps: float
    t_max_ps: float
    n_time: int
    log_heatmaps: bool = True
    temperature_K: float | None = None


# ============================================================
# molecules from your screenshots (droplet values)
# ============================================================
CS2 = Molecule(
    name="CS2",
    B_Hz=730e6,
    D_Hz=1.2e6,
    delta_alpha_A3=8.7,
)

OCS = Molecule(
    name="OCS",
    B_Hz=2.18e9,
    D_Hz=9.5e6,
    delta_alpha_A3=3.7,
)


# ============================================================
# basis helpers
# ============================================================
def build_basis(J_max: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    states: list[tuple[int, int]] = []
    index: dict[tuple[int, int], int] = {}

    k = 0
    for J in range(J_max + 1):
        for M in range(-J, J + 1):
            states.append((J, M))
            index[(J, M)] = k
            k += 1

    return states, index


# ============================================================
# unit conversions and pulse functions
# ============================================================
def ps_to_s(x_ps: float) -> float:
    return x_ps * 1e-12


def GHz_to_Hz(x_GHz: float) -> float:
    return x_GHz * 1e9


def intensity_W_cm2_to_W_m2(x: float) -> float:
    return x * 1e4


def delta_alpha_A3_to_SI(alpha_A3: float) -> float:
    alpha_m3 = alpha_A3 * 1e-30
    return 4.0 * PI * EPS0 * alpha_m3


def gaussian_intensity_W_m2(t_s: float, pulse: Pulse) -> float:
    t0 = ps_to_s(pulse.center_ps)
    fwhm = ps_to_s(pulse.fwhm_ps)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    I0 = intensity_W_cm2_to_W_m2(pulse.intensity_peak_W_cm2)
    x = (t_s - t0) / sigma
    return I0 * np.exp(-0.5 * x * x)


def chirp_time_window_s(pulse: Pulse) -> tuple[float, float]:
    half_width = 0.5 * ps_to_s(pulse.fwhm_ps)
    center = ps_to_s(pulse.center_ps)
    return center - half_width, center + half_width


def cfg_real_frequency_Hz(t_s: float, pulse: Pulse) -> float:
    t1, t2 = chirp_time_window_s(pulse)
    nu1 = GHz_to_Hz(pulse.cfg_freq_start_GHz)
    nu2 = GHz_to_Hz(pulse.cfg_freq_end_GHz)

    if t_s <= t1:
        return nu1
    if t_s >= t2:
        return nu2

    frac = (t_s - t1) / (t2 - t1)
    return nu1 + frac * (nu2 - nu1)


def tdse_frequency_Hz(t_s: float, pulse: Pulse) -> float:
    return 0.5 * cfg_real_frequency_Hz(t_s, pulse)


def cfg_angular_acceleration_rad_s2(pulse: Pulse) -> float:
    t1, t2 = chirp_time_window_s(pulse)
    f1 = GHz_to_Hz(pulse.cfg_freq_start_GHz)
    f2 = GHz_to_Hz(pulse.cfg_freq_end_GHz)
    return 2.0 * PI * (f2 - f1) / (t2 - t1)


def u0_J(t_s: float, molecule: Molecule, pulse: Pulse) -> float:
    delta_alpha_si = delta_alpha_A3_to_SI(molecule.delta_alpha_A3)
    intensity = gaussian_intensity_W_m2(t_s, pulse)
    return delta_alpha_si * intensity / (2.0 * C * EPS0)


# ============================================================
# angular matrix elements
# Operator: O = sin^2(theta) cos^2(phi)
# ============================================================
@lru_cache(maxsize=None)
def w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def spherical_harmonic_matrix_element(Jp: int, Mp: int, J: int, M: int, q: int) -> float:
    pref = (-1) ** Mp * np.sqrt((2 * J + 1) * 5 * (2 * Jp + 1) / (4.0 * PI))
    return pref * w3j(J, 2, Jp, 0, 0, 0) * w3j(J, 2, Jp, M, q, -Mp)


def build_in_plane_operator(J_max: int) -> tuple[csr_matrix, list[tuple[int, int]], np.ndarray, np.ndarray]:
    states, index = build_basis(J_max)
    n = len(states)

    J_vals = np.array([J for J, _ in states], dtype=int)
    M_vals = np.array([M for _, M in states], dtype=int)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    c_const = 1.0 / 3.0
    c_q0 = -(1.0 / 3.0) * np.sqrt(4.0 * PI / 5.0)
    c_q2 = np.sqrt(2.0 * PI / 15.0)

    for bra_idx, (Jp, Mp) in enumerate(states):
        rows.append(bra_idx)
        cols.append(bra_idx)
        data.append(c_const)

        for q, coeff in ((0, c_q0), (2, c_q2), (-2, c_q2)):
            M = Mp - q
            if abs(M) > J_max:
                continue

            for J in range(max(0, Jp - 2), min(J_max, Jp + 2) + 1):
                if abs(M) > J:
                    continue
                col_idx = index.get((J, M))
                if col_idx is None:
                    continue
                me = spherical_harmonic_matrix_element(Jp, Mp, J, M, q)
                if me != 0.0:
                    rows.append(bra_idx)
                    cols.append(col_idx)
                    data.append(coeff * me)

    O_op = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    return O_op, states, J_vals, M_vals


# ============================================================
# Hamiltonian pieces
# ============================================================
def rigid_rotor_energy_J(J: float, molecule: Molecule) -> float:
    jj = J * (J + 1.0)
    return H * (molecule.B_Hz * jj - molecule.D_Hz * jj * jj)


def free_rotor_energies_J(states: Iterable[tuple[int, int]], molecule: Molecule) -> np.ndarray:
    diag = []
    for J, _ in states:
        diag.append(rigid_rotor_energy_J(J, molecule))
    return np.array(diag, dtype=np.float64)


def build_hamiltonian_midpoint(
    t_s: float,
    free_diag_J: np.ndarray,
    M_vals: np.ndarray,
    O_op: csr_matrix,
    molecule: Molecule,
    pulse: Pulse,
) -> csr_matrix:
    nu_tdse = tdse_frequency_Hz(t_s, pulse)
    U0 = u0_J(t_s, molecule, pulse)

    diag = free_diag_J - H * nu_tdse * M_vals
    return diags(diag, offsets=0, format="csr") - U0 * O_op


# ============================================================
# spinnability diagnostics
# ============================================================
def transition_frequency_Hz(J: float, molecule: Molecule) -> float:
    return (rigid_rotor_energy_J(J + 2.0, molecule) - rigid_rotor_energy_J(J, molecule)) / (2.0 * H)


def transition_frequency_derivative_Hz_per_J(J: float, molecule: Molecule) -> float:
    B = molecule.B_Hz
    D = molecule.D_Hz
    return 2.0 * B - 12.0 * D * J * J - 36.0 * D * J - 30.0 * D


def wall_J(molecule: Molecule) -> float:
    B = molecule.B_Hz
    D = molecule.D_Hz
    if D <= 0.0:
        return np.inf

    disc = 36.0**2 - 4.0 * 12.0 * (30.0 - 2.0 * B / D)
    if disc < 0.0:
        return np.inf

    root = (-36.0 + np.sqrt(disc)) / 24.0
    return root if root >= 0.0 else np.inf


def wall_tdse_frequency_Hz(molecule: Molecule) -> float:
    jw = wall_J(molecule)
    if not np.isfinite(jw):
        return np.inf
    return transition_frequency_Hz(jw, molecule)


def resonant_J_from_tdse_frequency(nu_tdse_Hz: float, molecule: Molecule) -> float:
    if nu_tdse_Hz < 0.0:
        raise ValueError("Requested TDSE frequency must be >= 0.")

    nu_wall = wall_tdse_frequency_Hz(molecule)
    if not np.isfinite(nu_wall):
        raise ValueError("No finite wall found for the current molecule model.")
    if nu_tdse_Hz >= nu_wall:
        raise ValueError("The requested TDSE frequency is at or beyond the centrifugal wall.")

    jw = wall_J(molecule)

    def f(J: float) -> float:
        return transition_frequency_Hz(J, molecule) - nu_tdse_Hz

    return brentq(f, 0.0, jw)


def wall_effective_B_Hz(molecule: Molecule) -> float:
    jw = wall_J(molecule)
    nu_wall = wall_tdse_frequency_Hz(molecule)
    return nu_wall / (2.0 * jw + 3.0)


def local_effective_B_Hz_from_tdse_frequency(nu_tdse_Hz: float, molecule: Molecule) -> float:
    J_res = resonant_J_from_tdse_frequency(nu_tdse_Hz, molecule)
    return nu_tdse_Hz / (2.0 * J_res + 3.0)


def local_effective_B_Hz_fullrange(nu_tdse_Hz: float, molecule: Molecule) -> float:
    """
    Full-range version used for plotting sigma over the whole time axis.
    Below the wall it uses the resonant rising-branch J.
    At and beyond the wall it clamps to the wall value.
    """
    nu_wall = wall_tdse_frequency_Hz(molecule)
    if not np.isfinite(nu_wall):
        return max(local_effective_B_Hz_from_tdse_frequency(nu_tdse_Hz, molecule), 0.0)

    if nu_tdse_Hz < nu_wall:
        return local_effective_B_Hz_from_tdse_frequency(nu_tdse_Hz, molecule)
    return wall_effective_B_Hz(molecule)


def spinnability(
    t_s: float,
    molecule: Molecule,
    pulse: Pulse,
    temperature_K: float | None = None,
) -> float:
    nu_tdse = tdse_frequency_Hz(t_s, pulse)
    eff_inverse_I = local_effective_B_Hz_fullrange(nu_tdse, molecule)

    if eff_inverse_I <= 0.0:
        raise ValueError("B_eff_hz became <= 0.")

    I_eff_kg_m2 = H / (8.0 * PI**2 * eff_inverse_I)
    U0 = u0_J(t_s, molecule, pulse)
    alpha_cfg = cfg_angular_acceleration_rad_s2(pulse)

    sigma0 = 2.0 * U0 / (PI * I_eff_kg_m2 * alpha_cfg)

    if temperature_K is None:
        return sigma0

    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be > 0.")

    sigmaT = sigma0 * U0 / (0.5 * BOLTZMANN_J_K * temperature_K)
    return min(sigma0, sigmaT)


def spinnability_array(
    t_s: np.ndarray,
    molecule: Molecule,
    pulse: Pulse,
    temperature_K: float | None = None,
) -> np.ndarray:
    vals = np.full_like(t_s, np.nan, dtype=float)
    for i, t in enumerate(t_s):
        try:
            vals[i] = spinnability(t, molecule, pulse, temperature_K=temperature_K)
        except ValueError:
            vals[i] = np.nan
    return vals


# ============================================================
# propagation and observables
# ============================================================
@dataclass
class Result:
    t_s: np.ndarray
    J_values: np.ndarray
    M_values: np.ndarray
    P_J_t: np.ndarray
    P_M_t: np.ndarray
    final_psi: np.ndarray
    states: list[tuple[int, int]]


def initial_state_vector(states: list[tuple[int, int]], J0: int, M0: int) -> np.ndarray:
    psi = np.zeros(len(states), dtype=np.complex128)
    try:
        idx = states.index((J0, M0))
    except ValueError as exc:
        raise ValueError(f"Initial state |J0={J0}, M0={M0}> is outside the chosen basis.") from exc
    psi[idx] = 1.0 + 0.0j
    return psi


def accumulate_PJ_PM(prob_state: np.ndarray, J_vals: np.ndarray, M_vals: np.ndarray, J_max: int) -> tuple[np.ndarray, np.ndarray]:
    P_J = np.zeros(J_max + 1, dtype=np.float64)
    P_M = np.zeros(2 * J_max + 1, dtype=np.float64)
    np.add.at(P_J, J_vals, prob_state)
    np.add.at(P_M, M_vals + J_max, prob_state)
    return P_J, P_M


def propagate_tdse(molecule: Molecule, pulse: Pulse, sim: Simulation) -> Result:
    O_op, states, J_vals, M_vals = build_in_plane_operator(sim.J_max)
    free_diag_J = free_rotor_energies_J(states, molecule)

    t_s = np.linspace(ps_to_s(sim.t_min_ps), ps_to_s(sim.t_max_ps), sim.n_time)
    psi = initial_state_vector(states, sim.J0, sim.M0)

    P_J_t = np.zeros((sim.n_time, sim.J_max + 1), dtype=np.float64)
    P_M_t = np.zeros((sim.n_time, 2 * sim.J_max + 1), dtype=np.float64)

    prob = np.abs(psi) ** 2
    P_J_t[0], P_M_t[0] = accumulate_PJ_PM(prob, J_vals, M_vals, sim.J_max)

    for k in range(sim.n_time - 1):
        t_mid = 0.5 * (t_s[k] + t_s[k + 1])
        dt = t_s[k + 1] - t_s[k]
        H_mid = build_hamiltonian_midpoint(t_mid, free_diag_J, M_vals, O_op, molecule, pulse)
        A = (-1j * dt / HBAR) * H_mid
        psi = expm_multiply(A, psi)
        psi /= np.linalg.norm(psi)

        prob = np.abs(psi) ** 2
        P_J_t[k + 1], P_M_t[k + 1] = accumulate_PJ_PM(prob, J_vals, M_vals, sim.J_max)

    return Result(
        t_s=t_s,
        J_values=np.arange(sim.J_max + 1, dtype=int),
        M_values=np.arange(-sim.J_max, sim.J_max + 1, dtype=int),
        P_J_t=P_J_t,
        P_M_t=P_M_t,
        final_psi=psi,
        states=states,
    )


# ============================================================
# plotting
# ============================================================
def _heatmap_norm(use_log: bool):
    return LogNorm(vmin=1e-10, vmax=1.0) if use_log else None


def plot_main_comparison(
    pulse: Pulse,
    result_cs2: Result,
    sim_cs2: Simulation,
    result_ocs: Result,
    sim_ocs: Simulation,
) -> None:
    t_ps = result_cs2.t_s * 1e12
    intensity_W_cm2 = np.array([gaussian_intensity_W_m2(t, pulse) for t in result_cs2.t_s]) / 1e4
    nu_cfg_GHz = np.array([cfg_real_frequency_Hz(t, pulse) for t in result_cs2.t_s]) / 1e9
    nu_tdse_GHz = np.array([tdse_frequency_Hz(t, pulse) for t in result_cs2.t_s]) / 1e9

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 4, 4], hspace=0.25)

    ax_I = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0], sharex=ax_I)
    ax_cs2 = fig.add_subplot(gs[2, 0], sharex=ax_I)
    ax_ocs = fig.add_subplot(gs[3, 0], sharex=ax_I)

    ax_I.plot(t_ps, intensity_W_cm2)
    ax_I.set_ylabel("I (W/cm²)")
    ax_I.set_title("Optical centrifuge pulse and 3D TDSE J-populations")

    ax_f.plot(t_ps, nu_cfg_GHz, label="real cfg freq")
    ax_f.plot(t_ps, nu_tdse_GHz, label="TDSE freq")
    ax_f.set_ylabel("f (GHz)")
    ax_f.legend(loc="best")

    im_cs2 = ax_cs2.imshow(
        result_cs2.P_J_t.T,
        origin="lower",
        aspect="auto",
        extent=[t_ps[0], t_ps[-1], result_cs2.J_values[0] - 0.5, result_cs2.J_values[-1] + 0.5],
        interpolation="nearest",
        norm=_heatmap_norm(sim_cs2.log_heatmaps),
    )
    ax_cs2.set_ylabel("J")
    ax_cs2.set_title("CS2: P_J(t)")
    cbar_cs2 = fig.colorbar(im_cs2, ax=ax_cs2)
    cbar_cs2.set_label("population")

    im_ocs = ax_ocs.imshow(
        result_ocs.P_J_t.T,
        origin="lower",
        aspect="auto",
        extent=[t_ps[0], t_ps[-1], result_ocs.J_values[0] - 0.5, result_ocs.J_values[-1] + 0.5],
        interpolation="nearest",
        norm=_heatmap_norm(sim_ocs.log_heatmaps),
    )
    ax_ocs.set_xlabel("time (ps)")
    ax_ocs.set_ylabel("J")
    ax_ocs.set_title("OCS: P_J(t)")
    cbar_ocs = fig.colorbar(im_ocs, ax=ax_ocs)
    cbar_ocs.set_label("population")

    plt.show()


def plot_spinnability_comparison_fullrange(
    pulse: Pulse,
    temperature_K: float | None = None,
    n_time: int = 1400,
) -> None:
    t_min_ps = pulse.center_ps - 1.5 * pulse.fwhm_ps
    t_max_ps = pulse.center_ps + 1.5 * pulse.fwhm_ps
    t_s = np.linspace(ps_to_s(t_min_ps), ps_to_s(t_max_ps), n_time)
    t_ps = t_s * 1e12

    sigma_cs2 = spinnability_array(t_s, CS2, pulse, temperature_K=temperature_K)
    sigma_ocs = spinnability_array(t_s, OCS, pulse, temperature_K=temperature_K)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    suffix = "" if temperature_K is None else f" at {temperature_K:g} K"
    ax.plot(t_ps, sigma_cs2, label=f"CS2{suffix}")
    ax.plot(t_ps, sigma_ocs, label=f"OCS{suffix}")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("spinnability")
    ax.set_title("Spinnability comparison over full pulse window")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.show()


# ============================================================
# main
# ============================================================
def main() -> None:
    pulse = Pulse(
        intensity_peak_W_cm2=1.0e12,
        fwhm_ps=600.0,
        center_ps=0.0,
        cfg_freq_start_GHz=0.0,
        cfg_freq_end_GHz=50.0,
    )

    sim_cs2 = Simulation(
        J_max=30,
        J0=0,
        M0=0,
        t_min_ps=-900.0,
        t_max_ps=900.0,
        n_time=1200,
        log_heatmaps=True,
        temperature_K=None,
    )

    result_cs2 = propagate_tdse(CS2, pulse, sim_cs2)
    result_ocs = propagate_tdse(OCS, pulse, sim_cs2)

    plot_main_comparison(pulse, result_cs2, sim_cs2, result_ocs, sim_cs2)
    plot_spinnability_comparison_fullrange(pulse, temperature_K=None)


if __name__ == "__main__":
    main()
