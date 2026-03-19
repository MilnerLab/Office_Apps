from __future__ import annotations

"""
Full 3D rigid-rotor TDSE for an optical centrifuge in the rotating frame.

Model:
    H_rot(t) = h [ B J(J+1) - D J^2(J+1)^2 - nu(t) M ] - U0(t) * sin^2(theta) cos^2(phi)

with basis |J, M> and the centrifuge geometry fully retained.

Notes
-----
- This is the full 3D model in the rotating frame, not the planar rotor model.
- The angular operator sin^2(theta) cos^2(phi) is built analytically using spherical-harmonic
  matrix elements and Wigner 3j symbols.
- Time propagation uses midpoint exponential propagation with scipy.sparse.linalg.expm_multiply.
- For strong chirps and small B (for example CS2 in droplets), a converged calculation can require
  rather large J_max and may become computationally heavy. Start modestly and increase J_max.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
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
    freq_start_GHz: float
    freq_end_GHz: float
    chirp_start_ps: float
    chirp_end_ps: float


@dataclass(frozen=True)
class Simulation:
    J_max: int
    J0: int
    M0: int
    t_min_ps: float
    t_max_ps: float
    n_time: int
    log_heatmaps: bool = True


# ============================================================
# example molecules from your screenshots (droplet values)
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
    """
    Convert polarizability volume in A^3 to SI polarizability.

    alpha_SI = 4*pi*eps0 * alpha_volume, with alpha_volume in m^3.
    """
    alpha_m3 = alpha_A3 * 1e-30
    return 4.0 * PI * EPS0 * alpha_m3


def gaussian_intensity_W_m2(t_s: float, pulse: Pulse) -> float:
    t0 = ps_to_s(pulse.center_ps)
    fwhm = ps_to_s(pulse.fwhm_ps)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    I0 = intensity_W_cm2_to_W_m2(pulse.intensity_peak_W_cm2)
    x = (t_s - t0) / sigma
    return I0 * np.exp(-0.5 * x * x)


def centrifuge_frequency_Hz(t_s: float, pulse: Pulse) -> float:
    """Ordinary rotation frequency nu(t), not angular frequency Omega(t)."""
    t1 = ps_to_s(pulse.chirp_start_ps)
    t2 = ps_to_s(pulse.chirp_end_ps)

    nu1 = GHz_to_Hz(pulse.freq_start_GHz)
    nu2 = GHz_to_Hz(pulse.freq_end_GHz)

    if t_s <= t1:
        return nu1
    if t_s >= t2:
        return nu2
    frac = (t_s - t1) / (t2 - t1)
    return nu1 + frac * (nu2 - nu1)


def u0_J(t_s: float, molecule: Molecule, pulse: Pulse) -> float:
    """
    U0 = Delta_alpha * E^2 / 4 = Delta_alpha_SI * I / (2 c eps0)
    """
    delta_alpha_si = delta_alpha_A3_to_SI(molecule.delta_alpha_A3)
    intensity = gaussian_intensity_W_m2(t_s, pulse)
    return delta_alpha_si * intensity / (2.0 * C * EPS0)


# ============================================================
# angular matrix elements
# Operator: O = sin^2(theta) cos^2(phi)
# Decomposition:
#   O = 1/3 - (1/3) sqrt(4pi/5) Y_20 + sqrt(2pi/15) [Y_22 + Y_2,-2]
# ============================================================
@lru_cache(maxsize=None)
def w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def spherical_harmonic_matrix_element(Jp: int, Mp: int, J: int, M: int, q: int) -> float:
    """
    <J',M'|Y_{2q}|J,M>

    Uses
        integral Y*_{J'M'} Y_{2q} Y_{JM} dOmega
    in Condon-Shortley phase convention.
    """
    pref = (-1) ** Mp * np.sqrt((2 * J + 1) * 5 * (2 * Jp + 1) / (4.0 * PI))
    return pref * w3j(J, 2, Jp, 0, 0, 0) * w3j(J, 2, Jp, M, q, -Mp)


def build_in_plane_operator(J_max: int) -> tuple[csr_matrix, list[tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Build the matrix of O = sin^2(theta) cos^2(phi) in the |J,M> basis.

    Returns
    -------
    O_op : csr_matrix
        Sparse operator matrix.
    states : list[(J, M)]
        Basis states in the same order as the matrix.
    J_vals : ndarray[int]
        J quantum number for each basis state.
    M_vals : ndarray[int]
        M quantum number for each basis state.
    """
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
        # constant term
        rows.append(bra_idx)
        cols.append(bra_idx)
        data.append(c_const)

        # rank-2 pieces: q = 0, +/- 2
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
def free_rotor_energies_J(states: Iterable[tuple[int, int]], molecule: Molecule) -> np.ndarray:
    """
    Diagonal field-free energies in Joule:
        E_J = h [ B J(J+1) - D J^2 (J+1)^2 ]
    """
    diag = []
    for J, _ in states:
        jj = J * (J + 1)
        diag.append(H * (molecule.B_Hz * jj - molecule.D_Hz * jj * jj))
    return np.array(diag, dtype=np.float64)


def build_hamiltonian_midpoint(
    t_s: float,
    free_diag_J: np.ndarray,
    M_vals: np.ndarray,
    O_op: csr_matrix,
    molecule: Molecule,
    pulse: Pulse,
) -> csr_matrix:
    """
    H_rot(t) = H_free - h nu(t) M - U0(t) O
    """
    nu = centrifuge_frequency_Hz(t_s, pulse)
    U0 = u0_J(t_s, molecule, pulse)

    diag = free_diag_J - H * nu * M_vals
    H_mid = diags(diag, offsets=0, format="csr") - U0 * O_op
    return H_mid


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

        # small numerical cleanup
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
def final_population_JM_grid(final_psi: np.ndarray, states: list[tuple[int, int]], J_max: int) -> np.ndarray:
    grid = np.full((J_max + 1, 2 * J_max + 1), np.nan, dtype=np.float64)
    probs = np.abs(final_psi) ** 2
    for p, (J, M) in zip(probs, states):
        grid[J, M + J_max] = p
    return grid



def plot_summary(result: Result, molecule: Molecule, pulse: Pulse, sim: Simulation) -> None:
    t_ps = result.t_s * 1e12
    intensity_W_cm2 = np.array([gaussian_intensity_W_m2(t, pulse) for t in result.t_s]) / 1e4
    nu_GHz = np.array([centrifuge_frequency_Hz(t, pulse) for t in result.t_s]) / 1e9

    final_JM = final_population_JM_grid(result.final_psi, result.states, sim.J_max)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 4, 4], hspace=0.28)

    ax_I = fig.add_subplot(gs[0, 0])
    ax_nu = fig.add_subplot(gs[1, 0], sharex=ax_I)
    ax_J = fig.add_subplot(gs[2, 0], sharex=ax_I)
    ax_M = fig.add_subplot(gs[3, 0], sharex=ax_I)

    ax_I.plot(t_ps, intensity_W_cm2)
    ax_I.set_ylabel("I (W/cm²)")
    ax_I.set_title(f"{molecule.name} | full 3D rotor TDSE in rotating frame")

    ax_nu.plot(t_ps, nu_GHz)
    ax_nu.set_ylabel("ν (GHz)")

    if sim.log_heatmaps:
        norm = LogNorm(vmin=1e-10, vmax=1.0)
    else:
        norm = None

    imJ = ax_J.imshow(
        result.P_J_t.T,
        origin="lower",
        aspect="auto",
        extent=[t_ps[0], t_ps[-1], result.J_values[0] - 0.5, result.J_values[-1] + 0.5],
        interpolation="nearest",
        norm=norm,
    )
    ax_J.set_ylabel("J")
    ax_J.set_title(r"$P_J(t) = \sum_M |c_{J,M}(t)|^2$")
    cbarJ = fig.colorbar(imJ, ax=ax_J)
    cbarJ.set_label("population")

    imM = ax_M.imshow(
        result.P_M_t.T,
        origin="lower",
        aspect="auto",
        extent=[t_ps[0], t_ps[-1], result.M_values[0] - 0.5, result.M_values[-1] + 0.5],
        interpolation="nearest",
        norm=norm,
    )
    ax_M.set_xlabel("time (ps)")
    ax_M.set_ylabel("M")
    ax_M.set_title(r"$P_M(t) = \sum_J |c_{J,M}(t)|^2$")
    cbarM = fig.colorbar(imM, ax=ax_M)
    cbarM.set_label("population")

    plt.show()

    # separate final-state figure
    fig2, ax = plt.subplots(figsize=(8, 5))
    if sim.log_heatmaps:
        im = ax.imshow(
            final_JM,
            origin="lower",
            aspect="auto",
            extent=[-sim.J_max - 0.5, sim.J_max + 0.5, -0.5, sim.J_max + 0.5],
            interpolation="nearest",
            norm=LogNorm(vmin=1e-10, vmax=np.nanmax(final_JM)),
        )
    else:
        im = ax.imshow(
            final_JM,
            origin="lower",
            aspect="auto",
            extent=[-sim.J_max - 0.5, sim.J_max + 0.5, -0.5, sim.J_max + 0.5],
            interpolation="nearest",
        )
    ax.set_xlabel("M")
    ax.set_ylabel("J")
    ax.set_title("final population in |J,M>")
    cbar = fig2.colorbar(im, ax=ax)
    cbar.set_label("population")
    plt.show()


# ============================================================
# main
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # choose molecule here
    # --------------------------------------------------------
    molecule = OCS
    #molecule = CS2

    # --------------------------------------------------------
    # pulse: your approximate cfg
    # --------------------------------------------------------
    pulse = Pulse(
        intensity_peak_W_cm2=5.0e12,   # adjust this
        fwhm_ps=600.0,
        center_ps=0.0,
        freq_start_GHz=100.0,
        freq_end_GHz=100.0,
        chirp_start_ps=300.0,
        chirp_end_ps=300.0,
    )

    # --------------------------------------------------------
    # simulation settings
    # --------------------------------------------------------
    sim = Simulation(
        J_max=24,      # start modestly; increase gradually for convergence
        J0=0,
        M0=0,
        t_min_ps=-900.0,
        t_max_ps=900.0,
        n_time=1200,
        log_heatmaps=True,
    )

    result = propagate_tdse(molecule, pulse, sim)
    plot_summary(result, molecule, pulse, sim)


if __name__ == "__main__":
    main()
