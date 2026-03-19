from __future__ import annotations

"""
Compact-support 3D rigid-rotor TDSE for an optical centrifuge in the lab frame,
with explicit branch tracking.

Why this version is different from the previous script
-----------------------------------------------------
1. The propagation is done in the LAB frame:
       H(t) = H0 - U0(t) * sin^2(theta) * cos^2(phi - beta(t))
   so that for U0(t)=0 the Hamiltonian is EXACTLY field-free.
   There is no residual rotating-frame term after the pulse.

2. The pulse envelope has exact compact support:
   it is exactly zero before the pulse and exactly zero after the pulse.
   The turn-on and turn-off are smooth cosine-squared ramps, which are much
   better suited for near-adiabatic switching than a hard cut.

3. In addition to the TDSE propagation in the fixed |J,M> basis, the script
   also diagonalizes the instantaneous Hamiltonian on a coarse analysis grid
   and tracks one adiabatic branch by maximum-overlap continuity.

4. Because the optical centrifuge coupling only changes J by 0, ±2 and M by
   0, ±2, the Hamiltonian preserves J parity and M parity. The code therefore
   works in the symmetry sector selected by (J0 mod 2, M0 mod 2), which makes
   branch tracking much cheaper.

What the branch tracker means here
----------------------------------
- The TDSE state psi(t) is always represented in the fixed field-free basis |J,M>.
- The tracked branch phi_track(t) is an instantaneous eigenstate of H(t), sampled
  on a coarse grid.
- The quantity
      p_track(t) = |<phi_track(t)|psi(t)>|^2
  tells you how well the evolving TDSE state follows that single adiabatic branch.

Important caveat at EXACTLY zero field
--------------------------------------
At U0=0 the field-free rotor has M-degeneracies for each J. Therefore the branch
identity becomes ambiguous exactly at the zero-field endpoints. For that reason,
branch tracking is only performed while the envelope is above a tiny threshold.
The final state itself is still completely well defined and is reported in the
field-free |J,M> basis and in the reduced P_J distribution.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import expm_multiply
from sympy.physics.wigner import wigner_3j


# ============================================================
# constants
# ============================================================
H = 6.62607015e-34
HBAR = 1.054571817e-34
C = 299792458.0
EPS0 = 8.8541878128e-12
PI = np.pi
BOLTZMANN_J_K = 1.380649e-23


# ============================================================
# data classes
# ============================================================
@dataclass(frozen=True)
class Molecule:
    name: str
    B_Hz: float
    D_Hz: float
    delta_alpha_A3: float


@dataclass(frozen=True)
class CompactSupportPulse:
    intensity_peak_W_cm2: float
    start_ps: float
    ramp_on_ps: float
    chirp_ps: float
    ramp_off_ps: float
    cfg_freq_start_GHz: float
    cfg_freq_end_GHz: float


@dataclass(frozen=True)
class Simulation:
    J_max: int
    J0: int
    M0: int
    free_pre_ps: float
    free_post_ps: float
    n_time: int
    n_branch_samples: int = 140
    branch_env_threshold: float = 1e-8
    log_heatmaps: bool = True
    temperature_K: float | None = None
    store_psi_t: bool = True


@dataclass
class Result:
    t_s: np.ndarray
    psi_t: np.ndarray | None
    final_psi: np.ndarray
    states: list[tuple[int, int]]
    J_vals: np.ndarray
    M_vals: np.ndarray
    P_J_t: np.ndarray
    P_M_t: np.ndarray
    envelope_t: np.ndarray
    cfg_freq_t_Hz: np.ndarray
    beta_t_rad: np.ndarray


@dataclass
class BranchResult:
    t_s: np.ndarray
    sample_indices: np.ndarray
    energies_J: np.ndarray
    overlap_with_tdse: np.ndarray
    branch_P_J_t: np.ndarray
    branch_P_M_t: np.ndarray
    followed_indices: np.ndarray


# ============================================================
# example molecules
# ============================================================
CS2 = Molecule(name="CS2", B_Hz=730e6, D_Hz=1.2e6, delta_alpha_A3=8.7)
OCS = Molecule(name="OCS", B_Hz=2.18e9, D_Hz=9.5e6, delta_alpha_A3=3.7)


# ============================================================
# unit helpers
# ============================================================
def ps_to_s(x_ps: float) -> float:
    return x_ps * 1e-12


def s_to_ps(x_s: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(x_s) * 1e12


def GHz_to_Hz(x_GHz: float) -> float:
    return x_GHz * 1e9


def intensity_W_cm2_to_W_m2(x: float) -> float:
    return x * 1e4


def delta_alpha_A3_to_SI(alpha_A3: float) -> float:
    alpha_m3 = alpha_A3 * 1e-30
    return 4.0 * PI * EPS0 * alpha_m3


# ============================================================
# pulse helpers with exact zero field outside support
# ============================================================
def pulse_times_s(pulse: CompactSupportPulse) -> tuple[float, float, float, float]:
    t0 = ps_to_s(pulse.start_ps)
    t1 = t0 + ps_to_s(pulse.ramp_on_ps)
    t2 = t1 + ps_to_s(pulse.chirp_ps)
    t3 = t2 + ps_to_s(pulse.ramp_off_ps)
    return t0, t1, t2, t3



def envelope(t_s: float, pulse: CompactSupportPulse) -> float:
    """
    Exact compact-support intensity envelope.

    0                           for t <= t0
    sin^2(pi/2 * x)             on the turn-on ramp
    1                           during the chirp interval
    cos^2(pi/2 * x)             on the turn-off ramp
    0                           for t >= t3
    """
    t0, t1, t2, t3 = pulse_times_s(pulse)

    if t_s <= t0 or t_s >= t3:
        return 0.0

    if t_s < t1:
        x = (t_s - t0) / (t1 - t0)
        return np.sin(0.5 * PI * x) ** 2

    if t_s <= t2:
        return 1.0

    x = (t_s - t2) / (t3 - t2)
    return np.cos(0.5 * PI * x) ** 2



def gaussian_like_envelope_note() -> str:
    return (
        "This script uses a compact-support cosine-squared envelope instead of a Gaussian, "
        "so the intensity is exactly zero before and after the pulse while remaining smooth."
    )



def intensity_W_m2(t_s: float, pulse: CompactSupportPulse) -> float:
    I0 = intensity_W_cm2_to_W_m2(pulse.intensity_peak_W_cm2)
    return I0 * envelope(t_s, pulse)



def cfg_frequency_Hz(t_s: float, pulse: CompactSupportPulse) -> float:
    """
    Real centrifuge frequency.

    Choice here:
    - constant start frequency during smooth turn-on
    - linear chirp during the central interval
    - constant end frequency during smooth turn-off

    This makes the ramp regions more adiabatic than chirping while the field is tiny.
    """
    t0, t1, t2, t3 = pulse_times_s(pulse)
    f0 = GHz_to_Hz(pulse.cfg_freq_start_GHz)
    f1 = GHz_to_Hz(pulse.cfg_freq_end_GHz)

    if t_s <= t1:
        return f0
    if t_s >= t2:
        return f1

    frac = (t_s - t1) / (t2 - t1)
    return f0 + frac * (f1 - f0)



def polarization_angle_rad(t_s: float, pulse: CompactSupportPulse) -> float:
    """
    beta(t) = integral 2*pi*f_cfg(t) dt, piecewise exact for the chosen profile.
    For t <= t0 we set beta=0. For t >= t3 it is frozen to beta(t3).
    """
    t0, t1, t2, t3 = pulse_times_s(pulse)
    f0 = GHz_to_Hz(pulse.cfg_freq_start_GHz)
    f1 = GHz_to_Hz(pulse.cfg_freq_end_GHz)

    beta1 = 2.0 * PI * f0 * (t1 - t0)

    if t2 > t1:
        alpha = (f1 - f0) / (t2 - t1)
        beta2 = beta1 + 2.0 * PI * (f0 * (t2 - t1) + 0.5 * alpha * (t2 - t1) ** 2)
    else:
        alpha = 0.0
        beta2 = beta1

    beta3 = beta2 + 2.0 * PI * f1 * (t3 - t2)

    if t_s <= t0:
        return 0.0
    if t_s <= t1:
        return 2.0 * PI * f0 * (t_s - t0)
    if t_s <= t2:
        tau = t_s - t1
        return beta1 + 2.0 * PI * (f0 * tau + 0.5 * alpha * tau * tau)
    if t_s <= t3:
        return beta2 + 2.0 * PI * f1 * (t_s - t2)
    return beta3



def u0_J(t_s: float, molecule: Molecule, pulse: CompactSupportPulse) -> float:
    delta_alpha_si = delta_alpha_A3_to_SI(molecule.delta_alpha_A3)
    return delta_alpha_si * intensity_W_m2(t_s, pulse) / (2.0 * C * EPS0)



def cfg_angular_acceleration_rad_s2(pulse: CompactSupportPulse) -> float:
    if pulse.chirp_ps <= 0.0:
        return 0.0
    f0 = GHz_to_Hz(pulse.cfg_freq_start_GHz)
    f1 = GHz_to_Hz(pulse.cfg_freq_end_GHz)
    return 2.0 * PI * (f1 - f0) / ps_to_s(pulse.chirp_ps)


# ============================================================
# basis restricted to the symmetry sector of the initial state
# ============================================================
def build_sector_basis(J_max: int, J0: int, M0: int) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    J_parity = J0 % 2
    M_parity = M0 % 2

    states: list[tuple[int, int]] = []
    index: dict[tuple[int, int], int] = {}

    k = 0
    for J in range(J_max + 1):
        if J % 2 != J_parity:
            continue
        for M in range(-J, J + 1):
            if M % 2 != M_parity:
                continue
            states.append((J, M))
            index[(J, M)] = k
            k += 1

    return states, index


# ============================================================
# angular matrix elements
# ============================================================
@lru_cache(maxsize=None)
def w3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))



def spherical_harmonic_matrix_element(Jp: int, Mp: int, J: int, M: int, q: int) -> float:
    pref = (-1) ** Mp * np.sqrt((2 * J + 1) * 5.0 * (2 * Jp + 1) / (4.0 * PI))
    return pref * w3j(J, 2, Jp, 0, 0, 0) * w3j(J, 2, Jp, M, q, -Mp)


# decomposition coefficients for sin^2(theta) cos^2(phi - beta)
C_CONST = 1.0 / 3.0
C_Q0 = -(1.0 / 3.0) * np.sqrt(4.0 * PI / 5.0)
C_Q2 = np.sqrt(2.0 * PI / 15.0)



def build_operator_components(
    J_max: int,
    J0: int,
    M0: int,
) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix, list[tuple[int, int]], np.ndarray, np.ndarray]:
    states, index = build_sector_basis(J_max, J0, M0)
    n = len(states)

    J_vals = np.array([J for J, _ in states], dtype=int)
    M_vals = np.array([M for _, M in states], dtype=int)

    rows_q0: list[int] = []
    cols_q0: list[int] = []
    data_q0: list[float] = []

    rows_q2: list[int] = []
    cols_q2: list[int] = []
    data_q2: list[float] = []

    rows_qm2: list[int] = []
    cols_qm2: list[int] = []
    data_qm2: list[float] = []

    for bra_idx, (Jp, Mp) in enumerate(states):
        for q, rows, cols, data in (
            (0, rows_q0, cols_q0, data_q0),
            (2, rows_q2, cols_q2, data_q2),
            (-2, rows_qm2, cols_qm2, data_qm2),
        ):
            M = Mp - q
            for J in range(max(0, Jp - 2), min(J_max, Jp + 2) + 1):
                col_idx = index.get((J, M))
                if col_idx is None:
                    continue
                me = spherical_harmonic_matrix_element(Jp, Mp, J, M, q)
                if me != 0.0:
                    rows.append(bra_idx)
                    cols.append(col_idx)
                    data.append(me)

    I_op = eye(n, format="csr", dtype=np.complex128)
    T_q0 = coo_matrix((data_q0, (rows_q0, cols_q0)), shape=(n, n), dtype=np.complex128).tocsr()
    T_q2 = coo_matrix((data_q2, (rows_q2, cols_q2)), shape=(n, n), dtype=np.complex128).tocsr()
    T_qm2 = coo_matrix((data_qm2, (rows_qm2, cols_qm2)), shape=(n, n), dtype=np.complex128).tocsr()

    return I_op, T_q0, T_q2, T_qm2, states, J_vals, M_vals


# ============================================================
# Hamiltonian
# ============================================================
def rigid_rotor_energy_J(J: int, molecule: Molecule) -> float:
    jj = float(J * (J + 1))
    return H * (molecule.B_Hz * jj - molecule.D_Hz * jj * jj)



def free_rotor_diag(states: Iterable[tuple[int, int]], molecule: Molecule) -> np.ndarray:
    return np.array([rigid_rotor_energy_J(J, molecule) for J, _ in states], dtype=np.float64)



def build_hamiltonian_lab(
    t_s: float,
    free_diag_J: np.ndarray,
    I_op: csr_matrix,
    T_q0: csr_matrix,
    T_q2: csr_matrix,
    T_qm2: csr_matrix,
    molecule: Molecule,
    pulse: CompactSupportPulse,
) -> csr_matrix:
    H0 = diags(free_diag_J.astype(np.complex128), offsets=0, format="csr")
    U0 = u0_J(t_s, molecule, pulse)

    if U0 == 0.0:
        return H0

    beta = polarization_angle_rad(t_s, pulse)
    O_lab = (
        C_CONST * I_op
        + C_Q0 * T_q0
        + C_Q2 * np.exp(-2j * beta) * T_q2
        + C_Q2 * np.exp(+2j * beta) * T_qm2
    )
    return H0 - U0 * O_lab


# ============================================================
# observables and diagnostics
# ============================================================
def initial_state_vector(states: list[tuple[int, int]], J0: int, M0: int) -> np.ndarray:
    psi = np.zeros(len(states), dtype=np.complex128)
    try:
        idx = states.index((J0, M0))
    except ValueError as exc:
        raise ValueError(f"Initial state |J0={J0}, M0={M0}> is not inside the chosen symmetry sector.") from exc
    psi[idx] = 1.0 + 0.0j
    return psi



def accumulate_PJ_PM(prob_state: np.ndarray, J_vals: np.ndarray, M_vals: np.ndarray, J_max: int) -> tuple[np.ndarray, np.ndarray]:
    P_J = np.zeros(J_max + 1, dtype=np.float64)
    P_M = np.zeros(2 * J_max + 1, dtype=np.float64)
    np.add.at(P_J, J_vals, prob_state)
    np.add.at(P_M, M_vals + J_max, prob_state)
    return P_J, P_M



def mean_JJp1_t(P_J_t: np.ndarray) -> np.ndarray:
    J = np.arange(P_J_t.shape[1], dtype=np.float64)
    JJp1 = J * (J + 1.0)
    return P_J_t @ JJp1



def effective_B_from_mean_JJp1_Hz(mean_JJp1: np.ndarray, molecule: Molecule) -> np.ndarray:
    return molecule.B_Hz - 2.0 * molecule.D_Hz * mean_JJp1



def effective_I_kg_m2_t(Beff_t_Hz: np.ndarray) -> np.ndarray:
    Ieff = np.full_like(Beff_t_Hz, np.nan, dtype=np.float64)
    mask = Beff_t_Hz > 0.0
    Ieff[mask] = H / (8.0 * PI**2 * Beff_t_Hz[mask])
    return Ieff



def spinnability_from_PJ(
    t_s: np.ndarray,
    P_J_t: np.ndarray,
    molecule: Molecule,
    pulse: CompactSupportPulse,
    temperature_K: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    mean_JJp1 = mean_JJp1_t(P_J_t)
    Beff_t_Hz = effective_B_from_mean_JJp1_Hz(mean_JJp1, molecule)
    Ieff_t_kg_m2 = effective_I_kg_m2_t(Beff_t_Hz)

    alpha_cfg = cfg_angular_acceleration_rad_s2(pulse)
    U0_t = np.array([u0_J(t, molecule, pulse) for t in t_s], dtype=np.float64)

    sigma0_t = np.zeros_like(U0_t)
    valid = np.isfinite(Ieff_t_kg_m2) & (Ieff_t_kg_m2 > 0.0) & (alpha_cfg > 0.0)
    sigma0_t[valid] = 2.0 * U0_t[valid] / (PI * Ieff_t_kg_m2[valid] * alpha_cfg)

    if temperature_K is None:
        return mean_JJp1, Beff_t_Hz, Ieff_t_kg_m2, sigma0_t, None

    sigmaT_t = np.zeros_like(U0_t)
    validT = valid & (temperature_K > 0.0)
    sigmaT_t[validT] = sigma0_t[validT] * U0_t[validT] / (0.5 * BOLTZMANN_J_K * temperature_K)
    sigma_used_t = np.minimum(sigma0_t, sigmaT_t)
    return mean_JJp1, Beff_t_Hz, Ieff_t_kg_m2, sigma0_t, sigma_used_t


# ============================================================
# TDSE propagation
# ============================================================
def simulation_time_grid_s(pulse: CompactSupportPulse, sim: Simulation) -> np.ndarray:
    t0, _, _, t3 = pulse_times_s(pulse)
    t_min = t0 - ps_to_s(sim.free_pre_ps)
    t_max = t3 + ps_to_s(sim.free_post_ps)
    return np.linspace(t_min, t_max, sim.n_time)



def propagate_tdse(molecule: Molecule, pulse: CompactSupportPulse, sim: Simulation) -> Result:
    I_op, T_q0, T_q2, T_qm2, states, J_vals, M_vals = build_operator_components(sim.J_max, sim.J0, sim.M0)
    free_diag_J = free_rotor_diag(states, molecule)
    t_s = simulation_time_grid_s(pulse, sim)

    psi = initial_state_vector(states, sim.J0, sim.M0)
    psi_t = np.zeros((sim.n_time, len(states)), dtype=np.complex128) if sim.store_psi_t else None
    if psi_t is not None:
        psi_t[0] = psi

    P_J_t = np.zeros((sim.n_time, sim.J_max + 1), dtype=np.float64)
    P_M_t = np.zeros((sim.n_time, 2 * sim.J_max + 1), dtype=np.float64)

    prob = np.abs(psi) ** 2
    P_J_t[0], P_M_t[0] = accumulate_PJ_PM(prob, J_vals, M_vals, sim.J_max)

    envelope_t = np.array([envelope(t, pulse) for t in t_s], dtype=np.float64)
    cfg_freq_t_Hz = np.array([cfg_frequency_Hz(t, pulse) for t in t_s], dtype=np.float64)
    beta_t_rad = np.array([polarization_angle_rad(t, pulse) for t in t_s], dtype=np.float64)

    for k in range(sim.n_time - 1):
        t_mid = 0.5 * (t_s[k] + t_s[k + 1])
        dt = t_s[k + 1] - t_s[k]
        H_mid = build_hamiltonian_lab(t_mid, free_diag_J, I_op, T_q0, T_q2, T_qm2, molecule, pulse)
        A = (-1j * dt / HBAR) * H_mid
        psi = expm_multiply(A, psi)
        psi /= np.linalg.norm(psi)

        if psi_t is not None:
            psi_t[k + 1] = psi

        prob = np.abs(psi) ** 2
        P_J_t[k + 1], P_M_t[k + 1] = accumulate_PJ_PM(prob, J_vals, M_vals, sim.J_max)

    return Result(
        t_s=t_s,
        psi_t=psi_t,
        final_psi=psi,
        states=states,
        J_vals=J_vals,
        M_vals=M_vals,
        P_J_t=P_J_t,
        P_M_t=P_M_t,
        envelope_t=envelope_t,
        cfg_freq_t_Hz=cfg_freq_t_Hz,
        beta_t_rad=beta_t_rad,
    )


# ============================================================
# branch tracking
# ============================================================
def _choose_branch_indices(envelope_t: np.ndarray, n_branch_samples: int, threshold: float) -> np.ndarray:
    active = np.flatnonzero(envelope_t > threshold)
    if active.size == 0:
        raise ValueError("No active pulse region found above branch_env_threshold.")

    if active.size <= n_branch_samples:
        return active

    raw = np.linspace(active[0], active[-1], n_branch_samples)
    return np.unique(np.round(raw).astype(int))



def track_branch(
    molecule: Molecule,
    pulse: CompactSupportPulse,
    sim: Simulation,
    result: Result,
) -> BranchResult:
    if result.psi_t is None:
        raise ValueError("Branch tracking needs store_psi_t=True so the TDSE wavefunction is available.")

    I_op, T_q0, T_q2, T_qm2, states_check, J_vals, M_vals = build_operator_components(sim.J_max, sim.J0, sim.M0)
    if states_check != result.states:
        raise RuntimeError("Internal mismatch in basis reconstruction.")

    free_diag_J = free_rotor_diag(result.states, molecule)
    sample_indices = _choose_branch_indices(result.envelope_t, sim.n_branch_samples, sim.branch_env_threshold)

    n_samples = len(sample_indices)
    branch_P_J_t = np.zeros((n_samples, sim.J_max + 1), dtype=np.float64)
    branch_P_M_t = np.zeros((n_samples, 2 * sim.J_max + 1), dtype=np.float64)
    energies_J = np.zeros(n_samples, dtype=np.float64)
    overlap_with_tdse = np.zeros(n_samples, dtype=np.float64)
    followed_indices = np.zeros(n_samples, dtype=int)

    tracked_vec: np.ndarray | None = None

    for i, idx in enumerate(sample_indices):
        t = result.t_s[idx]
        H_t = build_hamiltonian_lab(t, free_diag_J, I_op, T_q0, T_q2, T_qm2, molecule, pulse).toarray()
        evals, evecs = np.linalg.eigh(H_t)

        if tracked_vec is None:
            overlaps = np.abs(evecs.conj().T @ result.psi_t[idx]) ** 2
            chosen = int(np.argmax(overlaps))
        else:
            overlaps = np.abs(evecs.conj().T @ tracked_vec)
            chosen = int(np.argmax(overlaps))

        vec = evecs[:, chosen]

        if tracked_vec is not None:
            phase = np.vdot(tracked_vec, vec)
            if np.abs(phase) > 0.0:
                vec = vec * np.exp(-1j * np.angle(phase))

        tracked_vec = vec
        followed_indices[i] = chosen
        energies_J[i] = float(np.real(evals[chosen]))
        overlap_with_tdse[i] = float(np.abs(np.vdot(vec, result.psi_t[idx])) ** 2)

        prob = np.abs(vec) ** 2
        branch_P_J_t[i], branch_P_M_t[i] = accumulate_PJ_PM(prob, J_vals, M_vals, sim.J_max)

    return BranchResult(
        t_s=result.t_s[sample_indices],
        sample_indices=sample_indices,
        energies_J=energies_J,
        overlap_with_tdse=overlap_with_tdse,
        branch_P_J_t=branch_P_J_t,
        branch_P_M_t=branch_P_M_t,
        followed_indices=followed_indices,
    )


# ============================================================
# final-state summaries
# ============================================================
def top_components(final_psi: np.ndarray, states: list[tuple[int, int]], n: int = 10) -> list[tuple[tuple[int, int], float]]:
    prob = np.abs(final_psi) ** 2
    order = np.argsort(prob)[::-1][:n]
    return [(states[i], float(prob[i])) for i in order]



def dominant_final_J(P_J_t: np.ndarray) -> tuple[int, float]:
    P = P_J_t[-1]
    J = int(np.argmax(P))
    return J, float(P[J])


# ============================================================
# plotting
# ============================================================
def _heatmap_norm(use_log: bool):
    return LogNorm(vmin=1e-10, vmax=1.0) if use_log else None



def plot_overview(
    molecule: Molecule,
    pulse: CompactSupportPulse,
    sim: Simulation,
    result: Result,
    branch: BranchResult,
) -> None:
    t_ps = s_to_ps(result.t_s)
    t_branch_ps = s_to_ps(branch.t_s)
    intensity_cm2 = np.array([intensity_W_m2(t, pulse) for t in result.t_s]) / 1e4
    freq_GHz = result.cfg_freq_t_Hz / 1e9

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 4, 3, 1.5], hspace=0.25)

    ax_I = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[1, 0], sharex=ax_I)
    ax_PJ = fig.add_subplot(gs[2, 0], sharex=ax_I)
    ax_branch = fig.add_subplot(gs[3, 0], sharex=ax_I)
    ax_follow = fig.add_subplot(gs[4, 0], sharex=ax_I)

    ax_I.plot(t_ps, intensity_cm2)
    ax_I.set_ylabel("I (W/cm²)")
    ax_I.set_title(f"{molecule.name}: compact-support centrifuge TDSE with branch tracking")

    ax_f.plot(t_ps, freq_GHz)
    ax_f.set_ylabel("f_cfg (GHz)")

    im1 = ax_PJ.imshow(
        result.P_J_t.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[t_ps[0], t_ps[-1], -0.5, sim.J_max + 0.5],
        norm=_heatmap_norm(sim.log_heatmaps),
    )
    ax_PJ.set_ylabel("J")
    ax_PJ.set_title("TDSE population P_J(t)")
    cbar1 = fig.colorbar(im1, ax=ax_PJ)
    cbar1.set_label("population")

    im2 = ax_branch.imshow(
        branch.branch_P_J_t.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[t_branch_ps[0], t_branch_ps[-1], -0.5, sim.J_max + 0.5],
        norm=_heatmap_norm(sim.log_heatmaps),
    )
    ax_branch.set_ylabel("J")
    ax_branch.set_title("Tracked instantaneous branch written in the field-free J basis")
    cbar2 = fig.colorbar(im2, ax=ax_branch)
    cbar2.set_label("branch weight")

    ax_follow.plot(t_branch_ps, branch.overlap_with_tdse)
    ax_follow.set_ylim(0.0, 1.05)
    ax_follow.set_ylabel(r"$|\langle \phi_{\rm track}|\psi\rangle|^2$")
    ax_follow.set_xlabel("time (ps)")
    ax_follow.set_title("How well the TDSE state follows the tracked branch")
    ax_follow.grid(True, alpha=0.3)

    plt.show()



def plot_final_populations(result: Result, sim: Simulation) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    ax1.bar(np.arange(sim.J_max + 1), result.P_J_t[-1])
    ax1.set_xlabel("J")
    ax1.set_ylabel("final P_J")
    ax1.set_title("Final field-free J distribution")

    ax2.bar(np.arange(-sim.J_max, sim.J_max + 1), result.P_M_t[-1])
    ax2.set_xlabel("M")
    ax2.set_ylabel("final P_M")
    ax2.set_title("Final field-free M distribution")

    plt.show()



def plot_spinnability(molecule: Molecule, pulse: CompactSupportPulse, result: Result, temperature_K: float | None) -> None:
    t_ps = s_to_ps(result.t_s)
    mean_JJp1, _, _, sigma0_t, sigma_used_t = spinnability_from_PJ(
        result.t_s,
        result.P_J_t,
        molecule,
        pulse,
        temperature_K=temperature_K,
    )

    sigma_plot = sigma0_t if sigma_used_t is None else sigma_used_t

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(t_ps, mean_JJp1, label=r"$\langle J(J+1)\rangle$");
    ax1.set_xlabel("time (ps)")
    ax1.set_ylabel(r"$\langle J(J+1)\rangle$");
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(t_ps, sigma_plot, linestyle="--", label="spinnability")
    ax2.set_ylabel("spinnability")
    ax2.set_yscale("log")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"{molecule.name}: TDSE-weighted diagnostics")
    plt.show()


# ============================================================
# main example
# ============================================================
def main() -> None:
    molecule = CS2

    pulse = CompactSupportPulse(
        intensity_peak_W_cm2=0.10e12,
        start_ps=-400.0,
        ramp_on_ps=220.0,
        chirp_ps=520.0,
        ramp_off_ps=220.0,
        cfg_freq_start_GHz=0.0,
        cfg_freq_end_GHz=100.0,
    )

    sim = Simulation(
        J_max=35,
        J0=1,
        M0=0,
        free_pre_ps=150.0,
        free_post_ps=150.0,
        n_time=2200,
        n_branch_samples=120,
        branch_env_threshold=1e-7,
        log_heatmaps=True,
        temperature_K=None,
        store_psi_t=True,
    )

    print(gaussian_like_envelope_note())
    print("Running TDSE propagation...")
    result = propagate_tdse(molecule, pulse, sim)

    print("Tracking branch on a coarse instantaneous-eigenstate grid...")
    branch = track_branch(molecule, pulse, sim, result)

    J_dom, P_dom = dominant_final_J(result.P_J_t)
    print(f"Dominant final J = {J_dom} with population {P_dom:.6f}")
    print("Top final |J,M> components:")
    for (J, M), p in top_components(result.final_psi, result.states, n=10):
        print(f"  |J={J:2d}, M={M:3d}> : {p:.6e}")

    print(f"Max tracked-branch fidelity = {branch.overlap_with_tdse.max():.6f}")
    print(f"Min tracked-branch fidelity = {branch.overlap_with_tdse.min():.6f}")

    plot_overview(molecule, pulse, sim, result, branch)
    plot_final_populations(result, sim)
    plot_spinnability(molecule, pulse, result, temperature_K=sim.temperature_K)


if __name__ == "__main__":
    main()
