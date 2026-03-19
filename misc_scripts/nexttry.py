from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp


# ============================================================
# constants
# ============================================================
H = 6.62607015e-34
HBAR = 1.054571817e-34
C = 299792458.0
EPS0 = 8.8541878128e-12
PI = np.pi


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
    m_max: int
    m0: int
    t_min_ps: float
    t_max_ps: float
    n_time: int
    log_heatmap: bool = True


# ============================================================
# your molecules (droplet values from screenshots)
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
# helpers
# ============================================================
def ps_to_s(x_ps: float) -> float:
    return x_ps * 1e-12


def GHz_to_Hz(x_GHz: float) -> float:
    return x_GHz * 1e9


def intensity_W_cm2_to_W_m2(x: float) -> float:
    return x * 1e4


def delta_alpha_A3_to_SI(alpha_A3: float) -> float:
    """
    Treat input as polarizability volume in Å^3.
    alpha_SI = 4*pi*eps0 * alpha_volume
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
    """
    Linear ramp in ordinary frequency nu(t), not angular frequency.
    """
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


def u0_J(t_s: float, mol: Molecule, pulse: Pulse) -> float:
    """
    U0 = Delta_alpha * E^2 / 4
       = Delta_alpha_SI * I / (2 c eps0)
    """
    delta_alpha_si = delta_alpha_A3_to_SI(mol.delta_alpha_A3)
    intensity = gaussian_intensity_W_m2(t_s, pulse)
    return delta_alpha_si * intensity / (2.0 * C * EPS0)


def m_basis(m_max: int) -> np.ndarray:
    return np.arange(-m_max, m_max + 1, dtype=int)


# ============================================================
# Hamiltonian and TDSE
# ============================================================
def build_hamiltonian(t_s: float, mol: Molecule, pulse: Pulse, sim: Simulation) -> np.ndarray:
    """
    H = h(B m^2 - D m^4) - h*nu(t)*m - U0(t) cos^2(delta)

    cos^2(delta) = 1/2 + 1/4 exp(+2i delta) + 1/4 exp(-2i delta)
    => couplings m <-> m and m <-> m±2
    """
    m_vals = m_basis(sim.m_max)
    n = len(m_vals)

    nu = centrifuge_frequency_Hz(t_s, pulse)
    U0 = u0_J(t_s, mol, pulse)

    Hmat = np.zeros((n, n), dtype=np.complex128)

    diag = H * (mol.B_Hz * m_vals**2 - mol.D_Hz * m_vals**4 - nu * m_vals)
    diag -= 0.5 * U0
    np.fill_diagonal(Hmat, diag)

    coupling = -0.25 * U0
    for i in range(n - 2):
        Hmat[i, i + 2] = coupling
        Hmat[i + 2, i] = coupling

    return Hmat


def tdse_rhs(t_s: float, psi: np.ndarray, mol: Molecule, pulse: Pulse, sim: Simulation) -> np.ndarray:
    Hmat = build_hamiltonian(t_s, mol, pulse, sim)
    return -1j / HBAR * (Hmat @ psi)


def initial_state(sim: Simulation) -> np.ndarray:
    m_vals = m_basis(sim.m_max)
    psi0 = np.zeros(len(m_vals), dtype=np.complex128)

    idx = np.where(m_vals == sim.m0)[0]
    if len(idx) != 1:
        raise ValueError(f"m0={sim.m0} lies outside chosen basis.")
    psi0[idx[0]] = 1.0
    return psi0


def solve_tdse(mol: Molecule, pulse: Pulse, sim: Simulation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_eval_s = np.linspace(ps_to_s(sim.t_min_ps), ps_to_s(sim.t_max_ps), sim.n_time)
    psi0 = initial_state(sim)

    sol = solve_ivp(
        fun=lambda t, y: tdse_rhs(t, y, mol, pulse, sim),
        t_span=(t_eval_s[0], t_eval_s[-1]),
        y0=psi0,
        t_eval=t_eval_s,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    psi_t = sol.y.T
    norms = np.linalg.norm(psi_t, axis=1)
    psi_t = psi_t / norms[:, None]

    return t_eval_s, m_basis(sim.m_max), psi_t


def populations(psi_t: np.ndarray) -> np.ndarray:
    return np.abs(psi_t) ** 2


# ============================================================
# plotting
# ============================================================
def plot_comparison(
    pulse: Pulse,
    sim_cs2: Simulation,
    sim_ocs: Simulation,
    result_cs2: tuple[np.ndarray, np.ndarray, np.ndarray],
    result_ocs: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    t_cs2_s, m_cs2, psi_cs2 = result_cs2
    t_ocs_s, m_ocs, psi_ocs = result_ocs

    p_cs2 = populations(psi_cs2).T
    p_ocs = populations(psi_ocs).T

    t_ps = t_cs2_s * 1e12
    intensity = np.array([gaussian_intensity_W_m2(t, pulse) for t in t_cs2_s]) / 1e4
    freq = np.array([centrifuge_frequency_Hz(t, pulse) for t in t_cs2_s]) / 1e9

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 4], hspace=0.25, wspace=0.15)

    ax_I = fig.add_subplot(gs[0, :])
    ax_f = fig.add_subplot(gs[1, :], sharex=ax_I)
    ax_cs2 = fig.add_subplot(gs[2, 0], sharex=ax_I)
    ax_ocs = fig.add_subplot(gs[2, 1], sharex=ax_I)

    ax_I.plot(t_ps, intensity)
    ax_I.set_ylabel("I (W/cm²)")
    ax_I.set_title("Planar rotor TDSE in rotating frame")

    ax_f.plot(t_ps, freq)
    ax_f.set_ylabel("ν (GHz)")

    if sim_cs2.log_heatmap:
        norm_cs2 = LogNorm(vmin=1e-8, vmax=1.0)
        norm_ocs = LogNorm(vmin=1e-8, vmax=1.0)
    else:
        norm_cs2 = None
        norm_ocs = None

    im1 = ax_cs2.imshow(
        p_cs2,
        origin="lower",
        aspect="auto",
        extent=[t_ps[0], t_ps[-1], m_cs2[0] - 0.5, m_cs2[-1] + 0.5],
        interpolation="nearest",
        norm=norm_cs2,
    )
    ax_cs2.set_title("CS2")
    ax_cs2.set_xlabel("Time (ps)")
    ax_cs2.set_ylabel("m")

    im2 = ax_ocs.imshow(
        p_ocs,
        origin="lower",
        aspect="auto",
        extent=[t_ps[0], t_ps[-1], m_ocs[0] - 0.5, m_ocs[-1] + 0.5],
        interpolation="nearest",
        norm=norm_ocs,
    )
    ax_ocs.set_title("OCS")
    ax_ocs.set_xlabel("Time (ps)")
    ax_ocs.set_ylabel("m")

    cbar1 = fig.colorbar(im1, ax=ax_cs2)
    cbar1.set_label(r"$|c_m(t)|^2$")

    cbar2 = fig.colorbar(im2, ax=ax_ocs)
    cbar2.set_label(r"$|c_m(t)|^2$")

    plt.show()


# ============================================================
# main
# ============================================================
def main() -> None:
    pulse = Pulse(
        intensity_peak_W_cm2=5.0e12,  # adjust this
        fwhm_ps=600.0,
        center_ps=0.0,
        freq_start_GHz=0.0,
        freq_end_GHz=100.0,
        chirp_start_ps=-300.0,
        chirp_end_ps=300.0,
    )

    # CS2 reaches much higher m than OCS for the same final frequency,
    # so it needs a larger basis.
    sim_cs2 = Simulation(
        m_max=20,
        m0=0,
        t_min_ps=-900.0,
        t_max_ps=900.0,
        n_time=1400,
        log_heatmap=True,
    )

    sim_ocs = Simulation(
        m_max=20,
        m0=0,
        t_min_ps=-900.0,
        t_max_ps=900.0,
        n_time=1400,
        log_heatmap=True,
    )

    result_cs2 = solve_tdse(CS2, pulse, sim_cs2)
    result_ocs = solve_tdse(OCS, pulse, sim_ocs)

    plot_comparison(pulse, sim_cs2, sim_ocs, result_cs2, result_ocs)


if __name__ == "__main__":
    main()