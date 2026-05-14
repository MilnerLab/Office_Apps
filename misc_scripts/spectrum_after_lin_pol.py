import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

c0 = 299_792_458.0  # m/s

A0 = 1.0

# Pulse duration parameter T in seconds.
# Your time-domain envelope is exp(-t^2 / (2 T^2)).
T = 600e-12  # 300 ps

# Central optical wavelength
lambda0 = 800e-9  # 800 nm
f0 = c0 / lambda0
omega0 = 2 * np.pi * f0

# Central angular frequencies of the two arms
omega_R0 = omega0
omega_L0 = omega0

# Chirps alpha_R, alpha_L in rad/s^2
# Time-domain phase: omega0*t + 0.5*alpha*t^2
alpha_base_ps2 = 3.7e-2       # rad / ps^2
delta_alpha_ps2 = 5.9e-4      # rad / ps^2

alpha_R = alpha_base_ps2 * 1e24
alpha_L = (alpha_base_ps2 ) * 1e24

# Time delay of left arm in seconds
tau = -0e-12  # 50 ps

# Constant phases
phi_R0 = 0.0
phi_L0 = 0.0

# Frequency window around carrier for spectrum
df_span = 2.0e12  # +/- 2 THz around f0
n_points = 20_000

# Time window for instantaneous frequency plot
t_min = -600e-12
t_max = 600e-12


# ============================================================
# Helper functions
# ============================================================

def E_R_omega(omega: np.ndarray) -> np.ndarray:
    """
    Fourier transform of right arm:
    E_R(t) = A0/sqrt(2) * exp(-t^2/(2T^2))
             * exp(i(omega_R0*t + 0.5*alpha_R*t^2 + phi_R0))
    """
    q_R = 1 / T**2 - 1j * alpha_R

    return (
        A0 / np.sqrt(2)
        * np.exp(1j * phi_R0)
        * np.sqrt(2 * np.pi / q_R)
        * np.exp(-((omega - omega_R0) ** 2) / (2 * q_R))
    )


def E_L_omega(omega: np.ndarray) -> np.ndarray:
    """
    Fourier transform of left arm, delayed by tau:
    E_L(t) = E_L_unshifted(t - tau)

    Fourier shift theorem gives extra factor exp(-i omega tau).
    """
    q_L = 1 / T**2 - 1j * alpha_L

    return (
        A0 / np.sqrt(2)
        * np.exp(1j * phi_L0)
        * np.exp(-1j * omega * tau)
        * np.sqrt(2 * np.pi / q_L)
        * np.exp(-((omega - omega_L0) ** 2) / (2 * q_L))
    )


def E_x_omega(omega: np.ndarray) -> np.ndarray:
    """
    x projection after combining opposite circular components.
    """
    return (E_R_omega(omega) + E_L_omega(omega)) / np.sqrt(2)


def I_x_omega(omega: np.ndarray) -> np.ndarray:
    """
    Spectral intensity as function of angular frequency omega.
    """
    Ex = E_x_omega(omega)
    return np.abs(Ex) ** 2


# ============================================================
# 1. Instantaneous frequency in time
# ============================================================

t = np.linspace(t_min, t_max, 3000)

# Right arm phase:
# phi_R(t) = omega_R0*t + 0.5*alpha_R*t^2 + phi_R0
# omega_inst_R(t) = dphi_R/dt = omega_R0 + alpha_R*t
omega_inst_R = omega_R0 + alpha_R * t

# Left arm is delayed:
# phi_L(t) = omega_L0*(t - tau) + 0.5*alpha_L*(t - tau)^2 + phi_L0
# omega_inst_L(t) = omega_L0 + alpha_L*(t - tau)
omega_inst_L = omega_L0 + alpha_L * (t - tau)

f_inst_R = omega_inst_R / (2 * np.pi)
f_inst_L = omega_inst_L / (2 * np.pi)

# For readability, plot offset from f0 in GHz
plt.figure(figsize=(8, 4.8))
plt.plot(t * 1e12, (f_inst_R - f0) * 1e-9, label="Right arm")
plt.plot(t * 1e12, (f_inst_L - f0) * 1e-9, label="Left arm")
plt.xlabel("time t (ps)")
plt.ylabel(r"instantaneous frequency offset $f(t)-f_0$ (GHz)")
plt.title("Instantaneous frequencies of both arms")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 2. Intensity spectrum versus frequency
# ============================================================

f = np.linspace(f0 - df_span, f0 + df_span, n_points)
omega = 2 * np.pi * f

I_f = I_x_omega(omega)
I_f = I_f / np.max(I_f)

plt.figure(figsize=(8, 4.8))
plt.plot((f - f0) * 1e-12, I_f)
plt.xlabel(r"frequency offset $f-f_0$ (THz)")
plt.ylabel("normalized intensity")
plt.title(r"Spectral intensity $I_x(f)$")
plt.tight_layout()
plt.show()


# ============================================================
# 3. Intensity spectrum versus wavelength
# ============================================================

# Convert frequency axis to wavelength.
# lambda = c / f
lambda_axis = c0 / f

# Sorting is useful because lambda decreases when f increases.
sort_idx = np.argsort(lambda_axis)
lambda_sorted = lambda_axis[sort_idx]
I_lambda_display = I_f[sort_idx]

plt.figure(figsize=(8, 4.8))
plt.plot(lambda_sorted * 1e9, I_lambda_display)
plt.xlabel(r"wavelength $\lambda$ (nm)")
plt.ylabel("normalized intensity")
plt.title(r"Spectral intensity displayed versus wavelength")
plt.tight_layout()
plt.show()


# ============================================================
# Optional: wavelength spectral density correction
# ============================================================
# If you want an intensity density per wavelength instead of simply
# displaying I(omega(lambda)), multiply by |domega/dlambda| = 2*pi*c/lambda^2.
# Many spectrometer plots effectively show counts per wavelength bin,
# so this can matter for broad bandwidths.

use_jacobian = False

if use_jacobian:
    domega_dlambda = 2 * np.pi * c0 / lambda_axis**2
    I_lambda_density = I_f * domega_dlambda
    I_lambda_density = I_lambda_density / np.max(I_lambda_density)

    I_lambda_density_sorted = I_lambda_density[sort_idx]

    plt.figure(figsize=(8, 4.8))
    plt.plot(lambda_sorted * 1e9, I_lambda_density_sorted)
    plt.xlabel(r"wavelength $\lambda$ (nm)")
    plt.ylabel("normalized intensity density")
    plt.title(r"Spectral density corrected by $|d\omega/d\lambda|$")
    plt.tight_layout()
    plt.show()