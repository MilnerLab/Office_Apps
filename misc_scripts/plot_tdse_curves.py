import numpy as np
import matplotlib.pyplot as plt

from base_core.quantities.enums import Prefix
from base_core.quantities.models import Temperature
from base_core.quantities.molecules_tdse import CS2, OCS
from base_core.quantities.specific_models import Intensity

# --------------------------------------------------
# parameters
# --------------------------------------------------
peak_intensity = Intensity(0.1e12, area_prefix=Prefix.CENTI)
temperature = Temperature(0.37)
# temperature = None

angular_acceleration_rad_per_s2 = 6e20
ramp_duration_s = 600e-12

n_points = 300
j_max = 30 #25 #16
# gaussian_sigma_fraction = 1 / (2 * sqrt(2 * ln 2))  -> FWHM ~ ramp_duration
# numerical value:
gaussian_sigma_fraction = 0.42466

molecules = [
    CS2(),
    OCS(),
]

# --------------------------------------------------
# calculate curves once
# --------------------------------------------------
curves: dict[str, object] = {}

for molecule in molecules:
    curve = molecule.calculate_tdse_curve(
        peak_intensity=peak_intensity,
        temperature=temperature,
        angular_acceleration_rad_per_s2=angular_acceleration_rad_per_s2,
        ramp_duration_s=ramp_duration_s,
        n_points=n_points,
        j_max=j_max,
        gaussian_sigma_fraction=gaussian_sigma_fraction,
    )
    curves[molecule.formula] = curve

# --------------------------------------------------
# figure 1: sigma-like diagnostic from TDSE
# --------------------------------------------------
plt.figure(figsize=(8, 5))

for formula, curve in curves.items():
    plt.plot(curve.raman_frequency_hz / 1e9, curve.sigma_from_avg_inverse_I, label=formula)

plt.xlabel("2 × centrifuge frequency (GHz)")
plt.ylabel("Spinnability")
plt.title("TDSE: spinnability diagnostic vs centrifuge frequency")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# figure 2: J diagnostics
# --------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

for formula, curve in curves.items():
    x = curve.raman_frequency_hz / 1e9
    axes[0].plot(x, curve.peak_J, label=formula)
    axes[1].plot(x, curve.effective_J_from_mean, label=formula)
    axes[2].plot(x, curve.average_eff_inverse_I_hz / 1e9, label=formula)

axes[0].set_ylabel("peak J")
axes[0].set_title("TDSE J diagnostics")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_ylabel("J_eff from mean")
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel("avg eff inverse I (GHz)")
axes[2].set_xlabel("2 × centrifuge frequency (GHz)")
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()

# --------------------------------------------------
# figure 3: P_J heatmaps vs Raman frequency
# --------------------------------------------------
fig, axes = plt.subplots(
    len(molecules),
    1,
    figsize=(9, 3.5 * len(molecules)),
    sharex=True,
)

if len(molecules) == 1:
    axes = [axes]

for ax, molecule in zip(axes, molecules):
    curve = curves[molecule.formula]

    x_min = curve.raman_frequency_hz[0] / 1e9
    x_max = curve.raman_frequency_hz[-1] / 1e9

    im = ax.imshow(
        curve.p_J.T,
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, -0.5, curve.p_J.shape[1] - 0.5],
    )

    ax.plot(curve.raman_frequency_hz / 1e9, curve.peak_J, color="red", lw=1.2, label="peak J")
    ax.plot(
        curve.raman_frequency_hz / 1e9,
        curve.effective_J_from_mean,
        color="cyan",
        lw=1.2,
        label="J_eff from mean",
    )

    ax.set_ylabel("J")
    ax.set_title(f"{molecule.formula}: P_J heatmap")
    ax.legend(loc="upper right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P_J")

axes[-1].set_xlabel("2 × centrifuge frequency (GHz)")
fig.tight_layout()
plt.show()

# --------------------------------------------------
# figure 4: same heatmaps vs time
# --------------------------------------------------
fig, axes = plt.subplots(
    len(molecules),
    1,
    figsize=(9, 3.5 * len(molecules)),
    sharex=True,
)

if len(molecules) == 1:
    axes = [axes]

for ax, molecule in zip(axes, molecules):
    curve = curves[molecule.formula]

    x_min = curve.time_s[0] * 1e12
    x_max = curve.time_s[-1] * 1e12

    im = ax.imshow(
        curve.p_J.T,
        origin="lower",
        aspect="auto",
        extent=[x_min, x_max, -0.5, curve.p_J.shape[1] - 0.5],
    )

    ax.set_ylabel("J")
    ax.set_title(f"{molecule.formula}: P_J heatmap vs time")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P_J")

axes[-1].set_xlabel("time (ps)")
fig.tight_layout()
plt.show()

# --------------------------------------------------
# figure 5: selected P_J slices
# --------------------------------------------------
selected_indices = [30, 90, 150, 220, 280]

fig, axes = plt.subplots(1, len(molecules), figsize=(5 * len(molecules), 4), sharey=True)

if len(molecules) == 1:
    axes = [axes]

for ax, molecule in zip(axes, molecules):
    curve = curves[molecule.formula]
    J_values = np.arange(curve.p_J.shape[1])

    for idx in selected_indices:
        if idx >= len(curve.raman_frequency_hz):
            continue

        freq_GHz = curve.raman_frequency_hz[idx] / 1e9
        ax.plot(J_values, curve.p_J[idx], label=f"{freq_GHz:.1f} GHz")

    ax.set_title(molecule.formula)
    ax.set_xlabel("J")
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("P_J")
fig.suptitle("TDSE: selected P_J slices")
fig.tight_layout()
plt.show()
