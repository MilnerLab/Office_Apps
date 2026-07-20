import numpy as np
import matplotlib.pyplot as plt

from base_core.quantities.enums import Prefix
from base_core.quantities.models import Frequency, Temperature
from base_core.quantities.molecules import CS2, OCS
from base_core.quantities.specific_models import Intensity

# --------------------------------------------------
# parameters
# --------------------------------------------------
intensity = Intensity(0.1e12, area_prefix=Prefix.CENTI)                 # W/m^2
temperature = Temperature(0.37)             # K
#temperature = None                        # uncomment for sigma0 only

angular_acceleration_rad_per_s2 = 20e20

f_min_hz = 0.0
f_max_hz = 40e9
n_points = 600

molecules = [
    CS2(),
    OCS(),
    # I2(),
]

# --------------------------------------------------
# plot
# --------------------------------------------------
freqs_hz = np.linspace(f_min_hz, f_max_hz, n_points)

plt.figure(figsize=(8, 5))

for molecule in molecules:
    sigmas: list[float] = []

    for f_hz in freqs_hz:
        try:
            sigma = molecule.calculate_spinnability(
                intensity=intensity,
                temperature=temperature,
                centrifuge_frequency=Frequency(f_hz),   # assumes bare float = Hz
                angular_acceleration_rad_per_s2=angular_acceleration_rad_per_s2,
            )
        except ValueError:
            sigma = np.nan

        sigmas.append(sigma)

    plt.plot(2*freqs_hz / 1e9, sigmas, label=molecule.formula)

plt.xlabel("2 * Centrifuge frequency (GHz)")
plt.ylabel("Spinnability")
plt.title("Spinnability vs centrifuge frequency")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()