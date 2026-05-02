from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from base_core.quantities.models import Length
from base_core.quantities.enums import Prefix
from base_core.physics.optical_centrifuge import (
    BETA_0,
    CENTRAL_FREQUENCY,
    DELTA_BETA,
    PHASE_0,
    T,
    CircularChirpedPulse,
    CircularHandedness,
    OpticalCentrifuge,
    DELTA_DELAY_ARM,
)
from misc_scripts.animation_optical_centrifuge import create_centrifuge_for_animation


def main() -> None:
    centrifuge = create_centrifuge_for_animation()


    # Zeitachse in ps und SI-Sekunden
    t_ps = np.linspace(-500.0, 500.0, 2000)
    t_s = t_ps * 1e-12

    # Propagationslängen der beiden Arme
    z_L = Length(0.0)
    z_R = Length(0.12, Prefix.MILLI)

    # Momentane optische Kreisfrequenzen der beiden Arme in rad/s
    omega_R = centrifuge.right_arm.instantaneous_angular_frequency(
        t_s,
        float(z_R),
    )

    omega_L = centrifuge.left_arm.instantaneous_angular_frequency(
        t_s,
        float(z_L),
    )

    # Umrechnung rad/s -> Hz -> THz
    f_R_thz = omega_R / (2.0 * np.pi) / 1e12
    f_L_thz = omega_L / (2.0 * np.pi) / 1e12

    # Rotationsfrequenz der linearen Polarisation
    f_cfg_ghz = centrifuge.centrifuge_frequency(
        t_s,
        z_R=z_R,
        z_L=z_L,
    ) / 1e9

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(7, 6),
        sharex=True,
    )

    ax0.plot(t_ps, f_R_thz, label="right arm")
    ax0.plot(t_ps, f_L_thz, label="left arm")

    ax0.set_ylabel("optical frequency / THz")
    ax0.set_title("Instantaneous optical frequencies")
    ax0.grid(True)
    ax0.legend()

    ax1.plot(t_ps, f_cfg_ghz)

    ax1.set_xlabel("time / ps")
    ax1.set_ylabel("centrifuge frequency / GHz")
    ax1.set_title("Optical centrifuge rotation frequency")
    ax1.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()