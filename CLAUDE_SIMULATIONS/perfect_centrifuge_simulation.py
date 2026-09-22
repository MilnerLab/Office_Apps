import matplotlib.pyplot as plt
import numpy as np

from rotor_weights import (
    cos2theta_2D_vmi_x,
    cos2theta_state,
    plot_weights_3d,
    plot_weights_2d,
    plot_wavefunction_3d,
    raman_transition_frequency,
    rotor_weights,
    simulate_vmi_image,
)

if __name__ == "__main__":
    molecule = "CS$_2$" #name
    B = 0.1091   # cm-1 (rotational constant)
    T = 15      # K (temperature)
    J_Final = 14

    # CS2: linear, two spin-0 (32S) nuclei -> only even J exist
    J, M, w = rotor_weights(B=B, T=T, J_max=30, B_unit='cm-1', J_parity='even')

    # States whose Boltzmann weight has underflowed to exact zero (common
    # at very low T) contribute nothing to any ensemble average, but their
    # high J can still overflow the per-state calculations (e.g. lpmv),
    # producing 0 * inf = nan. Dropping them is exact, not an approximation.
    nonzero = w > 0
    J, M, w = J[nonzero], M[nonzero], w[nonzero]

    # A "perfect centrifuge": every state below J_Final is driven all the
    # way up to J_Final (states already at or above J_Final are left
    # alone). M shifts by the same amount as J, so M - J is preserved.
    shift = np.where(J < J_Final, J_Final - J, 0)
    J_exc = J + shift
    M_exc = M + shift

    # Frequency of the final Delta J = +2 Raman transition, the one that
    # actually lands molecules on J_Final (from J_Final - 2).
    c_cm_per_s = 2.99792458e10
    final_transition_cm1 = raman_transition_frequency(B, J_Final - 2, B_unit='cm-1')
    final_transition_THz = final_transition_cm1 * c_cm_per_s * 1e-12
    print(
        f"\nFinal Raman transition J={J_Final - 2} -> J={J_Final}: "
        f"{final_transition_cm1:.4f} cm^-1 ({final_transition_THz:.4f} THz)"
    )

    fig = plt.figure(figsize=(16.5, 6))
    ax_bar3d_pre = fig.add_subplot(1, 3, 1, projection='3d')
    ax_bar3d_post = fig.add_subplot(1, 3, 2, projection='3d')
    ax_wavefn = fig.add_subplot(1, 3, 3, projection='3d')

    plot_weights_3d(J, M, w, title="Population of |J,M> states before excitation", ax=ax_bar3d_pre, colorbar=False)
    plot_weights_3d(J_exc, M_exc, w, title="Population of |J,M> states after excitation", ax=ax_bar3d_post, colorbar=False)

    # <cos^2(theta)>, theta measured from Z (the rotor's own quantization
    # axis) -- the same single reference frame used by the 2D VMI value
    # below (Y line of sight, Z zenith, XZ detector image), so the two
    # printed values describe the same physical picture, not two different
    # rotated axes.
    cos2theta_per_state = cos2theta_state(J_exc, M_exc)
    cos2theta_avg = np.sum(w * cos2theta_per_state)
    print(f"\n<cos^2(theta)> (3D, ensemble average) = {cos2theta_avg:.6f}")

    # Projection onto the XZ plane (Z the rotor's own quantization axis,
    # lying in the detector plane; Y the line of sight), with theta_2D
    # measured from the X axis: <cos^2(theta)>_2D,X = 1 - <|cos(theta)|>_3D.
    cos2theta_2D_per_state = cos2theta_2D_vmi_x(J_exc, M_exc)
    cos2theta_2D_avg = np.sum(w * cos2theta_2D_per_state)
    print(rf"$\langle\cos^2(\theta_{{\mathrm{{2D}}}})\rangle$= {cos2theta_2D_avg:.6f}")

    # Final (excited) ensemble's angular probability density, before
    # the axis-rotation used above for the VMI-style observables.
    plot_wavefunction_3d(J_exc, M_exc, w, title="Final distribution", ax=ax_wavefn, colorbar=False)

    # Simulated raw VMI detector frame (detector parallel to the XZ
    # plane, line of sight along Y) for ion fragments recoiling along
    # the molecular axis of the final excited ensemble, painted onto
    # the same panel's XZ floor.
    simulate_vmi_image(J_exc, M_exc, w, ax=ax_wavefn, title=None, seed=0)

    fig.suptitle(
        rf"{molecule}, T = {T} K, perfect centrifuge to $J_{{final}}$ = {J_Final}" "\n"
        rf"$\langle\cos^2(\theta_{{\mathrm{{2D}}}})\rangle$= {cos2theta_2D_avg:.3f}"
        rf"   |   final transition ($J$={J_Final - 2} $\to$ {J_Final}): {final_transition_cm1:.3f} cm$^{{-1}}$ "
        rf"({final_transition_THz:.3f} THz)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    plt.show()
