from math import lgamma

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from scipy.special import sph_harm_y
from tqdm import tqdm


def rotor_weights(B, T, J_max=50, B_unit='cm-1', J_parity='all'):
    """
    Relative populations of the individual |J,M> spherical-harmonic
    states of a rigid linear rotor, including nuclear-spin statistics
    that can forbid every other J (e.g. CS2, with two spin-0 S nuclei,
    has only even J).

    E_J = B * J * (J+1)          (B given in units set by B_unit)
    P(J,M) = exp(-E_J / kT) / Z     for every M in -J..J of an allowed J
    P(J,M) = 0                       for a J forbidden by symmetry

    Parameters
    ----------
    B : float
        Rotational constant.
    T : float
        Temperature (K).
    J_max : int
        Highest J included in the sum/output.
    B_unit : {'cm-1', 'Hz', 'K'}
        Units of B. 'K' means B/k_B is already given in Kelvin.
    J_parity : {'all', 'even', 'odd'}
        Restricts which J are populated at all, e.g. 'even' for CS2
        (identical spin-0 nuclei forbid odd J entirely, not just
        weight them down).

    Returns
    -------
    J : ndarray
        J value for each returned state (repeated 2J+1 times).
    M : ndarray
        M value (-J..J) for each returned state.
    weights : ndarray
        Population weight of each |J,M> state, normalized so
        weights.sum() == 1. Forbidden-parity J are omitted, not
        included as zeros.
    """
    h = 6.62607015e-34   # J s
    c = 2.99792458e10    # cm/s
    kB = 1.380649e-23    # J/K

    J_all = np.arange(J_max + 1)

    if J_parity == 'even':
        J_allowed = J_all[J_all % 2 == 0]
    elif J_parity == 'odd':
        J_allowed = J_all[J_all % 2 == 1]
    elif J_parity == 'all':
        J_allowed = J_all
    else:
        raise ValueError("J_parity must be 'all', 'even', or 'odd'")

    if B_unit == 'cm-1':
        E_over_k = h * c * B * J_allowed * (J_allowed + 1) / kB
    elif B_unit == 'Hz':
        E_over_k = h * B * J_allowed * (J_allowed + 1) / kB
    elif B_unit == 'K':
        E_over_k = B * J_allowed * (J_allowed + 1)
    else:
        raise ValueError("B_unit must be 'cm-1', 'Hz', or 'K'")

    boltz_per_J = np.exp(-E_over_k / T)

    J_list, M_list, w_list = [], [], []
    for Jval, bJ in zip(J_allowed, boltz_per_J):
        for Mval in range(-Jval, Jval + 1):
            J_list.append(Jval)
            M_list.append(Mval)
            w_list.append(bJ)

    J_out = np.array(J_list)
    M_out = np.array(M_list)
    weights = np.array(w_list)
    weights /= weights.sum()

    return J_out, M_out, weights


def raman_transition_frequency(B, J, B_unit='cm-1'):
    """
    Frequency of the Delta J = +2 rotational Raman transition J -> J+2
    of a rigid linear rotor -- e.g. the last step of a stepwise
    "rotational ladder climbing" / optical centrifuge sequence that
    lands molecules on J+2.

    E_J = B * J * (J+1), so E_{J+2} - E_J = B * (4*J + 6).

    Parameters
    ----------
    B : float
        Rotational constant.
    J : int or array
        Lower-state J of the transition (upper state is J+2).
    B_unit : {'cm-1', 'Hz', 'K'}
        Units of B (same convention as rotor_weights); the returned
        value is in the same units.

    Returns
    -------
    float or ndarray
        Transition energy/frequency, in the units given by B_unit.
    """
    J = np.asarray(J)
    return B * (4 * J + 6)


def cos2theta_state(J, M):
    """
    Exact expectation value <J,M|cos^2(theta)|J,M> for a rigid-rotor
    spherical-harmonic state, from the ladder-operator matrix elements
    of cos(theta) in the Y_J^M basis.
    """
    J = np.asarray(J, dtype=float)
    M = np.asarray(M, dtype=float)

    upper = ((J + 1) ** 2 - M ** 2) / ((2 * J + 1) * (2 * J + 3))
    lower = (J ** 2 - M ** 2) / ((2 * J - 1) * (2 * J + 1))

    return upper + lower


_cosabs_cache = {}


def _cosabs_single(J, absM, n_quad):
    """
    <J,M| |cos(theta)| |J,M>, computed once per (J, |M|) and cached.

    No simple closed form exists for this absolute-value moment (unlike
    <cos^2 theta>), so it is done by Gauss-Legendre quadrature over the
    normalized angular density -- exact here since the integrand is a
    polynomial in cos(theta).

    Uses sph_harm_y (the fully normalized spherical harmonic) rather
    than the raw associated Legendre polynomial (lpmv): lpmv's raw
    magnitude and its normalization factor separately blow up/vanish
    for high J even though their product stays O(1), which overflows
    to inf in float before the two combine. sph_harm_y evaluates the
    already-normalized, bounded quantity directly, so this holds even
    at very high J.
    """
    key = (J, absM, n_quad)
    if key in _cosabs_cache:
        return _cosabs_cache[key]

    x, wgt = np.polynomial.legendre.leggauss(n_quad)   # nodes/weights on [-1, 1]
    u = (x + 1) / 2                                    # map to [0, 1]
    du_weight = wgt / 2
    theta = np.arccos(u)

    Y = sph_harm_y(J, absM, theta, 0.0)
    value = 4 * np.pi * np.sum(du_weight * u * np.abs(Y) ** 2)

    _cosabs_cache[key] = value
    return value


def cosabs_state(J, M, n_quad=100):
    """
    <J,M| |cos(theta)| |J,M> for an array (or scalar) of rigid-rotor
    states. This is the 3D quantity that a VMI-projected 2D image
    actually measures as <cos^2(theta)>_2D (see cos2theta_2D_vmi).
    """
    J = np.atleast_1d(np.asarray(J, dtype=int))
    M = np.atleast_1d(np.asarray(M, dtype=int))
    return np.array([_cosabs_single(int(j), abs(int(m)), n_quad) for j, m in zip(J, M)])


def cos2theta_2D_vmi(J, M, n_quad=100):
    """
    Expectation value of cos^2(theta_2D), the in-plane angle measured
    in a velocity-map-imaging detector, following the convention used
    by Stapelfeldt and coworkers: the sample is cylindrically symmetric
    about the alignment axis, and that axis lies in the detector plane
    (the standard geometry, e.g. Coulomb-explosion imaging).

    Integrating the 3D angular distribution over the azimuthal angle
    about the line of sight (the axis perpendicular to the detector)
    gives the exact identity

        <cos^2(theta)>_2D = <|cos(theta)|>_3D

    i.e. the projection converts the 3D cos^2(theta) moment into the
    3D |cos(theta)| moment of the same state/ensemble.
    """
    return cosabs_state(J, M, n_quad)


def cos2theta_2D_vmi_x(J, M, n_quad=100):
    """
    Same VMI geometry as cos2theta_2D_vmi (Z the rotor's quantization
    axis in the detector plane, Y the line of sight, image in the XZ
    plane), but theta_2D measured from the X axis instead of Z.

    Within the projected (x, z) image point, cos^2(theta_2D from X) =
    x^2/(x^2+z^2) = 1 - z^2/(x^2+z^2) = 1 - cos^2(theta_2D from Z), so
    integrating over the line of sight the same way as before gives

        <cos^2(theta)>_2D,X = 1 - <|cos(theta)|>_3D
    """
    return 1.0 - cosabs_state(J, M, n_quad)


_wigner_d_pi2_cache = {}


def wigner_d_pi2(j, mp, m):
    """
    Wigner small-d matrix element d^j_{m',m}(pi/2): the amplitude for
    a state |j,m> quantized along z to be found with M=m' along an
    axis rotated 90 degrees away (about y). At beta=pi/2,
    cos(beta/2) = sin(beta/2) = 1/sqrt(2), which collapses the usual
    Wigner-d sum to a single overall factor of 2^-j.
    """
    key = (j, mp, m)
    if key in _wigner_d_pi2_cache:
        return _wigner_d_pi2_cache[key]

    s_min = max(0, m - mp)
    s_max = min(j + m, j - mp)

    # Everything is done in log-space: the individual factorials (and
    # their products) can vastly exceed float range for large j even
    # though the final d-matrix element is always O(1).
    log_prefactor = -j * np.log(2.0) + 0.5 * (
        lgamma(j + m + 1) + lgamma(j - m + 1) + lgamma(j + mp + 1) + lgamma(j - mp + 1)
    )

    total = 0.0
    for s in range(s_min, s_max + 1):
        log_denom = (
            lgamma(j + m - s + 1) + lgamma(s + 1) + lgamma(j - mp - s + 1) + lgamma(s - m + mp + 1)
        )
        total += (-1) ** (mp - m + s) * np.exp(log_prefactor - log_denom)

    value = total
    _wigner_d_pi2_cache[key] = value
    return value


def _rotate_90(J, M, per_state_of_J, desc="Rotating axis (pi/2)"):
    """
    <J,M|f(theta')|J,M> where theta' is the polar angle about an axis
    rotated 90 degrees from the original z-quantization axis, obtained
    by mixing the M-components of the same J with Wigner-d(pi/2)
    weights: sum_M' d^J_{M',M}(pi/2)^2 * <f(theta)>_{J,M'}.

    per_state_of_J(Jval, Mprimes) must return <f(theta)> for that J at
    each M' in Mprimes (e.g. cos2theta_state or cosabs_state). It only
    depends on J, so it is computed once per unique J and reused for
    every M in that J shell, rather than once per state.
    """
    J = np.atleast_1d(np.asarray(J, dtype=int))
    M = np.atleast_1d(np.asarray(M, dtype=int))

    out = np.empty(len(J), dtype=float)

    with tqdm(total=len(J), desc=desc) as pbar:
        for Jval in np.unique(J):
            Jval = int(Jval)
            idx = np.nonzero(J == Jval)[0]
            Mprimes = np.arange(-Jval, Jval + 1)
            base_vals = per_state_of_J(Jval, Mprimes)

            for i in idx:
                Mval = int(M[i])
                d2 = np.array([wigner_d_pi2(Jval, int(Mp), Mval) ** 2 for Mp in Mprimes])
                out[i] = np.sum(d2 * base_vals)
                pbar.update(1)

    return out


def cos2theta_state_axis90(J, M):
    """<cos^2(theta')>, theta' measured about an axis rotated 90 degrees from z."""
    return _rotate_90(
        J, M, lambda Jval, Mp: cos2theta_state(np.full_like(Mp, Jval), Mp),
        desc="<cos^2(theta)> (rotated axis)",
    )


def cos2theta_2D_vmi_axis90(J, M, n_quad=100):
    """
    <cos^2(theta')>_2D as measured by a VMI detector whose in-plane
    axis is the rotated (90 degree) axis rather than the original
    rotor quantization axis. Same identity as cos2theta_2D_vmi,
    <cos^2(theta')>_2D = <|cos(theta')|>_3D, just evaluated about the
    rotated axis. Unlike cos^2(theta') itself (bounded above by 1/2
    for any single axis, since <x^2>+<y^2>+<z^2>=1), this quantity is
    not bounded by 1/2 -- an equatorial (ring-like) state has
    <|cos(theta')|> -> 2/pi (~0.637) about a perpendicular axis.
    """
    return _rotate_90(
        J, M, lambda Jval, Mp: cosabs_state(np.full_like(Mp, Jval), Mp, n_quad),
        desc="<cos^2(theta)>_2D (rotated axis)",
    )


def plot_weights_3d(J, M, w, title="Boltzmann population of |J,M> states", coverage=0.98, ax=None, colorbar=True):
    """
    3D bar chart of the population weight of each |J,M> state. The J
    axis is truncated at the smallest J that captures `coverage`
    (default 98%) of the total population, and states beyond that
    cutoff are dropped from the plot entirely rather than merely
    clipped by the axis limit.

    Pass an existing 3D-projection `ax` (e.g. from
    fig.add_subplot(..., projection='3d')) to draw into it as part of a
    larger figure instead of creating a new standalone figure.
    """
    order = np.argsort(J)
    J_sorted = J[order]
    w_by_J = np.bincount(J_sorted - J_sorted.min(), weights=w[order])
    J_values = J_sorted.min() + np.arange(len(w_by_J))
    cumulative = np.cumsum(w_by_J) / w.sum()
    J_cutoff = J_values[np.searchsorted(cumulative, coverage)]

    keep = J <= J_cutoff
    J, M, w = J[keep], M[keep], w[keep]

    # Multiple (J, M) states can collide onto the same bar position (e.g.
    # after the centrifuge excitation shift in
    # perfect_centrifuge_simulation.py collapses several original states
    # onto one final J), so weights must be summed per (J, M) before
    # plotting -- otherwise bar3d draws one overlapping bar per raw input
    # row instead of a single bar showing the combined population.
    coords, inverse = np.unique(np.stack([J, M], axis=1), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    w_agg = np.bincount(inverse, weights=w)
    J_u, M_u = coords[:, 0], coords[:, 1]

    owns_fig = ax is None
    if owns_fig:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    colors = cm.viridis(w_agg / w_agg.max())
    ax.bar3d(J_u - 0.3, M_u - 0.3, np.zeros_like(w_agg), 0.6, 0.6, w_agg, color=colors, shade=True)

    ax.set_xlabel("J")
    ax.set_ylabel("M")
    ax.set_zlabel("Population weight")
    ax.set_title(title)
    ax.set_xlim(0, J_cutoff + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    m_lim = np.abs(M).max() + 1
    ax.set_ylim(-m_lim, m_lim)

    if colorbar:
        mappable = cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(0, w_agg.max()))
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Population weight")

    if owns_fig:
        fig.tight_layout()
    return fig, ax


def plot_weights_2d(J, M, w, title="Boltzmann population of |J,M> states", coverage=0.98, ax=None):
    """
    2D imshow of the same |J,M> population weights shown by
    plot_weights_3d, with J truncated the same way (smallest J that
    captures `coverage` of the total population).

    Pass an existing `ax` to draw into it as part of a larger figure
    instead of creating a new standalone figure.
    """
    order = np.argsort(J)
    J_sorted = J[order]
    w_by_J = np.bincount(J_sorted - J_sorted.min(), weights=w[order])
    J_values = J_sorted.min() + np.arange(len(w_by_J))
    cumulative = np.cumsum(w_by_J) / w.sum()
    J_cutoff = J_values[np.searchsorted(cumulative, coverage)]

    keep = J <= J_cutoff
    J, M, w = J[keep], M[keep], w[keep]

    jmax = J.max()
    # M ranges over -jmax..jmax; shift by +jmax so it can index columns.
    # Multiple (J, M) states can land on the same cell (e.g. after the
    # centrifuge excitation shift in perfect_centrifuge_simulation.py
    # collapses several original states onto one final J), so weights
    # must be accumulated rather than assigned -- plain fancy-index
    # assignment would silently drop all but the last write per cell.
    grid = np.zeros((jmax + 1, 2 * jmax + 1))
    np.add.at(grid, (J, M + jmax), w)
    grid[grid == 0] = np.nan

    cmap = cm.viridis.copy()
    cmap.set_bad(color='white')

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    im = ax.imshow(
        grid,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        extent=[-jmax - 0.5, jmax + 0.5, -0.5, jmax + 0.5],
    )
    ax.set_xlabel("M")
    ax.set_ylabel("J")
    ax.set_title(title)

    # Gridlines on cell boundaries (half-integers) so each (J, M) cell
    # is visibly separated, rather than on the (integer) tick labels.
    ax.set_xticks(np.arange(-jmax - 0.5, jmax + 1.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, jmax + 1.5, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=0.5, alpha=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    fig.colorbar(im, ax=ax, label="Population weight")
    if owns_fig:
        fig.tight_layout()
    return fig, ax


def ensemble_angular_density(J, M, w, theta):
    """
    Ensemble-averaged angular probability density
    P(theta) = sum_i w_i * |Y_{J_i}^{M_i}(theta)|^2,
    axially symmetric about the states' own quantization axis (no phi
    dependence, since |e^{iM phi}| = 1).
    """
    theta = np.asarray(theta, dtype=float)
    density = np.zeros_like(theta)

    for Jval, Mval, wt in tqdm(zip(J, M, w), total=len(J), desc="Building angular density"):
        if wt == 0.0:
            # Boltzmann weight underflowed to exact zero (common at very
            # low T with high-J states after the excitation shift); such
            # states contribute nothing, so skip them outright.
            continue

        Jval = int(Jval)
        absM = abs(int(Mval))
        # sph_harm_y is the normalized (bounded) spherical harmonic, unlike
        # the raw associated Legendre polynomial (lpmv), which can overflow
        # to inf at high J before its normalization is applied.
        Y = sph_harm_y(Jval, absM, theta, 0.0)
        density += wt * np.abs(Y) ** 2

    return density


def plot_wavefunction_3d(J, M, w, n_theta=200, n_phi=100, title="Ensemble angular probability density", ax=None, colorbar=True):
    """
    3D surface plot of the ensemble's angular probability density,
    shown as a radial surface: radius(theta, phi) = P(theta), swept
    over phi since the density is axially symmetric. An isotropic
    ensemble renders as a sphere; an ensemble pushed toward the
    equator (high M/J, i.e. large Excitation_Amount) renders as a
    flattened disk/torus shape.

    Pass an existing 3D-projection `ax` to draw into it as part of a
    larger figure instead of creating a new standalone figure.
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    r = ensemble_angular_density(J, M, w, theta)
    r_grid = np.tile(r, (n_phi, 1))

    X = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    Y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    Z = r_grid * np.cos(theta_grid)

    norm = plt.Normalize(r_grid.min(), r_grid.max())
    colors = cm.viridis(norm(r_grid))

    owns_fig = ax is None
    if owns_fig:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, antialiased=True, shade=False)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    r_max = r_grid.max()
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)
    ax.set_box_aspect((1, 1, 1))
    # Put the XZ plane (rather than the default XY) on the floor, so the
    # Y axis is vertical on screen.
    ax.view_init(elev=30, azim=-60, vertical_axis='y')

    if colorbar:
        mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Angular probability density")

    if owns_fig:
        fig.tight_layout()
    return fig, ax


def simulate_vmi_image(
    J, M, w,
    n_events=200_000,
    v0=1.0,
    n_bins=250,
    blur_sigma=0.015,
    fragments_per_event=2,
    seed=None,
    title="Simulated VMI image",
    ax=None,
):
    """
    Monte Carlo simulated raw velocity-map image, as a VMI detector
    parallel to the XZ plane (line of sight along Y, the same
    convention used by cos2theta_2D_vmi_x etc.) would record it for ion
    fragments recoiling along the molecular axis of an ensemble with
    |J,M> populations (J, M, w).

    Each simulated molecule's axis direction is drawn from the
    ensemble's angular distribution -- theta from the marginal
    P(theta) * sin(theta) (P(theta) via ensemble_angular_density; sin
    theta is the solid-angle Jacobian), phi uniform, since the
    distribution has no phi dependence. `fragments_per_event=2` mimics
    a symmetric Coulomb-explosion-style breakup (e.g. CS2 -> S+ + CS+
    + S+, both S+ fragments flying off along +axis and -axis at once);
    use 1 for a single fragment recoiling along a random one of the two
    directions.

    All fragments are given the same recoil speed v0 (a monoenergetic
    Newton sphere); `blur_sigma` adds Gaussian jitter in velocity space
    to stand in for finite detector/recoil-energy resolution, so the
    image looks like a real (if idealized) raw VMI frame rather than a
    noiseless theoretical curve.

    Pass an existing `ax` to draw into it as part of a larger figure
    instead of creating a new standalone figure.
    """
    rng = np.random.default_rng(seed)

    n_theta = 4000
    theta_grid = np.linspace(0, np.pi, n_theta)
    P_theta = ensemble_angular_density(J, M, w, theta_grid)
    pdf = P_theta * np.sin(theta_grid)

    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(theta_grid))))
    cdf /= cdf[-1]

    u = rng.random(n_events)
    theta_samples = np.interp(u, cdf, theta_grid)
    phi_samples = rng.uniform(0, 2 * np.pi, n_events)

    vx = v0 * np.sin(theta_samples) * np.cos(phi_samples)
    vz = v0 * np.cos(theta_samples)

    if fragments_per_event >= 2:
        # Both ends of the axis fly apart at once (e.g. CS2's two S+
        # fragments), so every sampled axis contributes a hit in both
        # +axis and -axis directions.
        vx = np.concatenate([vx, -vx])
        vz = np.concatenate([vz, -vz])
    else:
        # Single fragment: only one of the two ends is the detected
        # species, chosen at random per event.
        sign = rng.choice([-1.0, 1.0], size=n_events)
        vx = sign * vx
        vz = sign * vz

    if blur_sigma > 0:
        vx = vx + rng.normal(0, blur_sigma, vx.shape)
        vz = vz + rng.normal(0, blur_sigma, vz.shape)

    v_lim = v0 * 1.2
    edges = np.linspace(-v_lim, v_lim, n_bins + 1)
    image, _, _ = np.histogram2d(vx, vz, bins=[edges, edges])

    is_3d = ax is not None and getattr(ax, 'name', None) == '3d'

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if is_3d:
        # Paint the image as a flat panel on the existing 3D axes' XZ
        # floor (its current lower Y limit -- see plot_wavefunction_3d's
        # vertical_axis='y' rotation), rescaled from velocity units to
        # that axes' own X/Z extent so it lines up with whatever else is
        # already drawn there.
        x_lo, x_hi = ax.get_xlim()
        z_lo, z_hi = ax.get_zlim()
        y_floor = ax.get_ylim()[0]

        norm_image = image / image.max() if image.max() > 0 else image
        centers = 0.5 * (edges[:-1] + edges[1:])
        Xc = np.interp(centers, (-v_lim, v_lim), (x_lo, x_hi))
        Zc = np.interp(centers, (-v_lim, v_lim), (z_lo, z_hi))
        Xg, Zg = np.meshgrid(Xc, Zc, indexing='ij')
        Yg = np.full_like(Xg, y_floor)

        ax.plot_surface(
            Xg, Yg, Zg,
            facecolors=cm.viridis(norm_image),
            rstride=1, cstride=1, shade=False, antialiased=False,
        )
        if title:
            ax.set_title(title)
        return fig, ax

    # image[i, j] indexes (vx bin i, vz bin j); imshow expects
    # row-major (row=y, col=x), so transpose to put vz on rows.
    im = ax.imshow(
        image.T,
        origin='lower',
        extent=[-v_lim, v_lim, -v_lim, v_lim],
        cmap=cm.viridis,
        aspect='equal',
    )
    ax.set_xlabel("$v_x$")
    ax.set_ylabel("$v_z$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Ion counts")

    if owns_fig:
        fig.tight_layout()
    return fig, ax
