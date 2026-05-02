import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from base_core.math.models import Points3D, CartesianAxis
from base_core.math.special_models import Histogram2D, SphericalHarmonicSuperposition


# 1) Zustand: Y_2^2
state = SphericalHarmonicSuperposition.from_mapping(
    {(2, 2): 1.0},
    normalize=True,
)

# 2) Zufällige Richtungen gleichmäßig auf der Kugel
n_points = 2_000_000

cos_theta = np.random.uniform(-1.0, 1.0, n_points)
theta = np.arccos(cos_theta)
phi = np.random.uniform(0.0, 2.0 * np.pi, n_points)

# 3) Wahrscheinlichkeitsdichte auf der Kugel
rho = state.probability_density(theta, phi)

# 4) Kugelrichtungen erzeugen und auf x-z projizieren
points_3d = Points3D.from_spherical(
    r=1.0,
    theta=theta,
    phi=phi,
)

projected = points_3d.project_to_plane(
    CartesianAxis.X | CartesianAxis.Z
)

# 5) Gewichtetes Histogramm
matrix, x_edges, y_edges = np.histogram2d(
    projected.x,
    projected.y,
    bins=350,
    range=[[-1.0, 1.0], [-1.0, 1.0]],
    weights=rho,
)

hist = Histogram2D(matrix=matrix, x_edges=x_edges, y_edges=y_edges)

# 6) Glätten für intensitätsartiges Bild
image = gaussian_filter(hist.matrix.T, sigma=2.0)

# optional: normalisieren
image = image / image.max()

# 7) Plot
plt.figure(figsize=(5, 5))
plt.imshow(
    image,
    extent=[
        hist.x_edges[0],
        hist.x_edges[-1],
        hist.y_edges[0],
        hist.y_edges[-1],
    ],
    origin="lower",
    aspect="equal",
    interpolation="bilinear",
)

plt.xlabel("x")
plt.ylabel("z")
plt.colorbar(label=r"projected $|Y_2^2|^2$")
plt.tight_layout()
plt.show()