"""
3D visualisation of the laser-induced alignment potential

    U(theta) = -U0 cos^2(theta)

theta = angle between the laser polarisation (x axis) and the most
polarisable molecular axis.

Render (still image):
    manim -sqh alignment_potential_3d.py AlignmentPotential3D
Render (short movie with a slow camera rotation):
    manim -qh alignment_potential_3d.py AlignmentPotential3D
"""

from manim import *
import numpy as np

# ---------------------------------------------------------------- parameters
R_DISK = 3.0                 # radius of the xy-plane disk (scene units)
Z_SCALE = 2.2                # length of the U/U0 = -1 interval (scene units)
THETA_0 = 28 * DEGREES       # instantaneous molecular angle
R_IN = 0.55                  # inner radius of the surface (keeps centre clear)

MOL_COLORS = [PURE_RED, GREY_B, ORANGE]   # e.g. O - C - S
MOL_RADII = [0.16, 0.14, 0.19]
MOL_POS = [-0.75, 0.0, 0.90]             # positions along the molecular axis


def U(t):
    """potential in units of U0"""
    return -np.cos(t) ** 2


class AlignmentPotential3D(ThreeDScene):
    def construct(self):
        # side view, slightly from above
        self.set_camera_orientation(phi=45 * DEGREES, theta=-20 * DEGREES,
                                    zoom=1.15,
                                    frame_center=[0, 0, -0.9])

        # ------------------------------------------------ potential landscape
        surface = Surface(
            lambda u, v: np.array([u * np.cos(v),
                                   u * np.sin(v),
                                   U(v) * Z_SCALE]),
            u_range=[R_IN, R_DISK],
            v_range=[0, TAU],
            resolution=(8, 96),
            fill_opacity=0.75,
            stroke_width=0.4,
            stroke_opacity=0.35,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )

        # rim of the landscape, drawn solid for readability
        rim = ParametricFunction(
            lambda v: np.array([R_DISK * np.cos(v),
                                R_DISK * np.sin(v),
                                U(v) * Z_SCALE]),
            t_range=[0, TAU],
            color=BLUE_B,
            stroke_width=3,
        )

        # ------------------------------------------------------ xy-plane disk
        circle = Circle(radius=R_DISK, color=GREY_B, stroke_width=2)
        axis_x = DashedLine(LEFT * R_DISK, RIGHT * R_DISK,
                            color=GREY_C, stroke_width=1.5)
        axis_y = DashedLine(DOWN * R_DISK, UP * R_DISK,
                            color=GREY_C, stroke_width=1.5)

        # angle labels on the circle, always facing the camera
        angle_labels = VGroup()
        for ang, tex in [(0, "0"), (PI / 2, r"\tfrac{\pi}{2}"),
                         (PI, r"\pi"), (3 * PI / 2, r"\tfrac{3\pi}{2}")]:
            lab = MathTex(tex, font_size=34)
            lab.move_to((R_DISK + 0.6) * np.array([np.cos(ang),
                                                    np.sin(ang), 0.0]))
            angle_labels.add(lab)

        # -------------------------------------------------------- z ( = U ) axis
        z_axis = Arrow(start=ORIGIN, end=np.array([0, 0, -Z_SCALE * 1.15]),
                       buff=0, color=WHITE, stroke_width=3,
                       max_tip_length_to_length_ratio=0.08)
        z_ticks = VGroup()
        z_labels = VGroup()
        for val in (0.0, -0.5, -1.0):
            p = np.array([0, 0, val * Z_SCALE])
            z_ticks.add(Line(p + np.array([-0.09, 0, 0]),
                             p + np.array([0.09, 0, 0]),
                             color=WHITE, stroke_width=2))
            lab = MathTex(f"{val:.1f}".rstrip("0").rstrip("."), font_size=28)
            lab.move_to(p + np.array([-0.42, 0, 0]))
            z_labels.add(lab)
        z_title = MathTex(r"U/U_0", font_size=34)
        z_title.move_to(np.array([0, 0, -Z_SCALE * 1.35]))

        # -------------------------------------------------- polarisation vector
        pol = VGroup(
            Arrow(ORIGIN, RIGHT * (R_DISK + 0.15), buff=0,
                  color=TEAL_B, stroke_width=5,
                  max_tip_length_to_length_ratio=0.06),
            Arrow(ORIGIN, LEFT * (R_DISK + 0.15), buff=0,
                  color=TEAL_B, stroke_width=5,
                  max_tip_length_to_length_ratio=0.06),
        )
        pol_label = MathTex(r"\vec{\varepsilon}", color=TEAL_B, font_size=40)
        pol_label.move_to(np.array([R_DISK + 0.45, -0.55, 0.0]))

        # ----------------------------------------------------------- molecule
        z_mol = U(THETA_0) * Z_SCALE
        u_hat = np.array([np.cos(THETA_0), np.sin(THETA_0), 0.0])
        molecule = VGroup()
        bond = Line3D(
            start=MOL_POS[0] * u_hat + np.array([0, 0, z_mol]),
            end=MOL_POS[2] * u_hat + np.array([0, 0, z_mol]),
            thickness=0.035, color=GREY_D,
        )
        molecule.add(bond)
        for s, rad, col in zip(MOL_POS, MOL_RADII, MOL_COLORS):
            molecule.add(Sphere(center=s * u_hat + np.array([0, 0, z_mol]),
                                radius=rad, resolution=(18, 18))
                         .set_color(col).set_opacity(1.0))

        # molecular axis, extended to the rim, and its projection
        mol_axis = Line(np.array([0, 0, z_mol]),
                        R_DISK * u_hat + np.array([0, 0, z_mol]),
                        color=GREEN_B, stroke_width=2.5)
        mol_axis_xy = DashedLine(ORIGIN, R_DISK * u_hat,
                                 color=GREEN_B, stroke_width=1.5,
                                 dash_length=0.1)

        # ------------------------------------------------------- theta marker
        theta_arc = Arc(radius=1.35, start_angle=0, angle=THETA_0,
                        color=YELLOW, stroke_width=6)
        theta_arc.add_tip(tip_length=0.22)
        theta_label = MathTex(r"\theta", color=YELLOW, font_size=40)
        theta_label.move_to(1.95 * np.array([np.cos(THETA_0 / 2),
                                             np.sin(THETA_0 / 2), 0.0]))

        # --------------------------------------------------------- assemble
        self.add(surface, rim)
        self.add(circle, axis_x, axis_y, pol)
        self.add(z_axis, z_ticks, z_title)
        self.add(mol_axis_xy, mol_axis, theta_arc, molecule)

        # billboard all text so it faces the camera
        self.add_fixed_orientation_mobjects(*angle_labels, *z_labels,
                                            z_title, pol_label, theta_label)

        # title
        title = MathTex(r"U(\theta) = -U_0\cos^2\theta", font_size=44)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)

        self.wait(1)
        # uncomment for a rotating movie:
        # self.begin_ambient_camera_rotation(rate=0.15)
        # self.wait(8)
        # self.stop_ambient_camera_rotation()