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
ANGLE_OFFSET = PI            # rotates the 0-angle reference to the far side

# height-based color gradient for the surface (deep violet -> pale lavender);
# kept out of the blue/teal family so it doesn't blend into the He droplet
SURFACE_COLOR_LOW = "#2E1065"    # deep violet, potential minima
SURFACE_COLOR_HIGH = "#E9D8FD"   # pale lavender, potential maxima
SURFACE_FILL_OPACITY = 0.6       # translucent

MOL_COLORS = [PURE_RED, GREY_B, ORANGE]   # e.g. O - C - S
MOL_RADII = [0.16, 0.14, 0.19]
MOL_POS = [-0.75, 0.0, 0.90]             # positions along the molecular axis

# helium droplet, colored to match animation_optical_centrifuge.py;
# sized to enclose the whole potential landscape, not just the molecule
DROPLET_COLOR = TEAL_E
DROPLET_LABEL_COLOR = TEAL_A
DROPLET_OPACITY = 0.07
DROPLET_RADIUS = 1.3
DROPLET_SCALE = np.array([3.5, 3.5, 1.35])


def U(t):
    """potential in units of U0"""
    return -np.cos(t) ** 2


def make_arrow_tip(tip_position, direction, color, radius=0.09, height=0.22,
                   resolution=(24, 8)):
    """A true 3D cone tip (instead of a camera-facing 2D triangle)."""
    direction = direction / np.linalg.norm(direction)
    cone = Cone(base_radius=radius, height=height, direction=direction,
               resolution=resolution)
    cone.set_fill(color, opacity=1.0)
    cone.set_stroke(color, width=0, opacity=0.0)
    cone.set_shade_in_3d(True)
    points = cone.get_all_points()
    apex = points[np.argmax(points @ direction)]
    cone.shift(tip_position - apex)
    return cone


class AlignmentPotential3D(ThreeDScene):
    def construct(self):
        # side view, slightly from above
        self.set_camera_orientation(phi=40 * DEGREES, theta=-20 * DEGREES,
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
            fill_opacity=SURFACE_FILL_OPACITY,
            stroke_width=0.3,
            stroke_opacity=0.2,
            stroke_color=WHITE,
        )

        # color the surface by height (U value) instead of a checkerboard;
        # the axes below are only used to convert points back to z-values
        # and are never added to the scene
        z_min, z_max = -Z_SCALE * 1.05, Z_SCALE * 0.05
        color_axes = ThreeDAxes(
            x_range=[-R_DISK, R_DISK, 1],
            y_range=[-R_DISK, R_DISK, 1],
            z_range=[z_min, z_max, 0.5],
            x_length=2 * R_DISK,
            y_length=2 * R_DISK,
            z_length=z_max - z_min,
        )
        surface.set_fill_by_value(
            axes=color_axes,
            colorscale=[SURFACE_COLOR_LOW, SURFACE_COLOR_HIGH],
            axis=2,
        )
        surface.set_fill(opacity=SURFACE_FILL_OPACITY)

        # rim of the landscape, drawn solid for readability
        rim = ParametricFunction(
            lambda v: np.array([R_DISK * np.cos(v),
                                R_DISK * np.sin(v),
                                U(v) * Z_SCALE]),
            t_range=[0, TAU],
            color=SURFACE_COLOR_HIGH,
            stroke_width=3,
        )

        # ------------------------------------------------------ xy-plane disk
        circle = Circle(radius=R_DISK, color=GREY_B, stroke_width=2)

        # angle labels on the circle, always facing the camera
        angle_labels = VGroup()
        for ang, tex in [(0, "0"), (PI / 2, r"\tfrac{\pi}{2}"),
                         (PI, r"\pi"), (3 * PI / 2, r"\tfrac{3\pi}{2}")]:
            world_ang = ang + ANGLE_OFFSET
            lab = MathTex(tex, font_size=34)
            lab.move_to((R_DISK + 0.6) * np.array([np.cos(world_ang),
                                                    np.sin(world_ang), 0.0]))
            angle_labels.add(lab)

        # -------------------------------------------------------- z ( = U ) axis
        z_axis = Arrow(start=ORIGIN, end=np.array([0, 0, -Z_SCALE * 1.15]),
                       buff=0, color=WHITE, stroke_width=3,
                       max_tip_length_to_length_ratio=0.08)

        # -------------------------------------------------- polarisation vector
        # only spans from the "0" label to the centre, not across the whole disk
        pol = VGroup(
            Arrow(ORIGIN, LEFT * (R_DISK + 0.15), buff=0,
                  color=TEAL_B, stroke_width=5,
                  max_tip_length_to_length_ratio=0.06),
        )

        # ----------------------------------------------------------- molecule
        z_mol = U(THETA_0) * Z_SCALE
        world_theta = THETA_0 + ANGLE_OFFSET
        u_hat = np.array([np.cos(world_theta), np.sin(world_theta), 0.0])
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

        # ------------------------------------------------- helium droplet
        # translucent wobbly shell enclosing the whole potential landscape,
        # styled after the droplet in animation_optical_centrifuge.py
        droplet_center = np.array([0, 0, -Z_SCALE / 2])

        def droplet_point(u, v):
            theta, phi = u, v
            radius = DROPLET_RADIUS * (
                1.0
                + 0.05 * np.sin(3 * theta) * np.sin(2 * phi)
                + 0.04 * np.cos(5 * theta + 1.7) * np.sin(phi) ** 2
                + 0.033 * np.sin(4 * phi + 1)
            )
            point = radius * DROPLET_SCALE * np.array([np.sin(phi) * np.cos(theta),
                                                        np.sin(phi) * np.sin(theta),
                                                        np.cos(phi)])
            return droplet_center + point

        droplet = Surface(
            droplet_point,
            u_range=[0, TAU], v_range=[0, PI],
            resolution=(32, 16),
            stroke_width=0, stroke_opacity=0,
        )
        droplet.set_fill(DROPLET_COLOR, opacity=DROPLET_OPACITY)
        droplet.set_style(stroke_width=0, stroke_opacity=0, fill_opacity=DROPLET_OPACITY)
        droplet.set_shade_in_3d(True)

        helium_label = Text("He", font_size=32, color=DROPLET_LABEL_COLOR,
                            fill_opacity=0.8)
        helium_label.move_to(droplet_center + np.array([2.7, 2.1, 0.9]))

        # ------------------------------------------------- dissipation arrows
        # short 3D arrows with a sinusoidal (wavy) tail, radiating from the
        # molecule out into the helium to represent dissipation
        def wavy_arrow(start, end, color, n_waves=2.5, amplitude=0.07,
                       stroke_width=3, tip_radius=0.05, tip_height=0.15):
            start, end = np.array(start), np.array(end)
            direction = end - start
            length = np.linalg.norm(direction)
            dir_hat = direction / length
            perp = np.cross(dir_hat, UP)
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(dir_hat, RIGHT)
            perp = perp / np.linalg.norm(perp)

            n_points = 60
            pts = []
            for i in range(n_points + 1):
                t = i / n_points
                taper = np.sin(PI * t)   # zero at both ends, full in the middle
                offset = amplitude * taper * np.sin(t * n_waves * TAU) * perp
                pts.append(start + dir_hat * (t * length) + offset)

            tail = TipableVMobject(color=color, stroke_width=stroke_width)
            tail.set_points_smoothly(pts)
            tip = make_arrow_tip(end, dir_hat, color,
                                 radius=tip_radius, height=tip_height)
            return VGroup(tail, tip)

        DISSIPATION_COLOR = TEAL_A
        # directions perpendicular to the molecular axis (u_hat): one
        # in-plane perpendicular, one vertical, fanned out between them
        perp_inplane = np.array([-np.sin(world_theta), np.cos(world_theta), 0.0])
        perp_vertical = np.array([0.0, 0.0, 1.0])

        def perp_dir(phi_deg):
            phi = phi_deg * DEGREES
            return np.cos(phi) * perp_inplane + np.sin(phi) * perp_vertical

        orange_center = MOL_POS[2] * u_hat + np.array([0, 0, z_mol])
        red_center = MOL_POS[0] * u_hat + np.array([0, 0, z_mol])

        dissipation_arrows = VGroup()
        for atom_center, atom_rad, phis in [
            (orange_center, MOL_RADII[2], [40, 160, 280]),
            (red_center, MOL_RADII[0], [90, 260]),
        ]:
            for phi_deg in phis:
                dir3 = perp_dir(phi_deg)
                start = atom_center + (atom_rad + 0.1) * dir3
                end = atom_center + (atom_rad + 0.65) * dir3
                dissipation_arrows.add(wavy_arrow(start, end, DISSIPATION_COLOR))

        # molecular axis, extended to the rim, and its projection
        mol_axis = Line(np.array([0, 0, z_mol]),
                        R_DISK * u_hat + np.array([0, 0, z_mol]),
                        color=GREEN_B, stroke_width=2.5)
        mol_axis_xy = DashedLine(ORIGIN, R_DISK * u_hat,
                                 color=GREEN_B, stroke_width=1.5,
                                 dash_length=0.1)

        # ------------------------------------------------------- theta marker
        theta_arc = Arc(radius=1.35, start_angle=ANGLE_OFFSET, angle=THETA_0,
                        color=YELLOW, stroke_width=6)
        theta_arc.add_tip(tip_length=0.22)
        theta_label = MathTex(r"\theta", color=YELLOW, font_size=40)
        theta_mid = ANGLE_OFFSET + THETA_0 / 2
        theta_label.move_to(1.95 * np.array([np.cos(theta_mid),
                                             np.sin(theta_mid), 0.0]))

        # ------------------------------------------------- libration arrow
        # traces the molecule's oscillation: starts near the red atom, dips
        # through the bottom of the nearest potential well (raw v = 0), and
        # rises back up on the far wall to the same height; pushed out to a
        # larger radius so it doesn't sit on top of the molecule, and uses
        # true 3D cone tips (not flat, camera-facing triangles) so it reads
        # as lying on the tilted surface from any angle
        r_ps = abs(MOL_POS[0]) + 0.5
        lift = 0.06 * Z_SCALE   # small vertical offset so it sits above the surface

        def ps_point(t):
            return np.array([r_ps * np.cos(t), r_ps * np.sin(t), U(t) * Z_SCALE + lift])

        def ps_tangent(t):
            return np.array([-r_ps * np.sin(t), r_ps * np.cos(t), Z_SCALE * np.sin(2 * t)])

        slope_arrow = TipableVMobject(color=WHITE, stroke_width=4)
        slope_arrow.set_points_smoothly(
            [ps_point(t) for t in np.linspace(-THETA_0, THETA_0, 30)]
        )
        ps_tip_end = make_arrow_tip(ps_point(THETA_0), ps_tangent(THETA_0), WHITE)
        ps_tip_start = make_arrow_tip(ps_point(-THETA_0), -ps_tangent(-THETA_0), WHITE)
        fps_label = MathTex(r"f_{\rm ps}", font_size=32)
        fps_label.move_to(np.array([r_ps * np.cos(THETA_0), r_ps * np.sin(THETA_0),
                                    z_mol]) + np.array([0.75, -0.5, 0.55]))

        # ------------------------------------------------- rotation indicator
        rot_arrow = Arc(radius=0.4, start_angle=-20 * DEGREES,
                        angle=320 * DEGREES, color=WHITE, stroke_width=3)
        rot_arrow.add_tip(tip_length=0.15)
        fcfg_label = MathTex(r"f_{\rm cfg}", font_size=32)
        fcfg_label.move_to(np.array([0.75, 0.45, 0.0]))

        # --------------------------------------------------------- assemble
        self.add(droplet)   # translucent shell drawn first, behind everything
        self.add(surface, rim)
        self.add(circle, pol)
        self.add(z_axis)
        self.add(mol_axis_xy, mol_axis, theta_arc, molecule)
        self.add(rot_arrow, slope_arrow, ps_tip_start, ps_tip_end)
        self.add(dissipation_arrows)

        # billboard all text so it faces the camera
        self.add_fixed_orientation_mobjects(*angle_labels, theta_label,
                                            fcfg_label, fps_label, helium_label)

        self.wait(1)
        # uncomment for a rotating movie:
        # self.begin_ambient_camera_rotation(rate=0.15)
        # self.wait(8)
        # self.stop_ambient_camera_rotation()