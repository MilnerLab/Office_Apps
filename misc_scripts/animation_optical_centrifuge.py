from __future__ import annotations

import numpy as np
from manim import *
import matplotlib.pyplot as plt

from base_core.math.enums import CartesianAxis
from base_core.math.models import Points3D
from base_core.math.special_models import Histogram2D, SphericalHarmonicSuperposition
from base_core.physics.optical_centrifuge import (
    BETA_0,
    CENTRAL_FREQUENCY,
    DELTA_BETA,
    PHASE_0,
    CircularChirpedPulse,
    CircularHandedness,
    OpticalCentrifuge,
    Time,
)
from base_core.quantities.enums import Prefix
from base_core.quantities.models import Length
from base_core.quantities.specific_models import AngularChirp


# Compile it by running:
# manim -pql misc_scripts/animation_optical_centrifuge.py PhysicalOpticalCentrifuge3D
# manim -pqh misc_scripts/animation_optical_centrifuge.py PhysicalOpticalCentrifuge3D


def create_centrifuge_for_animation() -> OpticalCentrifuge:
    left_chirp = AngularChirp(BETA_0 + DELTA_BETA * 0.4)
    pulse_duration = Time(300, Prefix.PICO)

    right_arm = CircularChirpedPulse(
        1,
        CENTRAL_FREQUENCY,
        BETA_0,
        PHASE_0,
        pulse_duration,
        CircularHandedness.RIGHT,
    )

    left_arm = CircularChirpedPulse(
        1,
        CENTRAL_FREQUENCY,
        left_chirp,
        PHASE_0,
        pulse_duration,
        CircularHandedness.LEFT,
    )

    return OpticalCentrifuge(right_arm, left_arm)


def make_y22_histogram_image(
    n_points: int = 300_000,
    bins: int = 350,
    cmap_name: str = "turbo",
) -> np.ndarray:
    rng = np.random.default_rng(1)

    state = SphericalHarmonicSuperposition.from_mapping(
        {(2, 2): 1.0},
        normalize=True,
    )

    cos_theta = rng.uniform(-1.0, 1.0, n_points)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_points)

    rho = state.probability_density(theta, phi)

    points_3d = Points3D.from_spherical(
        r=1.0,
        theta=theta,
        phi=phi,
    )

    projected = points_3d.project_to_plane(
        CartesianAxis.X | CartesianAxis.Z
    )

    matrix, x_edges, y_edges = np.histogram2d(
        projected.x,
        projected.y,
        bins=bins,
        range=[[-1.0, 1.0], [-1.0, 1.0]],
        weights=rho,
    )

    hist = Histogram2D(
        matrix=matrix,
        x_edges=x_edges,
        y_edges=y_edges,
    )

    data = hist.matrix.T.astype(float)

    if np.max(data) > 0:
        data /= np.max(data)

    data = np.sqrt(data)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(data)

    rgba[..., 3] = np.clip(1.6 * data, 0.0, 1.0)

    return (255 * rgba).astype(np.uint8)





class PhysicalOpticalCentrifuge3D(ThreeDScene):
    def construct(self):
        # ============================================================
        # Build physical model
        # ============================================================

        cfg = create_centrifuge_for_animation()

        z_R_extra = Length(0.65, Prefix.MILLI)
        z_L_extra = Length(0)

        # ============================================================
        # Animation parameters
        # ============================================================

        t_min = -6.0
        t_max = 6.0
        t = ValueTracker(t_min)

        seconds_per_manim_unit = 80e-12
        capture_fraction = 0.05

        # ============================================================
        # Scene geometry
        # ============================================================

        propagation_length = 9.0

        droplet_center = np.array([2.2, 0.0, 0.0])
        molecule_center = droplet_center
        molecule_length = 1.45

        pulse_start_x = -5.2
        pulse_end_x = droplet_center[0] + 1.4

        lab_axes_origin = np.array([pulse_start_x, 0.0, 0.0])

        centrifuge_color = PURPLE
        ribbon_width = 1.35
        ribbon_opacity = 0.42

        centrifuge_resolution = (12, 120)
        sphere_resolution = (10, 5)
        droplet_resolution = (24, 12)

        detector_center = molecule_center + np.array([0.0, 0.0, -2.25])
        detector_size = 3.15

        # ============================================================
        # Camera
        # ============================================================

        self.set_camera_orientation(
            phi=68 * DEGREES,
            theta=-55 * DEGREES,
        )

        camera = self.renderer.camera
        if hasattr(camera, "light_source"):
            camera.light_source.move_to(3 * OUT + 4 * LEFT + 5 * UP)

        # ============================================================
        # Pulse-frame mapping
        # ============================================================

        xi_min = -propagation_length / 2
        xi_max = propagation_length / 2

        def pulse_center_x(anim_time: float) -> float:
            progress = (anim_time - t_min) / (t_max - t_min)
            return pulse_start_x + progress * (pulse_end_x - pulse_start_x)

        def pulse_coordinate(x: float, anim_time: float) -> float:
            return x - pulse_center_x(anim_time)

        def pulse_time_from_xi(xi: float) -> float:
            return -xi * seconds_per_manim_unit

        # ============================================================
        # Sample intensity once for normalization
        # ============================================================

        sample_xis = np.linspace(xi_min, xi_max, 800)
        sample_times = np.array([pulse_time_from_xi(xi) for xi in sample_xis])

        sample_intensity = np.asarray(
            cfg.intensity(sample_times, z_R_extra, z_L_extra),
            dtype=float,
        )

        intensity_norm = float(np.max(sample_intensity))
        if intensity_norm <= 0:
            raise ValueError("Centrifuge intensity is zero in the sampled animation window.")

        def normalized_field_amplitude_from_pulse_time(pulse_time: float) -> float:
            intensity = float(cfg.intensity(pulse_time, z_R_extra, z_L_extra))
            return np.sqrt(max(intensity, 0.0) / intensity_norm)

        def field_angle_from_pulse_time(pulse_time: float) -> float:
            return float(cfg.polarization_angle(pulse_time, z_R_extra, z_L_extra))

        def normalized_amplitude_at_x(x: float, anim_time: float) -> float:
            xi = pulse_coordinate(x, anim_time)
            pulse_time = pulse_time_from_xi(xi)
            return normalized_field_amplitude_from_pulse_time(pulse_time)

        def field_angle_at_x(x: float, anim_time: float) -> float:
            xi = pulse_coordinate(x, anim_time)
            pulse_time = pulse_time_from_xi(xi)
            return field_angle_from_pulse_time(pulse_time)

        def field_direction_from_angle(theta: float) -> np.ndarray:
            return np.array([
                0.0,
                np.cos(theta),
                np.sin(theta),
            ])

        def field_direction_at_x(x: float, anim_time: float) -> np.ndarray:
            return field_direction_from_angle(field_angle_at_x(x, anim_time))

        def field_strength_at_molecule(anim_time: float) -> float:
            return normalized_amplitude_at_x(droplet_center[0], anim_time)

        def field_direction_at_molecule(anim_time: float) -> np.ndarray:
            return field_direction_at_x(droplet_center[0], anim_time)

        def centrifuge_frequency_at_molecule(anim_time: float) -> float:
            xi = pulse_coordinate(droplet_center[0], anim_time)
            pulse_time = pulse_time_from_xi(xi)
            return float(cfg.centrifuge_frequency(pulse_time, z_R_extra, z_L_extra))

        def molecule_direction(anim_time: float) -> np.ndarray:
            if field_strength_at_molecule(anim_time) < capture_fraction:
                return np.array([0.0, 1.0, 0.0])

            return field_direction_at_molecule(anim_time)

        # ============================================================
        # Droplet design
        # ============================================================

        def blobby_radius(theta: float, phi: float, phase: float) -> float:
            return (
                1.25
                * (
                    1
                    + 0.080 * np.sin(3 * theta + 1.2 * phase) * np.sin(2 * phi)
                    + 0.060 * np.cos(5 * theta - 0.8 * phase) * np.sin(phi) ** 2
                    + 0.045 * np.sin(4 * phi + 0.6 * phase)
                )
            )

        def blobby_surface(
            u: float,
            v: float,
            center: np.ndarray,
            phase: float,
        ) -> np.ndarray:
            theta = u
            phi = v

            r = blobby_radius(theta, phi, phase)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            return center + np.array([x, y, z])

        # ============================================================
        # Smooth optical centrifuge
        # ============================================================

        def build_static_centrifuge_profile() -> VGroup:
            surface = Surface(
                lambda u, xi: np.array([
                    xi,
                    u
                    * ribbon_width
                    * normalized_field_amplitude_from_pulse_time(pulse_time_from_xi(xi))
                    * np.cos(field_angle_from_pulse_time(pulse_time_from_xi(xi))),
                    u
                    * ribbon_width
                    * normalized_field_amplitude_from_pulse_time(pulse_time_from_xi(xi))
                    * np.sin(field_angle_from_pulse_time(pulse_time_from_xi(xi))),
                ]),
                u_range=[-0.5, 0.5],
                v_range=[xi_min, xi_max],
                resolution=centrifuge_resolution,
                fill_opacity=ribbon_opacity,
                checkerboard_colors=[
                    ManimColor(centrifuge_color),
                    ManimColor(centrifuge_color),
                ],
            )

            surface.set_style(
                stroke_width=0,
                stroke_opacity=0,
                fill_opacity=ribbon_opacity,
            )
            surface.set_shade_in_3d(True)

            return VGroup(surface)

        pulse = build_static_centrifuge_profile()

        def update_pulse(mob: Mobject) -> None:
            current_time = t.get_value()
            mob.move_to(np.array([pulse_center_x(current_time), 0.0, 0.0]))

        pulse.add_updater(update_pulse)

        

        # ============================================================
        # Histogram detector plane
        # ============================================================

        detector_plate = Square(side_length=detector_size)
        detector_plate.set_fill(BLUE_E, opacity=0.65)
        detector_plate.set_stroke(BLUE_D, width=0)
        detector_plate.move_to(detector_center)

        hist_image_array = make_y22_histogram_image()

        histogram_image = ImageMobject(hist_image_array)
        histogram_image.height = detector_size * 0.94
        histogram_image.move_to(detector_center + np.array([0.0, 0.0, 0.01]))

        z_offset = 0.04

        reference_line = Line(
            detector_center + np.array([-detector_size * 0.48, 0.0, z_offset]),
            detector_center + np.array([detector_size * 0.48, 0.0, z_offset]),
            color=BLACK,
            stroke_width=4,
        )

        vertical_line = Line(
            detector_center + np.array([0.0, -detector_size * 0.48, z_offset + 0.002]),
            detector_center + np.array([0.0, detector_size * 0.48, z_offset + 0.002]),
            color=PURPLE,
            stroke_width=2,
        )

        theta_2d = 35 * DEGREES
        line_length = detector_size * 0.45

        angle_line = Line(
            detector_center + np.array([0.0, 0.0, z_offset + 0.004]),
            detector_center
            + line_length * np.array([np.cos(theta_2d), np.sin(theta_2d), 0.0])
            + np.array([0.0, 0.0, z_offset + 0.004]),
            color=GREEN,
            stroke_width=4,
        )

        theta_arc = Arc(
            radius=0.48,
            start_angle=0.0,
            angle=theta_2d,
            color=BLACK,
            stroke_width=4,
        )
        theta_arc.move_arc_center_to(
            detector_center + np.array([0.0, 0.0, z_offset + 0.006])
        )

        theta_label = MathTex(
            r"\theta_{2D}",
            font_size=28,
            color=BLACK,
        )
        theta_label.move_to(
            detector_center
            + 0.72
            * np.array([
                np.cos(theta_2d / 2),
                np.sin(theta_2d / 2),
                0.0,
            ])
            + np.array([0.0, 0.0, z_offset + 0.01])
        )

        detector_group = Group(
            detector_plate,
            histogram_image,
            reference_line,
            vertical_line,
            angle_line,
            theta_arc,
            theta_label,
        )

        detector_group.rotate(
            90 * DEGREES,
            axis=OUT,
            about_point=detector_center,
        )

        # ============================================================
        # Helium droplet
        # ============================================================

        droplet = always_redraw(
            lambda: Surface(
                lambda u, v: blobby_surface(
                    u,
                    v,
                    droplet_center,
                    phase=0.9 * t.get_value(),
                ),
                u_range=[0, TAU],
                v_range=[0, PI],
                resolution=droplet_resolution,
            )
            .set_fill(TEAL_E, opacity=0.16)
            .set_style(
                stroke_width=0,
                stroke_opacity=0,
                fill_opacity=0.16,
            )
            .scale([1.2, 1.0, 0.9])
            .set_shade_in_3d(True)
        )

        # ============================================================
        # Molecule
        # ============================================================

        def molecule() -> VGroup:
            current_time = t.get_value()
            direction = molecule_direction(current_time)

            p1 = molecule_center - 0.5 * molecule_length * direction
            p2 = molecule_center + 0.5 * molecule_length * direction

            bond = Line3D(
                start=p1,
                end=p2,
                color=WHITE,
                thickness=0.055,
            )

            atom1 = Sphere(center=p1, radius=0.17, resolution=sphere_resolution)
            atom1.set_fill(BLUE_C, opacity=1.0)
            atom1.set_style(stroke_width=0, stroke_opacity=0)
            atom1.set_shade_in_3d(True)

            atom2 = Sphere(center=p2, radius=0.17, resolution=sphere_resolution)
            atom2.set_fill(BLUE_C, opacity=1.0)
            atom2.set_style(stroke_width=0, stroke_opacity=0)
            atom2.set_shade_in_3d(True)

            center_atom = Sphere(
                center=molecule_center,
                radius=0.14,
                resolution=sphere_resolution,
            )
            center_atom.set_fill(RED_C, opacity=1.0)
            center_atom.set_style(stroke_width=0, stroke_opacity=0)
            center_atom.set_shade_in_3d(True)

            return VGroup(bond, atom1, atom2, center_atom)

        molecule_obj = always_redraw(molecule)

        # ============================================================
        # Local E-field vector
        # ============================================================

        def e_field_vector_at_molecule() -> Line3D:
            current_time = t.get_value()
            amp = field_strength_at_molecule(current_time)
            direction = field_direction_at_molecule(current_time)

            length = 0.15 + 1.2 * amp
            start = molecule_center - 0.5 * length * direction
            end = molecule_center + 0.5 * length * direction

            return Line3D(
                start=start,
                end=end,
                color=RED,
                thickness=0.035,
            )

        e_vector = always_redraw(e_field_vector_at_molecule)

        # ============================================================
        # Rotation guide
        # ============================================================

        rotation_circle = Circle(
            radius=0.95,
            color=ORANGE,
            stroke_width=4,
            stroke_opacity=0.75,
        )
        rotation_circle.rotate(PI / 2, axis=UP)
        rotation_circle.move_to(molecule_center)

        # ============================================================
        # Labels
        # ============================================================

        def make_lab_axes(
            origin: np.ndarray,
            length: float = 1.15,
            label_shift: float = 0.18,
        ) -> tuple[VGroup, tuple[MathTex, MathTex, MathTex]]:
            """
            Visual convention:

                Manim x-direction = centrifuge propagation direction = lab y
                Manim y-direction = lab x
                Manim z-direction = lab z
            """
            axis_thickness = 0.025

            x_axis = Line3D(
                start=origin,
                end=origin + np.array([0.0, -length, 0.0]),
                color=WHITE,
                thickness=axis_thickness,
            )

            y_axis = Line3D(
                start=origin,
                end=origin + np.array([length, 0.0, 0.0]),
                color=PURPLE,
                thickness=axis_thickness,
            )

            z_axis = Line3D(
                start=origin,
                end=origin + np.array([0.0, 0.0, length]),
                color=WHITE,
                thickness=axis_thickness,
            )

            x_label = MathTex("x", font_size=32, color=WHITE)
            x_label.move_to(origin + np.array([0.0, length + label_shift, 0.0]))

            y_label = MathTex("y", font_size=32, color=PURPLE)
            y_label.move_to(origin + np.array([length + label_shift, 0.0, 0.0]))

            z_label = MathTex("z", font_size=32, color=WHITE)
            z_label.move_to(origin + np.array([0.0, 0.0, length + label_shift]))

            axes = VGroup(x_axis, y_axis, z_axis)
            labels = (x_label, y_label, z_label)

            return axes, labels
        
        lab_axes, lab_axis_labels = make_lab_axes(
            origin=lab_axes_origin,
            length=1.15,
        )
        self.add(lab_axes)
        self.add_fixed_orientation_mobjects(*lab_axis_labels)
        
        he_label = Text("He", font_size=36, color=TEAL_A)
        he_label.move_to(droplet_center + np.array([-0.25, -0.75, 0.85]))

        self.add_fixed_orientation_mobjects(he_label)

        capture_label = MathTex(
            r"A_{\mathrm{mol}}/A_0 > p_{\mathrm{capture}}"
            r"\quad\Rightarrow\quad"
            r"\mathrm{molecule\ follows}\ \vec{E}",
            font_size=24,
            color=GRAY_B,
        )
        capture_label.to_corner(DL)
        self.add_fixed_in_frame_mobjects(capture_label)

        # ============================================================
        # Add objects
        # ============================================================

        self.add(pulse)
        self.add(detector_group)
        self.add(droplet)
        self.add(rotation_circle)
        self.add(molecule_obj)
        self.add(e_vector)
        self.add(he_label)

        # ============================================================
        # Animate
        # ============================================================

        self.play(
            t.animate.set_value(t_max),
            run_time=12,
            rate_func=linear,
        )

        pulse.clear_updaters()

        self.wait(1)