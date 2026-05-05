from __future__ import annotations

import numpy as np
from manim import *
import matplotlib.pyplot as plt

from base_core.math.enums import AngleUnit, CartesianAxis
from base_core.math.models import Points3D
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
from base_core.math.models import Angle, AngleUnit
from base_core.quantities.models import Length
from base_core.quantities.specific_models import AngularChirp


# Compile it by running:
# manim -pql misc_scripts/animation_optical_centrifuge.py PhysicalOpticalCentrifuge3D
# manim -pqh misc_scripts/animation_optical_centrifuge.py PhysicalOpticalCentrifuge3D


def create_centrifuge_for_animation() -> OpticalCentrifuge:
    left_chirp = AngularChirp(BETA_0 + DELTA_BETA * 0.4)
    pulse_duration = Time(300, Prefix.PICO)
    additional_phase = Angle(225, AngleUnit.DEG)

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
        Angle(PHASE_0 + additional_phase),
        pulse_duration,
        CircularHandedness.LEFT,
    )

    return OpticalCentrifuge(right_arm, left_arm)


def make_axis_aligned_histogram_image_and_c2t(
    molecule_axis: np.ndarray,
    n_points: int = 80_000,
    bins: int = 220,
    cmap_name: str = "turbo",
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(1)

    axis = np.asarray(molecule_axis, dtype=float)
    axis /= np.linalg.norm(axis)

    cos_theta = rng.uniform(-1.0, 1.0, n_points)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_points)

    points_3d = Points3D.from_spherical(
        r=1.0,
        theta=theta,
        phi=phi,
    )

    n = np.column_stack([
        points_3d.x,
        points_3d.y,
        points_3d.z,
    ])
    n /= np.linalg.norm(n, axis=1, keepdims=True)

    # Rotierte |Y_1^0|^2-artige Verteilung
    rho = np.abs(n @ axis) ** 8

    projected = points_3d.project_to_plane(
        CartesianAxis.X | CartesianAxis.Z
    )

    x = projected.x
    y = projected.y

    r2 = x**2 + y**2
    valid = r2 > 1e-12

    cos2_theta_2d = np.zeros_like(r2)
    cos2_theta_2d[valid] = y[valid] ** 2 / r2[valid]

    c2t = float(np.sum(rho[valid] * cos2_theta_2d[valid]) / np.sum(rho[valid]))

    matrix, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[-1.0, 1.0], [-1.0, 1.0]],
        weights=rho,
    )

    data = matrix.T.astype(float)

    if np.max(data) > 0:
        data /= np.max(data)

    data = np.sqrt(data)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(data)
    rgba[..., 3] = np.clip(1.6 * data, 0.0, 1.0)

    return (255 * rgba).astype(np.uint8), c2t




class PhysicalOpticalCentrifuge3D(ThreeDScene):
    def construct(self):
        # ============================================================
        # Build physical model
        # ============================================================
        
        # Black Background setup
        self.camera.background_color = BLACK
        
        color_coordinate_system = WHITE
        color_cfg = PURPLE
        
        color_HE = TEAL_A
        color_atom1 = GREY
        color_atom2 = BLUE_C
        color_bond = WHITE
        color_ring = ORANGE
        color_Efield = RED
        color_droplet = TEAL_E
        
        
        color_plot = WHITE
        color_plotline = YELLOW
        
        color_detector = BLUE_E
        color_detector_boarder = WHITE
        color_angle_line = GREEN
        color_angle = BLACK
        
        
        # White Background setup
        self.camera.background_color = WHITE
        
        color_coordinate_system = DARKER_GREY
        color_cfg = PINK
        
        color_HE = DARK_BLUE
        color_atom1 = GREY
        color_atom2 = RED_E
        color_bond = LIGHTER_GREY
        color_ring = GRAY
        color_Efield = RED
        color_droplet = TEAL_E
        
        color_plot = DARKER_GREY
        color_plotline = GRAY_BROWN
        
        color_detector = PURPLE_E
        color_detector_boarder = GREY_BROWN
        color_angle_line = GRAY_BROWN
        color_angle = GRAY_D
        
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

        # ============================================================
        # Scene geometry
        # ============================================================

        propagation_length = 9.0

        droplet_center = np.array([2.2, 0.0, 0.0])
        molecule_center = droplet_center
        molecule_length = 1.45

        pulse_start_x = -5.2
        pulse_end_x = droplet_center[0] + 4

        lab_axes_origin = np.array([pulse_start_x, 0.0, 0.0])

        ribbon_width = 1.35
        ribbon_opacity = 0.42

        centrifuge_resolution = (12, 120)
        sphere_resolution = (10, 5)
        droplet_resolution = (24, 12)
        
        '''centrifuge_resolution = (24, 240)
        sphere_resolution = (24, 12)
        droplet_resolution = (48, 24)'''

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

        
        def rotate_point_around_z(
            point: np.ndarray,
            angle: float,
            center: np.ndarray,
        ) -> np.ndarray:
            p = point - center

            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ])

            return center + rot @ p
        
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
            if centrifuge_frequency_at_molecule(anim_time) > 0:
                return np.array([0.0, 0.0, 1.0])
            
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
                    ManimColor(color_cfg),
                    ManimColor(color_cfg),
                ],
            )

            surface.set_style(
                stroke_width=0,
                stroke_opacity=0,
                fill_opacity=ribbon_opacity,
            )
            surface.set_shade_in_3d(True)

            return VGroup(surface)

        def make_static_cfg_e_vector() -> VGroup:
            xi = xi_max  # Position am Anfang der CFG, ggf. anpassen

            # Feldrichtung an dieser festen Stelle innerhalb der CFG
            pulse_time = pulse_time_from_xi(xi)
            theta = field_angle_from_pulse_time(pulse_time)
            direction = field_direction_from_angle(theta)
            direction = direction / np.linalg.norm(direction)

            length = 0.85
            shaft_thickness = 0.045
            cone_radius = 0.09
            cone_height = 0.20

            center = np.array([xi, 0.0, 0.0])

            start_tip = center - 0.5 * length * direction
            end_tip = center + 0.5 * length * direction

            start_line = start_tip + cone_height * direction
            end_line = end_tip - cone_height * direction

            def make_arrow_tip(
                tip_position: np.ndarray,
                tip_direction: np.ndarray,
            ) -> Cone:
                tip_direction = tip_direction / np.linalg.norm(tip_direction)

                cone = Cone(
                    base_radius=cone_radius,
                    height=cone_height,
                    direction=tip_direction,
                    resolution=(24, 8),
                )

                cone.set_fill(color_Efield, opacity=1.0)
                cone.set_stroke(color_Efield, width=0, opacity=0.0)
                cone.set_shade_in_3d(False)

                points = cone.get_all_points()
                projections = points @ tip_direction
                current_tip = points[np.argmax(projections)]

                cone.shift(tip_position - current_tip)

                return cone

            shaft = Line3D(
                start=start_line,
                end=end_line,
                color=color_Efield,
                thickness=shaft_thickness,
            )
            shaft.set_shade_in_3d(False)

            start_cone = make_arrow_tip(start_tip, -direction)
            end_cone = make_arrow_tip(end_tip, direction)

            return VGroup(shaft, start_cone, end_cone)
        
        pulse = build_static_centrifuge_profile()
        e_vector = make_static_cfg_e_vector()

        pulse.add(e_vector)

        def update_pulse(mob: Mobject) -> None:
            current_time = t.get_value()
            mob.move_to(np.array([pulse_center_x(current_time), 0.0, 0.0]))

        pulse.add_updater(update_pulse)
        
        e_label_xi = xi_max

        e_vector_label = MathTex(
            r"\vec{E}_{\mathrm{CFG}}",
            font_size=34,
            color=color_Efield,
        )

        def update_e_vector_label(mob: MathTex) -> None:
            current_time = t.get_value()
            mob.move_to(
                np.array([
                    pulse_center_x(current_time) + e_label_xi,
                    0.0,
                    0.75,
                ])
            )

        e_vector_label.add_updater(update_e_vector_label)

        self.add_fixed_orientation_mobjects(e_vector_label)

        

        # ============================================================
        # Histogram detector plane
        # ============================================================

        detector_plate = Square(side_length=detector_size)
        detector_plate.set_fill(color_detector, opacity=0.65)
        detector_plate.set_stroke(color_detector_boarder, width=2)
        detector_plate.move_to(detector_center)

        hist_cache = {
            "time": None,
            "image_array": None,
            "c2t": None,
        }
        c2t_history: list[tuple[float, float]] = []
        last_history_time = {"value": None}

        def get_histogram_frame_data() -> tuple[np.ndarray, float]:
            current_time = round(t.get_value(), 4)

            if hist_cache["time"] == current_time:
                return hist_cache["image_array"], hist_cache["c2t"]

            axis = molecule_direction(current_time)

            axis_for_hist = np.array([
                axis[0],
                axis[2],
                -axis[1],
            ])

            image_array, c2t = make_axis_aligned_histogram_image_and_c2t(
                axis_for_hist,
                n_points=30_000,
                bins=160,
            )
            field_strength = field_strength_at_molecule(current_time)
            field_strength = np.clip(field_strength, 0.0, 1.0)

            c2t = 0.5 + field_strength * (c2t - 0.5)
            
            hist_cache["time"] = current_time
            hist_cache["image_array"] = image_array
            hist_cache["c2t"] = c2t

            if last_history_time["value"] != current_time:
                c2t_history.append((current_time, c2t))
                last_history_time["value"] = current_time

            return image_array, c2t

        def make_histogram_mobject() -> ImageMobject:
            image_array, _ = get_histogram_frame_data()

            image = ImageMobject(image_array)
            image.height = detector_size * 0.94
            image.move_to(detector_center + np.array([0.0, 0.0, 0.01]))

            return image
        histogram_image = always_redraw(make_histogram_mobject)
        
        z_offset = 0.04

        reference_line = Line(
            detector_center + np.array([-detector_size * 0.48, 0.0, z_offset]),
            detector_center + np.array([detector_size * 0.48, 0.0, z_offset]),
            color=color_angle,
            stroke_width=4,
        )

        vertical_line = Line(
            detector_center + np.array([0.0, -detector_size * 0.48, z_offset + 0.002]),
            detector_center + np.array([0.0, detector_size * 0.48, z_offset + 0.002]),
            color=color_cfg,
            stroke_width=2,
        )

        theta_2d = 35 * DEGREES

        # grüne Linie: oberer linker Quadrant
        theta_line = PI - theta_2d
        line_length = detector_size * 0.45

        angle_line = Line(
            detector_center + np.array([0.0, 0.0, z_offset + 0.004]),
            detector_center
            + line_length
            * np.array([
                np.cos(theta_line),
                np.sin(theta_line),
                0.0,
            ])
            + np.array([0.0, 0.0, z_offset + 0.004]),
            color=color_angle_line,
            stroke_width=4,
        )

       # kleiner Winkel zwischen linker schwarzer Achse und grüner Linie
        arc_radius = 0.62

        theta_arc = Arc(
            radius=arc_radius,
            start_angle=PI,
            angle=-theta_2d,
            color=color_angle,
            stroke_width=4,
        )
        theta_arc.move_arc_center_to(
            detector_center + np.array([0.0, 0.0, z_offset + 0.006])
        )

        # Label schwebt über dem kleinen Winkel
        arc_mid_angle = PI - theta_2d / 2

        label_anchor = (
            detector_center
            + arc_radius * 0.7
            * np.array([
                np.cos(arc_mid_angle),
                np.sin(arc_mid_angle),
                0.0,
            ])
        )

        lable_line = DashedLine(
            label_anchor,
            label_anchor + np.array([-0.7, 0.4, 0.38]),
            color=color_angle_line,
            stroke_width=2,
            fill_opacity=0.6,
        )

        # Weil lable_line später mit detector_group rotiert wird,
        # muss die Label-Position genauso rotiert werden.
        theta_label_pos = rotate_point_around_z(
            label_anchor + np.array([-0.7, 0.4, 0.6]),
            90 * DEGREES,
            detector_center,
        )

        theta_label = MathTex(
            r"\theta_{2D}",
            font_size=36,
            color=color_angle,
        )

        detector_group = Group(
            detector_plate,
            histogram_image,
            reference_line,
            vertical_line,
            angle_line,
            theta_arc,
            lable_line,
        )

        detector_group.rotate(
            90 * DEGREES,
            axis=OUT,
            about_point=detector_center,
        )
        
        theta_label.move_to(theta_label_pos)
        self.add(detector_group)
        self.add_fixed_orientation_mobjects(theta_label)

        # ============================================================
        # Live c2t plot as fixed 2D camera overlay
        # ============================================================

        plot_width = 2.7
        plot_height = 1.0
        lable_scaling = 6

        plot_x_min = t_min
        plot_x_max = t_max

        plot_y_min = 0.45
        plot_y_max = 1.00

        # Screen position, not 3D world position
        plot_origin = np.array([-4, -2.2, 0.0])

        def plot_point_2d(x_value: float, y_value: float) -> np.ndarray:
            x_frac = (x_value - plot_x_min) / (plot_x_max - plot_x_min)
            y_frac = (y_value - plot_y_min) / (plot_y_max - plot_y_min)

            x_frac = np.clip(x_frac, 0.0, 1.0)
            y_frac = np.clip(y_frac, 0.0, 1.0)

            return plot_origin + np.array([
                plot_width * x_frac,
                plot_height * y_frac,
                0.0,
            ])
        
        plot_x_axis = Line(
            plot_point_2d(plot_x_min, plot_y_min),
            plot_point_2d(plot_x_max, plot_y_min),
            color=color_plot,
            stroke_width=2,
        )

        plot_y_axis = Line(
            plot_point_2d(plot_x_min, plot_y_min),
            plot_point_2d(plot_x_min, plot_y_max),
            color=color_plot,
            stroke_width=2,
        )

        plot_baseline = DashedLine(
            plot_point_2d(plot_x_min, 0.5),
            plot_point_2d(plot_x_max, 0.5),
            color=color_plot,
            stroke_width=1.5,
            dash_length=0.05,
        )

        plot_title = MathTex(
            r"\langle \cos^2\theta_{2D}\rangle",
            font_size=24 + lable_scaling,
            color=color_plot,
        )
        plot_title.move_to(plot_origin + np.array([0, plot_height + 0.28, 0.0]))

        plot_x_label = MathTex(
            r"\mathrm{probe\ delay}",
            font_size=18 + lable_scaling,
            color=color_plot,
        )
        plot_x_label.move_to(plot_origin + np.array([plot_width * 0.58, -0.22, 0.0]))

        plot_y05_label = MathTex(
            r"0.5",
            font_size=16 + lable_scaling,
            color=color_plot,
        )
        plot_y05_label.move_to(plot_point_2d(plot_x_min, 0.5) + np.array([-0.22, 0.0, 0.0]))

        plot_y1_label = MathTex(
            r"1.0",
            font_size=16 + lable_scaling,
            color=color_plot,
        )
        plot_y1_label.move_to(plot_point_2d(plot_x_min, 1.0) + np.array([-0.22, 0.0, 0.0]))

        plot_static = VGroup(
            plot_x_axis,
            plot_y_axis,
            plot_baseline,
            plot_title,
            plot_x_label,
            plot_y05_label,
            plot_y1_label,
        )

        self.add_fixed_in_frame_mobjects(plot_static)

        c2t_curve = VMobject()
        c2t_curve.set_stroke(color_plotline, width=3)

        c2t_dot = Dot(
            point=plot_point_2d(t_min, 0.5),
            radius=0.035,
            color=color_plotline,
        )

        self.add_fixed_in_frame_mobjects(c2t_curve, c2t_dot)
        
        def update_c2t_curve(mob: VMobject) -> None:
            get_histogram_frame_data()

            if len(c2t_history) < 2:
                mob.set_points([])
                return

            points = [
                plot_point_2d(time_value, c2t_value)
                for time_value, c2t_value in c2t_history
            ]

            mob.set_points_as_corners(points)
            mob.set_stroke(color_plotline, width=3)


        def update_c2t_dot(mob: Dot) -> None:
            get_histogram_frame_data()

            if len(c2t_history) == 0:
                return

            time_value, c2t_value = c2t_history[-1]
            mob.move_to(plot_point_2d(time_value, c2t_value))


        c2t_curve.add_updater(update_c2t_curve)
        c2t_dot.add_updater(update_c2t_dot)
        
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
            .set_fill(color_droplet, opacity=0.16)
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
            direction = direction / np.linalg.norm(direction)

            p1 = molecule_center - 0.5 * molecule_length * direction
            p2 = molecule_center + 0.5 * molecule_length * direction

            center_radius = 0.14

            bond1 = Line3D(
                start=p1,
                end=molecule_center - center_radius * direction,
                color=color_bond,
                thickness=0.03,
            )

            bond2 = Line3D(
                start=molecule_center + center_radius * direction,
                end=p2,
                color=color_bond,
                thickness=0.03,
            )

            atom1 = Sphere(center=p1, radius=0.17, resolution=sphere_resolution)
            atom1.set_fill(color_atom2, opacity=1.0)
            atom1.set_style(stroke_width=0, stroke_opacity=0)
            atom1.set_shade_in_3d(True)

            atom2 = Sphere(center=p2, radius=0.17, resolution=sphere_resolution)
            atom2.set_fill(color_atom2, opacity=1.0)
            atom2.set_style(stroke_width=0, stroke_opacity=0)
            atom2.set_shade_in_3d(True)

            center_atom = Sphere(
                center=molecule_center,
                radius=center_radius,
                resolution=sphere_resolution,
            )
            center_atom.set_fill(color_atom1, opacity=1.0)
            center_atom.set_style(stroke_width=0, stroke_opacity=0)
            center_atom.set_shade_in_3d(True)

            return VGroup(bond1, bond2, atom1, atom2, center_atom)

        molecule_obj = always_redraw(molecule)


        # ============================================================
        # Rotation guide
        # ============================================================

        rotation_circle = Circle(
            radius=0.95,
            color=color_ring,
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
            axis_thickness = 0.035
            cone_radius = 0.1
            cone_height = 0.22

            def make_arrow_tip(
                tip_position: np.ndarray,
                direction: np.ndarray,
                color: ManimColor | str,
            ) -> Cone:
                direction = direction / np.linalg.norm(direction)

                cone = Cone(
                    base_radius=cone_radius,
                    height=cone_height,
                    direction=direction,
                    resolution=(24, 8),
                )

                cone.set_fill(color, opacity=1.0)
                cone.set_stroke(color, width=0, opacity=0.0)
                cone.set_shade_in_3d(False)

                # tatsächliche Spitze des Kegels finden:
                # Das ist der Punkt mit maximaler Projektion auf die Achsenrichtung.
                points = cone.get_all_points()
                projections = points @ direction
                current_tip = points[np.argmax(projections)]

                # Kegel so verschieben, dass seine Spitze exakt am Achsenende sitzt.
                cone.shift(tip_position - current_tip)

                return cone

            x_dir = np.array([0.0, -1.0, 0.0])
            y_dir = np.array([1.0, 0.0, 0.0])
            z_dir = np.array([0.0, 0.0, 1.0])

            x_end = origin + length * x_dir 
            y_end = origin + length * y_dir
            z_end = origin + length * z_dir
            x_line_end = x_end - cone_height * x_dir
            y_line_end = y_end - cone_height * y_dir
            z_line_end = z_end - cone_height * z_dir

            x_axis = Line3D(
                start=origin,
                end=x_line_end,
                color=color_coordinate_system,
                thickness=axis_thickness,
            )

            y_axis = Line3D(
                start=origin,
                end=y_line_end,
                color=color_cfg,
                thickness=axis_thickness,
            )

            z_axis = Line3D(
                start=origin,
                end=z_line_end,
                color=color_coordinate_system,
                thickness=axis_thickness,
            )
            x_axis.set_shade_in_3d(False)
            y_axis.set_shade_in_3d(False)
            z_axis.set_shade_in_3d(False)
            
            x_tip = make_arrow_tip(x_end, x_dir, color_coordinate_system)
            y_tip = make_arrow_tip(y_end, y_dir, color_cfg)
            z_tip = make_arrow_tip(z_end, z_dir, color_coordinate_system)

            x_label = MathTex("x", font_size=40, color=color_coordinate_system)
            x_label.move_to(origin + (length + label_shift + cone_height) * x_dir)

            y_label = MathTex("y", font_size=40, color=color_cfg)
            y_label.move_to(origin + (length + label_shift + cone_height) * y_dir)

            z_label = MathTex("z", font_size=40, color=color_coordinate_system)
            z_label.move_to(origin + (length + label_shift + cone_height) * z_dir)

            axes = VGroup(
                x_axis,
                y_axis,
                z_axis,
                x_tip,
                y_tip,
                z_tip,
            )

            labels = (x_label, y_label, z_label)

            return axes, labels
        
        lab_axes, lab_axis_labels = make_lab_axes(
            origin=lab_axes_origin,
            length=1.15,
        )
        self.add(lab_axes)
        self.add_fixed_orientation_mobjects(*lab_axis_labels)
        
        he_label = Text("He", font_size=36, color=color_HE, fill_opacity=0.7)
        he_label.move_to(droplet_center + np.array([-0.25, -0.75, 0.85]))

        self.add_fixed_orientation_mobjects(he_label)

        # ============================================================
        # Add objects
        # ============================================================

        self.add(pulse)
        self.add(droplet)
        self.add(rotation_circle)
        self.add(molecule_obj)
        self.add(he_label)
        # ============================================================
        # Animate
        # ============================================================

        self.play(
            t.animate.set_value(t_max),
            run_time=1,
            rate_func=linear,
        )

        pulse.clear_updaters()

        self.wait(1)