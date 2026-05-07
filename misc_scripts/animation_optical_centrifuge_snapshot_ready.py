from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from manim import *

from base_core.math.enums import AngleUnit, CartesianAxis
from base_core.math.models import Angle, Points3D
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


# Render animation:
#   manim -pqh animation_optical_centrifuge_snapshot_ready.py PhysicalOpticalCentrifuge3D
#
# Render one high-resolution snapshot at a chosen animation time:
#   MANIM_SNAPSHOT_ONLY=1 MANIM_SNAPSHOT_TIME=1.5 \
#   manim -s -r 7680,4320 animation_optical_centrifuge_snapshot_ready.py PhysicalOpticalCentrifuge3D
#
# Optional high-detail geometry for still images:
#   MANIM_SNAPSHOT_ONLY=1 MANIM_HIGH_DETAIL=1 MANIM_SNAPSHOT_TIME=1.5 \
#   manim -s -r 7680,4320 animation_optical_centrifuge_snapshot_ready.py PhysicalOpticalCentrifuge3D


@dataclass(frozen=True)
class RenderMode:
    snapshot_only: bool = os.getenv("MANIM_SNAPSHOT_ONLY", "0") == "1"
    snapshot_time: float = float(os.getenv("MANIM_SNAPSHOT_TIME", "0.0"))
    high_detail: bool = os.getenv("MANIM_HIGH_DETAIL", "0") == "1"


@dataclass(frozen=True)
class SceneLayout:
    t_min: float = -6.0
    t_max: float = 6.0
    seconds_per_manim_unit: float = 80e-12
    propagation_length: float = 9.0
    droplet_center: np.ndarray = field(default_factory=lambda: np.array([2.2, 0.0, 0.0]))
    molecule_length: float = 1.45
    pulse_start_x: float = -5.2
    ribbon_width: float = 1.35
    ribbon_opacity: float = 0.42
    detector_size: float = 3.15

    @property
    def pulse_end_x(self) -> float:
        return self.droplet_center[0] + 4.0

    @property
    def molecule_center(self) -> np.ndarray:
        return self.droplet_center

    @property
    def detector_center(self) -> np.ndarray:
        return self.molecule_center + np.array([0.0, 0.0, -2.25])

    @property
    def lab_axes_origin(self) -> np.ndarray:
        return np.array([self.pulse_start_x, 0.0, 0.0])

    @property
    def xi_min(self) -> float:
        return -self.propagation_length / 2.0

    @property
    def xi_max(self) -> float:
        return self.propagation_length / 2.0


@dataclass(frozen=True)
class SceneColors:
    background: ManimColor = WHITE
    coordinate_system: ManimColor = DARKER_GREY
    cfg: ManimColor = PINK
    helium: ManimColor = DARK_BLUE
    atom_center: ManimColor = GREY
    atom_outer: ManimColor = RED_E
    bond: ManimColor = LIGHTER_GREY
    ring: ManimColor = GRAY
    e_field: ManimColor = RED
    droplet: ManimColor = TEAL_E
    plot: ManimColor = DARKER_GREY
    plot_line: ManimColor = GRAY_BROWN
    detector: ManimColor = PURPLE_E
    detector_border: ManimColor = GREY_BROWN
    angle_line: ManimColor = GRAY_BROWN
    angle: ManimColor = GRAY_D
    
    @staticmethod
    def dark() -> "SceneColors":
        return SceneColors(
            background=BLACK,
            coordinate_system=WHITE,
            cfg=PURPLE,
            helium=TEAL_A,
            atom_center=GREY,
            atom_outer=BLUE_C,
            bond=WHITE,
            ring=ORANGE,
            e_field=RED,
            droplet=TEAL_E,
            plot=WHITE,
            plot_line=YELLOW,
            detector=BLUE_E,
            detector_border=WHITE,
            angle_line=GREEN,
            angle=BLACK,
        )

    @staticmethod
    def light() -> "SceneColors":
        return SceneColors()


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


def rotate_point_around_z(point: np.ndarray, angle: float, center: np.ndarray) -> np.ndarray:
    p = point - center
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return center + rot @ p


def make_arrow_tip(
    tip_position: np.ndarray,
    direction: np.ndarray,
    color: ManimColor | str,
    radius: float,
    height: float,
    resolution: tuple[int, int] = (24, 8),
) -> Cone:
    direction = direction / np.linalg.norm(direction)

    cone = Cone(
        base_radius=radius,
        height=height,
        direction=direction,
        resolution=resolution,
    )
    cone.set_fill(color, opacity=1.0)
    cone.set_stroke(color, width=0, opacity=0.0)
    cone.set_shade_in_3d(False)

    points = cone.get_all_points()
    projections = points @ direction
    current_tip = points[np.argmax(projections)]
    cone.shift(tip_position - current_tip)
    return cone


def make_axis_aligned_histogram_image_and_c2t(
    molecule_axis: np.ndarray,
    n_points: int = 30_000,
    bins: int = 160,
    cmap_name: str = "turbo",
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(1)

    axis = np.asarray(molecule_axis, dtype=float)
    axis /= np.linalg.norm(axis)

    cos_theta = rng.uniform(-1.0, 1.0, n_points)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_points)
    
    r_mean = 0.8
    r_std = 0.1   # ca. 95% der Werte liegen zwischen 0.6 und 1.0

    r = rng.normal(loc=r_mean, scale=r_std, size=n_points)
    r = np.clip(r, 0.6, 1.0)

    points_3d = Points3D.from_spherical(r=r, theta=theta, phi=phi)
    vectors = np.column_stack([points_3d.x, points_3d.y, points_3d.z])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    # This is a simple axis-aligned, |Y_1^0|^2-like angular distribution.
    rho = np.abs(vectors @ axis) ** 8

    projected = points_3d.project_to_plane(CartesianAxis.X | CartesianAxis.Z)
    x = projected.x
    y = projected.y

    r2 = x**2 + y**2
    valid = r2 > 1e-12

    cos2_theta_2d = np.zeros_like(r2)
    cos2_theta_2d[valid] = y[valid] ** 2 / r2[valid]
    c2t = float(np.sum(rho[valid] * cos2_theta_2d[valid]) / np.sum(rho[valid]))

    matrix, _, _ = np.histogram2d(
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

    rgba = plt.get_cmap(cmap_name)(data)
    rgba[..., 3] = np.clip(1.6 * data, 0.0, 1.0)
    return (255 * rgba).astype(np.uint8), c2t


class PhysicalOpticalCentrifuge3D(ThreeDScene):
    def construct(self) -> None:
        
        # manim -s -r 7680,4320 misc_scripts/animation_optical_centrifuge_snapshot_ready.py PhysicalOpticalCentrifuge3D
        #mode = RenderMode(snapshot_only=True, snapshot_time= 1, high_detail=True)
        
        # manim -p -r 3840,2160 --fps 60 misc_scripts/animation_optical_centrifuge_snapshot_ready.py PhysicalOpticalCentrifuge3D
        # manim -pql misc_scripts/animation_optical_centrifuge_snapshot_ready.py PhysicalOpticalCentrifuge3D
        mode = RenderMode(snapshot_only=False, snapshot_time= 0, high_detail=True)
        
        layout = SceneLayout()
        colors = SceneColors().dark()
        geometry = self._geometry_resolution(mode)

        self.camera.background_color = colors.background
        self.set_camera_orientation(phi=68 * DEGREES, theta=-55 * DEGREES)

        camera = self.renderer.camera
        if hasattr(camera, "light_source"):
            camera.light_source.move_to(3 * OUT + 4 * LEFT + 5 * UP)

        cfg = create_centrifuge_for_animation()
        z_R_extra = Length(0.65, Prefix.MILLI)
        z_L_extra = Length(0)

        initial_time = np.clip(mode.snapshot_time, layout.t_min, layout.t_max) if mode.snapshot_only else layout.t_min
        time_tracker = ValueTracker(float(initial_time))

        model = self._build_model_functions(
            cfg=cfg,
            z_R_extra=z_R_extra,
            z_L_extra=z_L_extra,
            layout=layout,
        )

        histogram_provider = HistogramProvider(
            molecule_direction=model["molecule_direction"],
            field_strength_at_molecule=model["field_strength_at_molecule"],
        )

        pulse = self._make_pulse(layout, colors, geometry, model, time_tracker)
        e_label = self._make_e_field_label(layout, colors, model, time_tracker)
        detector_group, theta_label = self._make_detector(layout, colors, histogram_provider, time_tracker)
        plot_static, c2t_curve, c2t_dot = self._make_c2t_plot(layout, colors, histogram_provider, time_tracker)
        droplet = self._make_droplet(layout, colors, geometry, time_tracker)
        molecule = self._make_molecule(layout, colors, geometry, model, time_tracker)
        rotation_circle = self._make_rotation_circle(layout, colors)
        lab_axes, lab_axis_labels = self._make_lab_axes(layout.lab_axes_origin, colors)
        helium_label = self._make_helium_label(layout, colors)

        self.add(pulse, detector_group, droplet, rotation_circle, molecule, lab_axes)
        self.add_fixed_orientation_mobjects(e_label, theta_label, helium_label, *lab_axis_labels)
        self.add_fixed_in_frame_mobjects(plot_static, c2t_curve, c2t_dot)

        if mode.snapshot_only:
            self.update_mobjects(0)
            self.wait(1 / self.camera.frame_rate)
            return

        self.play(
            time_tracker.animate.set_value(layout.t_max),
            run_time=15,
            rate_func=linear,
        )
        pulse.clear_updaters()
        self.wait(1)

    def _geometry_resolution(self, mode: RenderMode) -> dict[str, tuple[int, int]]:
        if mode.high_detail:
            return {
                "centrifuge": (24, 240),
                "sphere": (24, 12),
                "droplet": (48, 24),
            }

        return {
            "centrifuge": (12, 120),
            "sphere": (10, 5),
            "droplet": (24, 12),
        }

    def _build_model_functions(
        self,
        cfg: OpticalCentrifuge,
        z_R_extra: Length,
        z_L_extra: Length,
        layout: SceneLayout,
    ) -> dict[str, object]:
        sample_xis = np.linspace(layout.xi_min, layout.xi_max, 800)
        sample_times = np.array([-xi * layout.seconds_per_manim_unit for xi in sample_xis])
        sample_intensity = np.asarray(cfg.intensity(sample_times, z_R_extra, z_L_extra), dtype=float)
        intensity_norm = float(np.max(sample_intensity))
        if intensity_norm <= 0:
            raise ValueError("Centrifuge intensity is zero in the sampled animation window.")

        def pulse_center_x(anim_time: float) -> float:
            progress = (anim_time - layout.t_min) / (layout.t_max - layout.t_min)
            return layout.pulse_start_x + progress * (layout.pulse_end_x - layout.pulse_start_x)

        def pulse_coordinate(x: float, anim_time: float) -> float:
            return x - pulse_center_x(anim_time)

        def pulse_time_from_xi(xi: float) -> float:
            return -xi * layout.seconds_per_manim_unit

        def normalized_field_amplitude_from_pulse_time(pulse_time: float) -> float:
            intensity = float(cfg.intensity(pulse_time, z_R_extra, z_L_extra))
            return np.sqrt(max(intensity, 0.0) / intensity_norm)

        def field_angle_from_pulse_time(pulse_time: float) -> float:
            return float(cfg.polarization_angle(pulse_time, z_R_extra, z_L_extra))

        def field_direction_from_angle(theta: float) -> np.ndarray:
            return np.array([0.0, np.cos(theta), np.sin(theta)])

        def normalized_amplitude_at_x(x: float, anim_time: float) -> float:
            xi = pulse_coordinate(x, anim_time)
            return normalized_field_amplitude_from_pulse_time(pulse_time_from_xi(xi))

        def field_angle_at_x(x: float, anim_time: float) -> float:
            xi = pulse_coordinate(x, anim_time)
            return field_angle_from_pulse_time(pulse_time_from_xi(xi))

        def field_direction_at_x(x: float, anim_time: float) -> np.ndarray:
            return field_direction_from_angle(field_angle_at_x(x, anim_time))

        def field_strength_at_molecule(anim_time: float) -> float:
            return normalized_amplitude_at_x(layout.droplet_center[0], anim_time)

        def field_direction_at_molecule(anim_time: float) -> np.ndarray:
            return field_direction_at_x(layout.droplet_center[0], anim_time)

        def centrifuge_frequency_at_molecule(anim_time: float) -> float:
            xi = pulse_coordinate(layout.droplet_center[0], anim_time)
            pulse_time = pulse_time_from_xi(xi)
            return float(cfg.centrifuge_frequency(pulse_time, z_R_extra, z_L_extra))

        def molecule_direction(anim_time: float) -> np.ndarray:
            if centrifuge_frequency_at_molecule(anim_time) > 0:
                return np.array([0.0, 0.0, 1.0])
            return field_direction_at_molecule(anim_time)

        return {
            "pulse_center_x": pulse_center_x,
            "pulse_time_from_xi": pulse_time_from_xi,
            "normalized_field_amplitude_from_pulse_time": normalized_field_amplitude_from_pulse_time,
            "field_angle_from_pulse_time": field_angle_from_pulse_time,
            "field_direction_from_angle": field_direction_from_angle,
            "field_strength_at_molecule": field_strength_at_molecule,
            "molecule_direction": molecule_direction,
        }

    def _make_pulse(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        model: dict[str, object],
        time_tracker: ValueTracker,
    ) -> VGroup:
        pulse_time_from_xi = model["pulse_time_from_xi"]
        field_angle_from_pulse_time = model["field_angle_from_pulse_time"]
        normalized_field_amplitude_from_pulse_time = model["normalized_field_amplitude_from_pulse_time"]
        field_direction_from_angle = model["field_direction_from_angle"]
        pulse_center_x = model["pulse_center_x"]

        surface = Surface(
            lambda u, xi: np.array(
                [
                    xi,
                    u
                    * layout.ribbon_width
                    * normalized_field_amplitude_from_pulse_time(pulse_time_from_xi(xi))
                    * np.cos(field_angle_from_pulse_time(pulse_time_from_xi(xi))),
                    u
                    * layout.ribbon_width
                    * normalized_field_amplitude_from_pulse_time(pulse_time_from_xi(xi))
                    * np.sin(field_angle_from_pulse_time(pulse_time_from_xi(xi))),
                ]
            ),
            u_range=[-0.5, 0.5],
            v_range=[layout.xi_min, layout.xi_max],
            resolution=geometry["centrifuge"],
            fill_opacity=layout.ribbon_opacity,
            checkerboard_colors=[ManimColor(colors.cfg), ManimColor(colors.cfg)],
        )
        surface.set_style(stroke_width=0, stroke_opacity=0, fill_opacity=layout.ribbon_opacity)
        surface.set_shade_in_3d(True)

        xi = layout.xi_max
        theta = field_angle_from_pulse_time(pulse_time_from_xi(xi))
        direction = field_direction_from_angle(theta)
        direction /= np.linalg.norm(direction)

        length = 0.85
        cone_height = 0.20
        center = np.array([xi, 0.0, 0.0])
        start_tip = center - 0.5 * length * direction
        end_tip = center + 0.5 * length * direction
        start_line = start_tip + cone_height * direction
        end_line = end_tip - cone_height * direction

        shaft = Line3D(start=start_line, end=end_line, color=colors.e_field, thickness=0.045)
        shaft.set_shade_in_3d(False)
        e_vector = VGroup(
            shaft,
            make_arrow_tip(start_tip, -direction, colors.e_field, radius=0.09, height=cone_height),
            make_arrow_tip(end_tip, direction, colors.e_field, radius=0.09, height=cone_height),
        )

        pulse = VGroup(surface, e_vector)

        def update_pulse(mob: Mobject) -> None:
            current_time = time_tracker.get_value()
            mob.move_to(np.array([pulse_center_x(current_time), 0.0, 0.0]))

        pulse.add_updater(update_pulse)
        return pulse

    def _make_e_field_label(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        model: dict[str, object],
        time_tracker: ValueTracker,
    ) -> MathTex:
        pulse_center_x = model["pulse_center_x"]
        label = MathTex(r"\vec{E}_{\mathrm{CFG}}", font_size=34, color=colors.e_field)

        def update_label(mob: MathTex) -> None:
            current_time = time_tracker.get_value()
            mob.move_to(np.array([pulse_center_x(current_time) + layout.xi_max, 0.0, 0.75]))

        label.add_updater(update_label)
        return label

    def _make_detector(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        histogram_provider: "HistogramProvider",
        time_tracker: ValueTracker,
    ) -> tuple[Group, MathTex]:
        detector_center = layout.detector_center
        detector_size = layout.detector_size
        z_offset = 0.04

        detector_plate = Square(side_length=detector_size)
        detector_plate.set_fill(colors.detector, opacity=0.65)
        detector_plate.set_stroke(colors.detector_border, width=2)
        detector_plate.move_to(detector_center)

        def make_histogram_mobject() -> ImageMobject:
            image_array, _ = histogram_provider.frame_data(time_tracker.get_value())
            image = ImageMobject(image_array)
            image.height = detector_size * 0.94
            image.move_to(detector_center + np.array([0.0, 0.0, 0.01]))
            return image

        histogram_image = always_redraw(make_histogram_mobject)

        reference_line = Line(
            detector_center + np.array([-detector_size * 0.48, 0.0, z_offset]),
            detector_center + np.array([detector_size * 0.48, 0.0, z_offset]),
            color=colors.angle,
            stroke_width=4,
        )
        vertical_line = Line(
            detector_center + np.array([0.0, -detector_size * 0.48, z_offset + 0.002]),
            detector_center + np.array([0.0, detector_size * 0.48, z_offset + 0.002]),
            color=colors.cfg,
            stroke_width=2,
        )

        theta_2d = 35 * DEGREES
        theta_line = PI - theta_2d
        line_length = detector_size * 0.45
        angle_line = Line(
            detector_center + np.array([0.0, 0.0, z_offset + 0.004]),
            detector_center
            + line_length * np.array([np.cos(theta_line), np.sin(theta_line), 0.0])
            + np.array([0.0, 0.0, z_offset + 0.004]),
            color=colors.angle_line,
            stroke_width=4,
        )

        arc_radius = 0.62
        theta_arc = Arc(radius=arc_radius, start_angle=PI, angle=-theta_2d, color=colors.angle, stroke_width=4)
        theta_arc.move_arc_center_to(detector_center + np.array([0.0, 0.0, z_offset + 0.006]))

        arc_mid_angle = PI - theta_2d / 2
        label_anchor = detector_center + arc_radius * 0.7 * np.array(
            [np.cos(arc_mid_angle), np.sin(arc_mid_angle), 0.0]
        )
        label_line = DashedLine(
            label_anchor,
            label_anchor + np.array([0, 0.4, 0.38]),
            color=colors.angle_line,
            stroke_width=2,
            fill_opacity=0.6,
        )
        theta_label_pos = rotate_point_around_z(
            label_anchor + np.array([0, 0.4, 0.6]),
            90 * DEGREES,
            detector_center,
        )
        theta_label = MathTex(r"\theta_{\mathrm{2D}}", font_size=36, color=colors.angle)
        theta_label.move_to(theta_label_pos)

        detector_group = Group(
            detector_plate,
            histogram_image,
            reference_line,
            vertical_line,
            angle_line,
            theta_arc,
            label_line,
        )
        detector_group.rotate(90 * DEGREES, axis=OUT, about_point=detector_center)
        return detector_group, theta_label

    def _make_c2t_plot(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        histogram_provider: "HistogramProvider",
        time_tracker: ValueTracker,
    ) -> tuple[VGroup, VMobject, Dot]:
        plot_width = 2.7
        plot_height = 1.0
        label_scaling = 6
        plot_x_min = layout.t_min
        plot_x_max = layout.t_max
        plot_y_min = 0.45
        plot_y_max = 1.00
        plot_origin = np.array([-4.0, -2.2, 0.0])

        def plot_point_2d(x_value: float, y_value: float) -> np.ndarray:
            x_frac = (x_value - plot_x_min) / (plot_x_max - plot_x_min)
            y_frac = (y_value - plot_y_min) / (plot_y_max - plot_y_min)
            return plot_origin + np.array(
                [
                    plot_width * np.clip(x_frac, 0.0, 1.0),
                    plot_height * np.clip(y_frac, 0.0, 1.0),
                    0.0,
                ]
            )

        plot_x_axis = Line(
            plot_point_2d(plot_x_min, plot_y_min),
            plot_point_2d(plot_x_max, plot_y_min),
            color=colors.plot,
            stroke_width=2,
        )
        plot_y_axis = Line(
            plot_point_2d(plot_x_min, plot_y_min),
            plot_point_2d(plot_x_min, plot_y_max),
            color=colors.plot,
            stroke_width=2,
        )
        plot_baseline = DashedLine(
            plot_point_2d(plot_x_min, 0.5),
            plot_point_2d(plot_x_max, 0.5),
            color=colors.plot,
            stroke_width=1.5,
            dash_length=0.05,
        )
        plot_title = MathTex(
            r"\langle \cos^2\theta_{\mathrm{2D}}\rangle",
            font_size=24 + label_scaling,
            color=colors.plot,
        )
        plot_title.move_to(plot_origin + np.array([0.0, plot_height + 0.28, 0.0]))
        plot_x_label = MathTex(r"\mathrm{probe\ delay}", font_size=18 + label_scaling, color=colors.plot)
        plot_x_label.move_to(plot_origin + np.array([plot_width * 0.58, -0.22, 0.0]))
        plot_y05_label = MathTex(r"0.5", font_size=16 + label_scaling, color=colors.plot)
        plot_y05_label.move_to(plot_point_2d(plot_x_min, 0.5) + np.array([-0.22, 0.0, 0.0]))
        plot_y1_label = MathTex(r"1.0", font_size=16 + label_scaling, color=colors.plot)
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

        curve = VMobject()
        curve.set_stroke(colors.plot_line, width=3)
        dot = Dot(point=plot_point_2d(layout.t_min, 0.5), radius=0.035, color=colors.plot_line)

        def sampled_curve_points(current_time: float) -> list[np.ndarray]:
            current_time = float(np.clip(current_time, layout.t_min, layout.t_max))
            sample_times = np.linspace(layout.t_min, current_time, 80)
            if current_time > layout.t_min:
                sample_times = np.unique(np.append(sample_times, current_time))
            return [plot_point_2d(sample_time, histogram_provider.c2t_at_time(sample_time)) for sample_time in sample_times]

        def update_curve(mob: VMobject) -> None:
            points = sampled_curve_points(time_tracker.get_value())
            if len(points) < 2:
                mob.set_points([])
                return
            mob.set_points_as_corners(points)
            mob.set_stroke(colors.plot_line, width=3)

        def update_dot(mob: Dot) -> None:
            current_time = float(time_tracker.get_value())
            mob.move_to(plot_point_2d(current_time, histogram_provider.c2t_at_time(current_time)))

        curve.add_updater(update_curve)
        dot.add_updater(update_dot)
        return plot_static, curve, dot

    def _make_droplet(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        time_tracker: ValueTracker,
    ) -> Mobject:
        def blobby_radius(theta: float, phi: float, phase: float) -> float:
            return 1.8 * (
                1.0
                + 0.09 * np.sin(3 * theta + 1.2 * phase) * np.sin(2 * phi)
                + 0.070 * np.cos(5 * theta - 0.8 * phase) * np.sin(phi) ** 2
                + 0.06 * np.sin(4 * phi + 0.6 * phase)
            )

        def blobby_surface(u: float, v: float, center: np.ndarray, phase: float) -> np.ndarray:
            theta = u
            phi = v
            radius = blobby_radius(theta, phi, phase)
            point = np.array(
                [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                ]
            )
            return center + point

        return always_redraw(
            lambda: Surface(
                lambda u, v: blobby_surface(u, v, layout.droplet_center, phase=0.9 * time_tracker.get_value()),
                u_range=[0, TAU],
                v_range=[0, PI],
                resolution=geometry["droplet"],
            )
            .set_fill(colors.droplet, opacity=0.16)
            .set_style(stroke_width=0, stroke_opacity=0, fill_opacity=0.16)
            .scale([1.2, 1.0, 0.9])
            .set_shade_in_3d(True)
        )

    def _make_molecule(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        model: dict[str, object],
        time_tracker: ValueTracker,
    ) -> Mobject:
        molecule_direction = model["molecule_direction"]

        def molecule() -> VGroup:
            direction = molecule_direction(time_tracker.get_value())
            direction = direction / np.linalg.norm(direction)
            p1 = layout.molecule_center - 0.5 * layout.molecule_length * direction
            p2 = layout.molecule_center + 0.5 * layout.molecule_length * direction
            center_radius = 0.14

            bond1 = Line3D(
                start=p1,
                end=layout.molecule_center - center_radius * direction,
                color=colors.bond,
                thickness=0.03,
            )
            bond2 = Line3D(
                start=layout.molecule_center + center_radius * direction,
                end=p2,
                color=colors.bond,
                thickness=0.03,
            )

            atom1 = Sphere(center=p1, radius=0.17, resolution=geometry["sphere"])
            atom2 = Sphere(center=p2, radius=0.17, resolution=geometry["sphere"])
            center_atom = Sphere(center=layout.molecule_center, radius=center_radius, resolution=geometry["sphere"])

            for atom, color in [(atom1, colors.atom_outer), (atom2, colors.atom_outer), (center_atom, colors.atom_center)]:
                atom.set_fill(color, opacity=1.0)
                atom.set_style(stroke_width=0, stroke_opacity=0)
                atom.set_shade_in_3d(True)

            return VGroup(bond1, bond2, atom1, atom2, center_atom)

        return always_redraw(molecule)

    def _make_rotation_circle(self, layout: SceneLayout, colors: SceneColors) -> Circle:
        circle = Circle(radius=0.95, color=colors.ring, stroke_width=4, stroke_opacity=0.75)
        circle.rotate(PI / 2, axis=UP)
        circle.move_to(layout.molecule_center)
        return circle

    def _make_lab_axes(
        self,
        origin: np.ndarray,
        colors: SceneColors,
        length: float = 1.15,
        label_shift: float = 0.18,
    ) -> tuple[VGroup, tuple[MathTex, MathTex, MathTex]]:
        axis_thickness = 0.035
        cone_radius = 0.1
        cone_height = 0.22

        x_dir = np.array([0.0, -1.0, 0.0])
        y_dir = np.array([1.0, 0.0, 0.0])
        z_dir = np.array([0.0, 0.0, 1.0])

        x_end = origin + length * x_dir
        y_end = origin + length * y_dir
        z_end = origin + length * z_dir

        x_axis = Line3D(start=origin, end=x_end - cone_height * x_dir, color=colors.coordinate_system, thickness=axis_thickness)
        y_axis = Line3D(start=origin, end=y_end - cone_height * y_dir, color=colors.cfg, thickness=axis_thickness)
        z_axis = Line3D(start=origin, end=z_end - cone_height * z_dir, color=colors.coordinate_system, thickness=axis_thickness)

        for axis in (x_axis, y_axis, z_axis):
            axis.set_shade_in_3d(False)

        axes = VGroup(
            x_axis,
            y_axis,
            z_axis,
            make_arrow_tip(x_end, x_dir, colors.coordinate_system, radius=cone_radius, height=cone_height),
            make_arrow_tip(y_end, y_dir, colors.cfg, radius=cone_radius, height=cone_height),
            make_arrow_tip(z_end, z_dir, colors.coordinate_system, radius=cone_radius, height=cone_height),
        )

        x_label = MathTex("x", font_size=40, color=colors.coordinate_system)
        y_label = MathTex("y", font_size=40, color=colors.cfg)
        z_label = MathTex("z", font_size=40, color=colors.coordinate_system)

        x_label.move_to(origin + (length + label_shift + cone_height) * x_dir)
        y_label.move_to(origin + (length + label_shift + cone_height) * y_dir)
        z_label.move_to(origin + (length + label_shift + cone_height) * z_dir)

        return axes, (x_label, y_label, z_label)

    def _make_helium_label(self, layout: SceneLayout, colors: SceneColors) -> Text:
        label = Text("He", font_size=36, color=colors.helium, fill_opacity=0.7)
        label.move_to(layout.droplet_center + np.array([0.25, 0.8, 1]))
        return label


class HistogramProvider:
    def __init__(
        self,
        molecule_direction,
        field_strength_at_molecule,
        n_points: int = 30_000,
        bins: int = 160,
    ) -> None:
        self.molecule_direction = molecule_direction
        self.field_strength_at_molecule = field_strength_at_molecule
        self.n_points = n_points
        self.bins = bins

    def _cache_key(self, anim_time: float) -> float:
        return round(float(anim_time), 4)

    @lru_cache(maxsize=512)
    def _frame_data_cached(self, anim_time: float) -> tuple[np.ndarray, float]:
        axis = self.molecule_direction(anim_time)
        axis_for_hist = np.array([axis[0], axis[2], -axis[1]])

        image_array, c2t = make_axis_aligned_histogram_image_and_c2t(
            axis_for_hist,
            n_points=self.n_points,
            bins=self.bins,
        )

        field_strength = np.clip(self.field_strength_at_molecule(anim_time), 0.0, 1.0)
        c2t = 0.5 + field_strength * (c2t - 0.5)
        return image_array, float(c2t)

    def frame_data(self, anim_time: float) -> tuple[np.ndarray, float]:
        return self._frame_data_cached(self._cache_key(anim_time))

    def c2t_at_time(self, anim_time: float) -> float:
        _, c2t = self.frame_data(anim_time)
        return c2t
