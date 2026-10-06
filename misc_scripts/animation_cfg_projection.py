from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from manim import *

from base_core.math.enums import AngleUnit
from base_core.math.models import Angle
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


# Quick preview:
#   manim -pql misc_scripts/animation_cfg_projection.py CfgProjectionThroughPBS
#
# Render one high-resolution snapshot:
#   manim -s -r 7680,4320 misc_scripts/animation_cfg_projection.py CfgProjectionThroughPBS
#
# Optional high-detail geometry for still images:
#   MANIM_HIGH_DETAIL=1 manim -s -r 7680,4320 misc_scripts/animation_cfg_projection.py CfgProjectionThroughPBS
#
# Switch between the two views by choosing the scene:
#   CfgProjectionThroughPBS  -> centrifuge split by the beam splitter into its projections
#   CfgWholeCentrifuge       -> the whole, unsplit centrifuge (no beam splitter)


class CfgView(Enum):
    WHOLE = "whole"
    PROJECTION = "projection"


@dataclass(frozen=True)
class RenderMode:
    high_detail: bool = os.getenv("MANIM_HIGH_DETAIL", "0") == "1"


@dataclass(frozen=True)
class SceneLayout:
    pulse_start_x: float = -7.2
    pbs_center: np.ndarray = field(default_factory=lambda: np.array([1.6, 0.0, 0.0]))
    pbs_size: float = 1.3
    ribbon_width: float = 1.35
    ribbon_opacity: float = 0.42
    seconds_per_manim_unit: float = 80e-12
    pulse_advance: float = 0.6
    intensity_cutoff: float = 0.3

    @property
    def lab_axes_origin(self) -> np.ndarray:
        return np.array([self.pulse_start_x, 0.0, 0.0])


@dataclass(frozen=True)
class SceneColors:
    background: ManimColor = WHITE
    coordinate_system: ManimColor = DARKER_GREY
    cfg: ManimColor = PINK
    cfg_reflected: ManimColor = PURPLE
    pbs: ManimColor = TEAL_E
    wall: ManimColor = GREY_BROWN
    e_field: ManimColor = RED

    @staticmethod
    def dark() -> "SceneColors":
        return SceneColors(
            background=BLACK,
            coordinate_system=WHITE,
            cfg=PINK,
            cfg_reflected=PURPLE_A,
            pbs=TEAL_A,
            wall=GREY_BROWN,
            e_field=RED,
        )

    @staticmethod
    def light() -> "SceneColors":
        return SceneColors()


def create_centrifuge_for_animation(pulse_duration: Time = Time(490, Prefix.PICO)) -> OpticalCentrifuge:
    left_chirp = AngularChirp(BETA_0 + DELTA_BETA * 0.4)
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


def make_double_arrow(
    center: np.ndarray,
    direction: np.ndarray,
    color: ManimColor | str,
    length: float = 0.85,
    cone_height: float = 0.20,
    cone_radius: float = 0.09,
    thickness: float = 0.045,
) -> VGroup:
    direction = direction / np.linalg.norm(direction)
    start_tip = center - 0.5 * length * direction
    end_tip = center + 0.5 * length * direction
    start_line = start_tip + cone_height * direction
    end_line = end_tip - cone_height * direction

    shaft = Line3D(start=start_line, end=end_line, color=color, thickness=thickness)
    shaft.set_shade_in_3d(False)

    return VGroup(
        shaft,
        make_arrow_tip(start_tip, -direction, color, radius=cone_radius, height=cone_height),
        make_arrow_tip(end_tip, direction, color, radius=cone_radius, height=cone_height),
    )


class CfgProjectionThroughPBS(ThreeDScene):
    view: CfgView = CfgView.PROJECTION

    def construct(self) -> None:
        mode = RenderMode()
        layout = SceneLayout()
        colors = SceneColors.light()
        geometry = self._geometry_resolution(mode)

        self.camera.background_color = colors.background
        self.set_camera_orientation(
            phi=68 * DEGREES,
            theta=-55 * DEGREES,
            zoom=0.85,
            frame_center=np.array([0.762, 0.226, 0.0]),
        )

        camera = self.renderer.camera
        if hasattr(camera, "light_source"):
            camera.light_source.move_to(3 * OUT + 4 * LEFT + 5 * UP)

        cfg = create_centrifuge_for_animation()
        z_R_extra = Length(0.65, Prefix.MILLI)
        z_L_extra = Length(0)

        amplitude, angle, s_start, s_end = self._build_model(cfg, z_R_extra, z_L_extra, layout)

        lab_axes, lab_axis_labels = self._make_lab_axes(layout.lab_axes_origin, colors)
        self.add(lab_axes)
        self.add_fixed_orientation_mobjects(*lab_axis_labels)

        if self.view is CfgView.WHOLE:
            self._add_whole_centrifuge(layout, colors, geometry, amplitude, angle, s_start, s_end)
        else:
            self._add_projection(layout, colors, geometry, amplitude, angle, s_start, s_end)

        self.wait(1 / self.camera.frame_rate)

    def _add_whole_centrifuge(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        amplitude,
        angle,
        s_start: float,
        s_end: float,
    ) -> None:
        beam = self._make_incoming_beam(layout, colors, geometry, amplitude, angle, s_start, s_end)
        e_vector, e_label = self._make_cfg_e_vector(layout, colors, angle, s_end)

        self.add(beam, e_vector)
        self.add_fixed_orientation_mobjects(e_label)

    def _add_projection(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        amplitude,
        angle,
        s_start: float,
        s_end: float,
    ) -> None:
        incoming_beam = self._make_incoming_beam(layout, colors, geometry, amplitude, angle, s_start)
        transmitted_beam = self._make_transmitted_beam(layout, colors, geometry, amplitude, angle, s_end)
        reflected_beam = self._make_reflected_beam(layout, colors, geometry, amplitude, angle, s_end)
        pbs, wall = self._make_pbs(layout, colors)
        transmitted_e_vector, transmitted_label = self._make_transmitted_e_vector(layout, colors, s_end)
        reflected_e_vector, reflected_label = self._make_reflected_e_vector(layout, colors, s_end)

        self.add(
            incoming_beam,
            transmitted_beam,
            reflected_beam,
            pbs,
            wall,
            transmitted_e_vector,
            reflected_e_vector,
        )
        self.add_fixed_orientation_mobjects(transmitted_label, reflected_label)

    def _geometry_resolution(self, mode: RenderMode) -> dict[str, tuple[int, int]]:
        if mode.high_detail:
            return {"centrifuge": (28, 320)}

        return {"centrifuge": (12, 120)}

    def _build_model(
        self,
        cfg: OpticalCentrifuge,
        z_R_extra: Length,
        z_L_extra: Length,
        layout: SceneLayout,
    ):
        def pulse_time(s: float) -> float:
            return (layout.pulse_advance - s) * layout.seconds_per_manim_unit

        # Find the envelope's peak and the two points where it has fallen to
        # layout.intensity_cutoff of the peak, by probing a wide, fine window
        # of physical time, independent of how the figure's s-domain is laid
        # out. The drawn beams are then trimmed to exactly this window so
        # both ends taper to the same (small) amplitude instead of an
        # arbitrary, visually mismatched cutoff.
        t_probe = np.linspace(-3000e-12, 3000e-12, 6001)
        intensity_probe = np.asarray(cfg.intensity(t_probe, z_R_extra, z_L_extra), dtype=float)
        intensity_max = float(np.max(intensity_probe))
        if intensity_max <= 0:
            raise ValueError("Centrifuge intensity is zero in the sampled window.")
        peak_index = int(np.argmax(intensity_probe))
        cutoff_intensity = layout.intensity_cutoff * intensity_max

        def crossing(indices: range) -> float:
            prev_t, prev_i = t_probe[indices[0]], intensity_probe[indices[0]]
            for idx in indices[1:]:
                t, i = t_probe[idx], intensity_probe[idx]
                if i <= cutoff_intensity:
                    frac = (prev_i - cutoff_intensity) / (prev_i - i)
                    return prev_t + frac * (t - prev_t)
                prev_t, prev_i = t, i
            return t_probe[indices[-1]]

        # Positive pulse_time is the not-yet-split (incoming) side, negative
        # pulse_time is the already-split (output) side, since pulse_time
        # decreases as s increases past pulse_advance.
        t_incoming_edge = crossing(range(peak_index, len(t_probe)))
        t_output_edge = crossing(range(peak_index, -1, -1))

        s_start = layout.pulse_advance - t_incoming_edge / layout.seconds_per_manim_unit
        s_end = layout.pulse_advance - t_output_edge / layout.seconds_per_manim_unit

        def amplitude(s: float) -> float:
            intensity = float(cfg.intensity(pulse_time(s), z_R_extra, z_L_extra))
            return np.sqrt(max(intensity, 0.0) / intensity_max)

        def angle(s: float) -> float:
            return float(cfg.polarization_angle(pulse_time(s), z_R_extra, z_L_extra))

        return amplitude, angle, s_start, s_end

    def _make_incoming_beam(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        amplitude,
        angle,
        s_start: float,
        s_stop: float = 0.0,
    ) -> Surface:
        def point(u: float, s: float) -> np.ndarray:
            a = amplitude(s)
            th = angle(s)
            return layout.pbs_center + np.array(
                [s, u * layout.ribbon_width * a * np.cos(th), u * layout.ribbon_width * a * np.sin(th)]
            )

        surface = Surface(
            point,
            u_range=[-0.5, 0.5],
            v_range=[s_start, s_stop],
            resolution=geometry["centrifuge"],
            fill_opacity=layout.ribbon_opacity,
            checkerboard_colors=[ManimColor(colors.cfg), ManimColor(colors.cfg)],
        )
        surface.set_style(stroke_width=0, stroke_opacity=0, fill_opacity=layout.ribbon_opacity)
        surface.set_shade_in_3d(True)
        return surface

    def _make_transmitted_beam(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        amplitude,
        angle,
        s_end: float,
    ) -> Surface:
        def point(u: float, s: float) -> np.ndarray:
            a = amplitude(s)
            th = angle(s)
            return layout.pbs_center + np.array([s, u * layout.ribbon_width * a * np.cos(th), 0.0])

        surface = Surface(
            point,
            u_range=[-0.5, 0.5],
            v_range=[0.0, s_end],
            resolution=geometry["centrifuge"],
            fill_opacity=layout.ribbon_opacity,
            checkerboard_colors=[ManimColor(colors.cfg), ManimColor(colors.cfg)],
        )
        surface.set_style(stroke_width=0, stroke_opacity=0, fill_opacity=layout.ribbon_opacity)
        surface.set_shade_in_3d(True)
        return surface

    def _make_reflected_beam(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        amplitude,
        angle,
        s_end: float,
    ) -> Surface:
        def point(u: float, s: float) -> np.ndarray:
            a = amplitude(s)
            th = angle(s)
            return layout.pbs_center + np.array([0.0, s, u * layout.ribbon_width * a * np.sin(th)])

        surface = Surface(
            point,
            u_range=[-0.5, 0.5],
            v_range=[0.0, s_end],
            resolution=geometry["centrifuge"],
            fill_opacity=layout.ribbon_opacity,
            checkerboard_colors=[ManimColor(colors.cfg_reflected), ManimColor(colors.cfg_reflected)],
        )
        surface.set_style(stroke_width=0, stroke_opacity=0, fill_opacity=layout.ribbon_opacity)
        surface.set_shade_in_3d(True)
        return surface

    def _make_pbs(self, layout: SceneLayout, colors: SceneColors) -> tuple[Mobject, Mobject]:
        half = layout.pbs_size / 2.0

        cube = Cube(side_length=layout.pbs_size)
        cube.move_to(layout.pbs_center)
        cube.set_fill(colors.pbs, opacity=0.16)
        cube.set_style(stroke_width=0.5, stroke_opacity=0.25, fill_opacity=0.16, stroke_color=colors.pbs)
        cube.set_shade_in_3d(True)

        # Diagonal dielectric interface: the plane that turns +x propagation
        # into +y propagation, i.e. the plane through the two opposite
        # vertical edges of the cube spanned by (1, 1, 0) and (0, 0, 1).
        c = layout.pbs_center
        wall = Polygon(
            c + np.array([-half, -half, -half]),
            c + np.array([half, half, -half]),
            c + np.array([half, half, half]),
            c + np.array([-half, -half, half]),
            color=colors.wall,
        )
        wall.set_fill(colors.wall, opacity=0.18)
        wall.set_stroke(colors.wall, width=1.2, opacity=0.4)
        wall.set_shade_in_3d(True)

        return cube, wall

    def _make_cfg_e_vector(self, layout: SceneLayout, colors: SceneColors, angle, s_end: float) -> tuple[VGroup, MathTex]:
        # Sits at the output end of the centrifuge, pointing along the local
        # polarization of the field there.
        s = s_end
        th = angle(s)
        direction = np.array([0.0, np.cos(th), np.sin(th)])
        center = layout.pbs_center + np.array([s, 0.0, 0.0])
        arrow = make_double_arrow(center, direction, colors.e_field)

        label = MathTex(r"\vec{E}_\mathrm{CFG}", font_size=44, color=colors.e_field)
        label.move_to(center + np.array([0.0, 0.0, 0.75]))
        return arrow, label

    def _make_transmitted_e_vector(self, layout: SceneLayout, colors: SceneColors, s_end: float) -> tuple[VGroup, MathTex]:
        s = 0.85 * s_end
        center = layout.pbs_center + np.array([s, 0.0, 0.0])
        arrow = make_double_arrow(center, np.array([0.0, 1.0, 0.0]), colors.cfg)

        label = MathTex(r"\vec{E}_x", font_size=44, color=colors.cfg)
        label.move_to(center + np.array([0.0, 0.75, 0.35]))
        return arrow, label

    def _make_reflected_e_vector(self, layout: SceneLayout, colors: SceneColors, s_end: float) -> tuple[VGroup, MathTex]:
        s = 0.85 * s_end
        center = layout.pbs_center + np.array([0.0, s, 0.0])
        arrow = make_double_arrow(center, np.array([0.0, 0.0, 1.0]), colors.cfg_reflected)

        label = MathTex(r"\vec{E}_y", font_size=44, color=colors.cfg_reflected)
        label.move_to(center + np.array([0.0, 0.0, 0.75]))
        return arrow, label

    def _make_lab_axes(
        self,
        origin: np.ndarray,
        colors: SceneColors,
        length: float = 1.15,
        label_shift: float = 0.18,
    ) -> tuple[VGroup, tuple[MathTex, MathTex, MathTex]]:
        # Same drawn frame as animation_optical_centrifuge.py, but with the
        # "y" and "z" labels swapped so that z is the beam-propagation axis
        # (matching the usual optics convention) and y is the vertical axis.
        # The arrow directions/colors are unchanged; only the two labels
        # trade places.
        axis_thickness = 0.035
        cone_radius = 0.1
        cone_height = 0.22

        x_dir = np.array([0.0, -1.0, 0.0])
        propagation_dir = np.array([1.0, 0.0, 0.0])
        vertical_dir = np.array([0.0, 0.0, 1.0])

        x_end = origin + length * x_dir
        propagation_end = origin + length * propagation_dir
        vertical_end = origin + length * vertical_dir

        x_axis = Line3D(start=origin, end=x_end - cone_height * x_dir, color=colors.coordinate_system, thickness=axis_thickness)
        propagation_axis = Line3D(start=origin, end=propagation_end - cone_height * propagation_dir, color=colors.cfg, thickness=axis_thickness)
        vertical_axis = Line3D(start=origin, end=vertical_end - cone_height * vertical_dir, color=colors.coordinate_system, thickness=axis_thickness)

        for axis in (x_axis, propagation_axis, vertical_axis):
            axis.set_shade_in_3d(False)

        axes = VGroup(
            x_axis,
            propagation_axis,
            vertical_axis,
            make_arrow_tip(x_end, x_dir, colors.coordinate_system, radius=cone_radius, height=cone_height),
            make_arrow_tip(propagation_end, propagation_dir, colors.cfg, radius=cone_radius, height=cone_height),
            make_arrow_tip(vertical_end, vertical_dir, colors.coordinate_system, radius=cone_radius, height=cone_height),
        )

        x_label = MathTex("x", font_size=72, color=colors.coordinate_system)
        z_label = MathTex("z", font_size=72, color=colors.cfg)
        y_label = MathTex("y", font_size=72, color=colors.coordinate_system)

        x_label.move_to(origin + (length + label_shift + cone_height) * x_dir)
        z_label.move_to(origin + (length + label_shift + cone_height) * propagation_dir)
        y_label.move_to(origin + (length + label_shift + cone_height) * vertical_dir)

        return axes, (x_label, y_label, z_label)


class CfgWholeCentrifuge(CfgProjectionThroughPBS):
    view: CfgView = CfgView.WHOLE
