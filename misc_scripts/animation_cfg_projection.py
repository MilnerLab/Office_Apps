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
    # The envelope peak sits at s ~ pulse_advance.
    pulse_advance: float = 0.6
    # Drawn extent of the beam along s (manim units). Fixed, rather than tied
    # to an intensity cutoff, so the corkscrew keeps its length and position
    # when the pulse duration or pulse_advance is changed.
    beam_start: float = -6.13
    beam_end: float = 7.31
    # Zoom inset (frame coordinates, relative to the camera's frame center).
    inset_center: np.ndarray = field(default_factory=lambda: np.array([-4.2, -2.4, 0.0]))
    inset_width: float = 5.3
    inset_height: float = 2.8
    inset_padding: float = 0.3
    # Gap between the zoomed drawing (incl. labels) and the box's lower edge.
    inset_bottom_gap: float = 0.12
    inset_label_space: float = 1.2
    # Length (along z) of the zoomed segment, trailing behind the E-vector.
    inset_window: float = 1.6
    # Minimum on-screen gap between the zoom target and the PBS outline.
    inset_target_margin: float = 0.12
    # The zoom target is the point nearest the envelope peak (but left of the
    # cube) where the field makes this angle with the lab x axis, so that
    # Theta_0 is readable.
    inset_theta_0_deg: float = 45.0
    inset_title: str = "Phase at max. Intensity"
    # Gap above and below the title; the zoomed drawing stays below it.
    inset_title_gap: float = 0.15

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


def create_centrifuge_for_animation(pulse_duration: Time = Time(320, Prefix.PICO)) -> OpticalCentrifuge:
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

        amplitude, angle, s_start, s_end, s_peak = self._build_model(cfg, z_R_extra, z_L_extra, layout)

        lab_axes, lab_axis_labels = self._make_lab_axes(layout.lab_axes_origin, colors)
        self.add(lab_axes)
        self.add_fixed_orientation_mobjects(*lab_axis_labels)

        if self.view is CfgView.WHOLE:
            self._add_whole_centrifuge(layout, colors, geometry, amplitude, angle, s_start, s_end)
        else:
            self._add_projection(layout, colors, geometry, amplitude, angle, s_start, s_end)
            self._add_zoom_inset(layout, colors, geometry, amplitude, angle, s_peak)

        self.wait(1 / self.camera.frame_rate)

    def _add_zoom_inset(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        geometry: dict[str, tuple[int, int]],
        amplitude,
        angle,
        s_peak: float,
    ) -> None:
        camera = self.renderer.camera
        camera.reset_rotation_matrix()
        rotation = camera.get_rotation_matrix()

        # Last point on the incoming beam that is visibly left of the cube's
        # on-screen outline; the zoom target must not lie beyond it.
        half = layout.pbs_size / 2.0
        cube_corners = layout.pbs_center + half * np.array(
            [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
        )
        cube_left = camera.project_points(cube_corners)[:, 0].min() - layout.inset_target_margin
        s_limit = 0.0
        while camera.project_point(layout.pbs_center + np.array([s_limit, 0.0, 0.0]))[0] > cube_left:
            s_limit -= 0.01

        def angle_to_x_deg(s: float) -> float:
            # Lab x is world -y; field direction is (0, cos, sin) in world coordinates.
            return float(np.degrees(np.arccos(abs(np.cos(angle(s))))))

        # Nearest point to the envelope peak (left of the cube) with the wanted Theta_0.
        s_high = min(s_peak + 1.0, s_limit)
        candidates = np.arange(s_high - 2.0, s_high, 0.002)
        matches = [s for s in candidates if abs(angle_to_x_deg(s) - layout.inset_theta_0_deg) < 1.0]
        s_target = min(matches, key=lambda s: abs(s - s_peak)) if matches else min(s_peak, s_limit)
        target = layout.pbs_center + np.array([s_target, 0.0, 0.0])
        target_2d = camera.project_point(target)

        # The segment fills the left part of the box, the label sits to its right.
        inset_center = camera.frame_center + layout.inset_center
        fit_width = layout.inset_width - layout.inset_label_space
        segment_center = inset_center + np.array([(fit_width - layout.inset_width) / 2.0, 0.0, 0.0])
        segment = self._make_incoming_beam(
            layout,
            colors,
            {"centrifuge": geometry["inset"]},
            amplitude,
            angle,
            s_target - layout.inset_window,
            s_target,
        )

        title = Tex(layout.inset_title, font_size=44, color=colors.coordinate_system)
        if title.width > layout.inset_width - 0.5:
            title.scale_to_fit_width(layout.inset_width - 0.5)
        label = MathTex(r"\vec{E}_\mathrm{CFG}", font_size=44, color=colors.e_field)

        # Magnify the segment's on-screen projection so that it fills the space
        # below the title, leaving room for the E-label hanging below the arrow.
        fit_height = (
            layout.inset_height
            - (title.height + 2.0 * layout.inset_title_gap)
            - (label.height / 2.0 + layout.inset_bottom_gap)
        )
        projected = camera.project_points(np.vstack([m.points for m in segment.family_members_with_points()]))[:, :2]
        lower, upper = projected.min(axis=0), projected.max(axis=0)
        anchor = np.array([*(lower + upper) / 2.0, 0.0])
        zoom = min(
            (fit_width - 2.0 * layout.inset_padding) / (upper - lower)[0],
            fit_height / (upper - lower)[1],
        )

        def to_inset(points: np.ndarray) -> np.ndarray:
            flat = segment_center + zoom * (camera.project_points(points) - anchor)
            flat[:, 2] = 0.0
            return flat

        def flatten(mob: Mobject) -> VGroup:
            # Bake the 3D shading and depth order in, then project into the inset.
            faces = mob.family_members_with_points()
            for face in faces:
                rgbas = camera.modified_rgbas(face, face.get_fill_rgbas())
                face.set_fill(rgb_to_color(rgbas[:, :3].mean(axis=0)), opacity=float(rgbas[:, 3].mean()))
            faces.sort(key=lambda face: (face.get_center() @ rotation.T)[2])
            for face in faces:
                face.points = to_inset(face.points)
                face.set_shade_in_3d(False)
            return VGroup(*faces)

        th = angle(s_target)
        direction = np.array([0.0, np.cos(th), np.sin(th)])
        e_length = layout.ribbon_width * amplitude(s_target)
        e_vector = make_double_arrow(
            target,
            direction,
            colors.e_field,
            length=e_length,
            cone_height=0.20 / zoom,
            cone_radius=0.09 / zoom,
            thickness=0.045 / zoom,
        )

        segment_2d = flatten(segment)
        e_vector_2d = flatten(e_vector)

        # Gray reference line along the lab x axis (world -y) through the
        # E-vector's center, and the angle Theta_0 between it and the E-vector.
        x_axis_dir = np.array([0.0, -1.0, 0.0])
        x_line_ends = to_inset(np.array([target - 0.65 * e_length * x_axis_dir, target + 0.65 * e_length * x_axis_dir]))
        x_line = Line(x_line_ends[0], x_line_ends[1], color=GREY, stroke_width=3)

        # Of the two acute (arrow-half, line-half) pairs, draw the angle on the
        # side facing away from the ribbon tail, i.e. towards the right of the inset.
        candidates = [(sign * direction, sign * np.sign(direction @ x_axis_dir or 1.0) * x_axis_dir) for sign in (1.0, -1.0)]
        e_half, x_half = max(candidates, key=lambda pair: to_inset(np.array([target + pair[0] + pair[1]]))[0, 0])
        theta_0 = np.arccos(np.clip(e_half @ x_half, -1.0, 1.0))
        sweep_dir = normalize(e_half - (e_half @ x_half) * x_half)
        arc_radius = 0.38 * e_length
        phis = np.linspace(0.0, theta_0, 40)
        arc_points = to_inset(target + arc_radius * (np.outer(np.cos(phis), x_half) + np.outer(np.sin(phis), sweep_dir)))
        theta_arc = VMobject(color=colors.coordinate_system, stroke_width=3)
        theta_arc.set_points_smoothly(arc_points)

        # Centre the label in front of the arc as drawn: just outside its
        # on-screen midpoint (by arc length), along the ray from the vertex.
        seg_lengths = np.linalg.norm(np.diff(arc_points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        arc_mid = np.array([np.interp(cumulative[-1] / 2.0, cumulative, arc_points[:, i]) for i in range(3)])
        vertex = to_inset(np.array([target]))[0]
        theta_label = MathTex(r"\alpha", font_size=40, color=colors.coordinate_system)
        theta_label.next_to(arc_mid, normalize(arc_mid - vertex), buff=0.08)

        # The angle sits by the upper arrow tip, so the label goes by the lower one.
        tips_2d = to_inset(np.array([target + 0.5 * e_length * direction, target - 0.5 * e_length * direction]))
        lower_tip = min(tips_2d, key=lambda p: p[1])
        label.next_to(lower_tip, RIGHT, buff=0.2)

        box = Rectangle(width=layout.inset_width, height=layout.inset_height)
        box.move_to(inset_center)
        box.set_fill(colors.background, opacity=1.0)
        box.set_stroke(BLACK, width=2.5)

        # Sit the zoomed drawing close to the box's lower edge.
        drawing = VGroup(segment_2d, x_line, theta_arc, e_vector_2d, label, theta_label)
        drawing.shift((box.get_bottom()[1] + layout.inset_bottom_gap - drawing.get_bottom()[1]) * UP)

        corners = [box.get_corner(c) for c in (UL, UR, DL, DR)]
        start = min(corners, key=lambda p: np.linalg.norm(p[:2] - target_2d[:2]))
        pointer = Line(start, np.array([target_2d[0], target_2d[1], 0.0]), color=BLACK, stroke_width=2.5)

        title.next_to(box.get_top(), DOWN, buff=layout.inset_title_gap)

        self.add_fixed_in_frame_mobjects(
            box, segment_2d, x_line, theta_arc, e_vector_2d, label, theta_label, title, pointer
        )

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
        pbs_label = self._make_pbs_label(layout, colors)
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
        self.add_fixed_orientation_mobjects(transmitted_label, reflected_label, pbs_label)

    def _geometry_resolution(self, mode: RenderMode) -> dict[str, tuple[int, int]]:
        if mode.high_detail:
            return {"centrifuge": (28, 320), "inset": (28, 120)}

        return {"centrifuge": (12, 120), "inset": (12, 48)}

    def _build_model(
        self,
        cfg: OpticalCentrifuge,
        z_R_extra: Length,
        z_L_extra: Length,
        layout: SceneLayout,
    ):
        def pulse_time(s: float) -> float:
            return (layout.pulse_advance - s) * layout.seconds_per_manim_unit

        # Find the envelope's peak by probing a wide, fine window of physical
        # time. The drawn extent is fixed by the layout, independent of it.
        t_probe = np.linspace(-3000e-12, 3000e-12, 6001)
        intensity_probe = np.asarray(cfg.intensity(t_probe, z_R_extra, z_L_extra), dtype=float)
        intensity_max = float(np.max(intensity_probe))
        if intensity_max <= 0:
            raise ValueError("Centrifuge intensity is zero in the sampled window.")
        peak_index = int(np.argmax(intensity_probe))

        s_peak = layout.pulse_advance - t_probe[peak_index] / layout.seconds_per_manim_unit
        s_start = layout.beam_start
        s_end = layout.beam_end

        def amplitude(s: float) -> float:
            intensity = float(cfg.intensity(pulse_time(s), z_R_extra, z_L_extra))
            return np.sqrt(max(intensity, 0.0) / intensity_max)

        def angle(s: float) -> float:
            return float(cfg.polarization_angle(pulse_time(s), z_R_extra, z_L_extra))

        return amplitude, angle, s_start, s_end, s_peak

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

    def _make_pbs_label(self, layout: SceneLayout, colors: SceneColors) -> Tex:
        label = Tex("PBS", font_size=44, color=colors.pbs)
        label.move_to(layout.pbs_center + np.array([0.0, 0.0, -(layout.pbs_size / 2.0 + 0.8)]))
        return label

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
