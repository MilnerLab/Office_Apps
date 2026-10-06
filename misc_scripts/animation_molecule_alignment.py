from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from manim import *


# Schematic of the alignment potential of a linear molecule in a linearly
# polarized field,
#
#   U(theta, varphi, t) = -1/4 eps_0^2(t) [alpha_perp + Delta alpha sin^2(theta) cos^2(varphi - Phi(t))],
#
# with the polarization plane drawn as a disc, theta measured from the z-axis
# (propagation direction) and varphi - Phi(t) measured in the disc between the
# projection of the molecular axis and the field direction.
#
# Quick preview:
#   manim -pql misc_scripts/animation_molecule_alignment.py MoleculeAlignmentSchematic
#
# Render one high-resolution snapshot:
#   manim -s -r 7680,4320 misc_scripts/animation_molecule_alignment.py MoleculeAlignmentSchematic
#
# Optional high-detail geometry for still images:
#   MANIM_HIGH_DETAIL=1 manim -s -r 7680,4320 misc_scripts/animation_molecule_alignment.py MoleculeAlignmentSchematic


def lab_to_scene(v: np.ndarray) -> np.ndarray:
    # Same lab frame as animation_cfg_projection.py: the propagation axis z
    # runs along manim's x, lab x along manim's -y and lab y is vertical.
    x, y, z = v
    return np.array([z, -x, y])


@dataclass(frozen=True)
class RenderMode:
    high_detail: bool = os.getenv("MANIM_HIGH_DETAIL", "0") == "1"


@dataclass(frozen=True)
class SceneLayout:
    disc_radius: float = 2.3
    disc_thickness: float = 0.1
    disc_opacity: float = 0.22
    molecule_length: float = 3.2
    atom_radius: float = 0.3
    bond_thickness: float = 0.07
    axis_length: float = 2.4
    theta: float = 50 * DEGREES
    varphi: float = 117 * DEGREES
    field_angle: float = 60 * DEGREES
    field_length: float = 3.8
    theta_arc_radius: float = 1.0
    in_plane_arc_radius: float = 0.9
    # Line3D thickness is the cylinder radius; the angle arcs are trimmed by
    # these radii so that they end exactly on the cylinders they measure from.
    z_axis_thickness: float = 0.035
    field_thickness: float = 0.045
    projection_thickness: float = 0.04
    drop_line_thickness: float = 0.018
    arc_thickness: float = 0.022
    # Angle labels sit on the bisector of their arc (distance from the arc's
    # center) with a leader line to the arc; the gap keeps the line clear of
    # the text.
    theta_label_distance: float = 1.7
    theta_label_gap: float = 0.22
    in_plane_label_distance: float = 2.0
    in_plane_label_gap: float = 0.18
    leader_thickness: float = 0.012
    dash_length: float = 0.14
    dash_gap: float = 0.09

    @property
    def molecule_axis(self) -> np.ndarray:
        return np.array(
            [
                np.sin(self.theta) * np.cos(self.varphi),
                np.sin(self.theta) * np.sin(self.varphi),
                np.cos(self.theta),
            ]
        )


@dataclass(frozen=True)
class SceneColors:
    background: ManimColor = WHITE
    z_axis: ManimColor = PINK
    disc: ManimColor = RED_B
    atom: ManimColor = BLUE_D
    bond: ManimColor = GREY_D
    projection: ManimColor = GREY
    e_field: ManimColor = RED
    angle: ManimColor = DARKER_GREY

    @staticmethod
    def dark() -> "SceneColors":
        return SceneColors(
            background=BLACK,
            z_axis=PINK,
            disc=RED_B,
            atom=BLUE_C,
            bond=GREY_B,
            projection=GREY_B,
            e_field=RED,
            angle=WHITE,
        )

    @staticmethod
    def light() -> "SceneColors":
        return SceneColors()


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
        show_base=True,
    )
    cone.set_fill(color, opacity=1.0)
    cone.set_stroke(color, width=0, opacity=0.0)
    cone.set_shade_in_3d(True)

    points = cone.get_all_points()
    projections = points @ direction
    current_tip = points[np.argmax(projections)]
    cone.shift(tip_position - current_tip)
    return cone


def make_arrow(
    start: np.ndarray,
    end: np.ndarray,
    color: ManimColor | str,
    thickness: float = 0.035,
    cone_radius: float = 0.1,
    cone_height: float = 0.22,
) -> VGroup:
    direction = (end - start) / np.linalg.norm(end - start)
    shaft = Line3D(start=start, end=end - cone_height * direction, color=color, thickness=thickness)
    return VGroup(shaft, make_arrow_tip(end, direction, color, radius=cone_radius, height=cone_height))


def make_double_arrow(
    start: np.ndarray,
    end: np.ndarray,
    color: ManimColor | str,
    thickness: float = 0.045,
    cone_radius: float = 0.12,
    cone_height: float = 0.26,
) -> VGroup:
    direction = (end - start) / np.linalg.norm(end - start)
    shaft = Line3D(
        start=start + cone_height * direction,
        end=end - cone_height * direction,
        color=color,
        thickness=thickness,
    )
    return VGroup(
        shaft,
        make_arrow_tip(start, -direction, color, radius=cone_radius, height=cone_height),
        make_arrow_tip(end, direction, color, radius=cone_radius, height=cone_height),
    )


def make_dashed_line3d(
    start: np.ndarray,
    end: np.ndarray,
    color: ManimColor | str,
    thickness: float,
    dash_length: float,
    dash_gap: float,
    anchor: float | None = None,
) -> VGroup:
    # Dashes are laid out so that one dash is centered on `anchor` (distance
    # from `start`), which lets an angle arc always meet a dash, not a gap.
    length = float(np.linalg.norm(end - start))
    direction = (end - start) / length
    period = dash_length + dash_gap
    if anchor is None:
        anchor = 0.5 * dash_length

    first_center = anchor - np.ceil(anchor / period) * period
    dashes = VGroup()
    for center in np.arange(first_center, length + period, period):
        a = max(center - 0.5 * dash_length, 0.0)
        b = min(center + 0.5 * dash_length, length)
        if b - a < 0.2 * dash_length:
            continue
        dash = Line3D(start=start + a * direction, end=start + b * direction, color=color, thickness=thickness)
        dashes.add(dash)
    return dashes


def make_arc_tube(
    center: np.ndarray,
    e1: np.ndarray,
    e2: np.ndarray,
    radius: float,
    t_range: tuple[float, float],
    tube_radius: float,
    color: ManimColor | str,
    resolution: tuple[int, int] = (8, 48),
) -> Surface:
    # Section of a torus: a tube of radius `tube_radius` following the arc
    # center + radius * (cos t e1 + sin t e2) with e1, e2 orthonormal.
    binormal = np.cross(e1, e2)

    def point(u: float, t: float) -> np.ndarray:
        radial = np.cos(t) * e1 + np.sin(t) * e2
        return center + (radius + tube_radius * np.cos(u)) * radial + tube_radius * np.sin(u) * binormal

    tube = Surface(
        point,
        u_range=[0.0, TAU],
        v_range=list(t_range),
        resolution=resolution,
        checkerboard_colors=[ManimColor(color), ManimColor(color)],
    )
    tube.set_style(stroke_width=0, stroke_opacity=0, fill_opacity=1.0)
    tube.set_shade_in_3d(True)
    return tube


class LayeredThreeDCamera(ThreeDCamera):
    # The stock ThreeDCamera paints shaded faces back to front by the depth of
    # each face's center, which goes wrong for faces lying on or passing
    # through the disc, and always paints unshaded mobjects last. This camera
    # instead paints by an explicit `depth_layer` (set via assign_depth_layer)
    # and only depth-sorts faces within a layer.
    def get_mobjects_to_display(self, *args, **kwargs) -> list[Mobject]:
        mobjects = Camera.get_mobjects_to_display(self, *args, **kwargs)
        rot_matrix = self.get_rotation_matrix()

        def key(mob: Mobject) -> tuple[float, float]:
            depth = float(np.dot(mob.get_z_index_reference_point(), rot_matrix.T)[2])
            return getattr(mob, "depth_layer", np.inf), depth

        return sorted(mobjects, key=key)

    # Manim's lighting adds up to +0.5 to every RGB channel, which washes thin
    # colored cylinders out towards white. A mobject's `shading_strength`
    # (set via set_shading_strength) scales that change.
    def modified_rgbas(self, vmobject: VMobject, rgbas: np.ndarray) -> np.ndarray:
        shaded = super().modified_rgbas(vmobject, rgbas)
        strength = getattr(vmobject, "shading_strength", 1.0)
        if strength == 1.0 or shaded is rgbas:
            return shaded
        base = rgbas.repeat(2, axis=0) if len(rgbas) < 2 else np.array(rgbas[:2])
        return base + strength * (shaded - base)


def assign_depth_layer(mobject: Mobject, layer: int) -> None:
    for member in mobject.get_family():
        member.depth_layer = layer


def set_shading_strength(mobject: Mobject, strength: float) -> None:
    for member in mobject.get_family():
        member.shading_strength = strength


class MoleculeAlignmentSchematic(ThreeDScene):
    def __init__(self, **kwargs) -> None:
        super().__init__(camera_class=LayeredThreeDCamera, **kwargs)

    def construct(self) -> None:
        mode = RenderMode()
        layout = SceneLayout()
        colors = SceneColors.light()
        geometry = self._geometry_resolution(mode)

        self.camera.background_color = colors.background
        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-50 * DEGREES,
            zoom=1.5,
            frame_center=lab_to_scene(np.array([0.0, 0.0, 0.3])),
        )

        camera = self.renderer.camera
        if hasattr(camera, "light_source"):
            camera.light_source.move_to(3 * OUT + 4 * LEFT + 5 * UP)

        disc_back, disc_front = self._make_disc(layout, colors, geometry)
        z_axis, z_label = self._make_z_axis(layout, colors)
        molecule_back, molecule_front = self._make_molecule(layout, colors, geometry)
        projection, upper_drop_line, lower_drop_line = self._make_projection(layout, colors)
        e_field, e_label = self._make_e_field(layout, colors)
        theta_arc, theta_label = self._make_theta_arc(layout, colors)
        phi_arc, phi_label = self._make_in_plane_arc(layout, colors)

        # Painted back to front: what is behind the disc's mid-plane, the back
        # half of the disc, what lies in the mid-plane, the front half of the
        # disc, and what is in front of the mid-plane.
        layers = [
            [molecule_back, lower_drop_line],
            [disc_back],
            [projection, e_field, phi_arc],
            [disc_front],
            [molecule_front, z_axis, upper_drop_line, theta_arc],
        ]
        for mobject in (e_field, z_axis, projection, upper_drop_line, lower_drop_line):
            set_shading_strength(mobject, 0.35)

        for depth, mobjects in enumerate(layers):
            for mobject in mobjects:
                assign_depth_layer(mobject, depth)
            self.add(*mobjects)
        self.add_fixed_orientation_mobjects(z_label, e_label, theta_label, phi_label)

        self.wait(1 / self.camera.frame_rate)

    def _geometry_resolution(self, mode: RenderMode) -> dict[str, tuple[int, int]]:
        if mode.high_detail:
            return {"disc": (96, 192), "atom": (48, 24)}

        return {"disc": (48, 96), "atom": (24, 12)}

    def _in_plane(self, layout: SceneLayout, angle: float, radius: float) -> np.ndarray:
        return lab_to_scene(np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0]))

    def _make_disc(
        self, layout: SceneLayout, colors: SceneColors, geometry: dict[str, tuple[int, int]]
    ) -> tuple[Surface, Surface]:
        # Surface of revolution of a stadium profile: flat front face,
        # half-circle rim, flat back face. u runs along the profile (by arc
        # length), v is the azimuth around the propagation axis. The profile
        # is symmetric, so u = 0.5 is the middle of the rim and splits the disc
        # into a back and a front half, between which the in-plane elements
        # in the mid-plane are painted.
        edge_radius = layout.disc_thickness / 2
        flat = layout.disc_radius - edge_radius
        rim = np.pi * edge_radius
        total = 2 * flat + rim

        def point(u: float, v: float) -> np.ndarray:
            s = u * total
            if s < flat:
                rho, z = s, edge_radius
            elif s < flat + rim:
                alpha = np.pi / 2 - (s - flat) / edge_radius
                rho, z = flat + edge_radius * np.cos(alpha), edge_radius * np.sin(alpha)
            else:
                rho, z = flat - (s - flat - rim), -edge_radius
            return lab_to_scene(np.array([rho * np.cos(v), rho * np.sin(v), z]))

        u_resolution, v_resolution = geometry["disc"]

        def half(u_range: tuple[float, float]) -> Surface:
            surface = Surface(
                point,
                u_range=list(u_range),
                v_range=[0.0, TAU],
                resolution=(u_resolution // 2, v_resolution),
                fill_opacity=layout.disc_opacity,
                checkerboard_colors=[ManimColor(colors.disc), ManimColor(colors.disc)],
            )
            surface.set_style(stroke_width=0, stroke_opacity=0, fill_opacity=layout.disc_opacity)
            surface.set_shade_in_3d(True)
            return surface

        return half((0.5, 1.0)), half((0.0, 0.5))

    def _make_z_axis(self, layout: SceneLayout, colors: SceneColors) -> tuple[VGroup, MathTex]:
        end = lab_to_scene(np.array([0.0, 0.0, layout.axis_length]))
        axis = make_arrow(ORIGIN, end, colors.z_axis, thickness=layout.z_axis_thickness)

        label = MathTex("z", font_size=60, color=colors.z_axis)
        label.move_to(lab_to_scene(np.array([0.0, 0.0, layout.axis_length + 0.4])))
        return axis, label

    def _make_molecule(
        self, layout: SceneLayout, colors: SceneColors, geometry: dict[str, tuple[int, int]]
    ) -> tuple[VGroup, VGroup]:
        # Returns the halves behind and in front of the disc mid-plane, so they
        # can be painted on either side of the disc.
        axis = lab_to_scene(layout.molecule_axis)
        half = 0.5 * layout.molecule_length * axis

        # The bond stops where its rim meets the sphere surface; letting it
        # run into the atoms makes the hidden part show through the spheres.
        bond_end = np.sqrt(layout.atom_radius**2 - layout.bond_thickness**2)

        def bond(start: np.ndarray, end: np.ndarray) -> Line3D:
            return Line3D(
                start=start,
                end=end,
                color=colors.bond,
                thickness=layout.bond_thickness,
                resolution=(12, 24),
                show_ends=False,
            )

        def atom(center: np.ndarray) -> Sphere:
            return Sphere(center=center, radius=layout.atom_radius, resolution=geometry["atom"]).set_color(colors.atom)

        back = VGroup(bond(-half + bond_end * axis, ORIGIN), atom(-half))
        front = VGroup(bond(ORIGIN, half - bond_end * axis), atom(half))
        return back, front

    def _make_projection(self, layout: SceneLayout, colors: SceneColors) -> tuple[VGroup, VGroup, VGroup]:
        half_length = 0.5 * layout.molecule_length * np.sin(layout.theta)
        tip = self._in_plane(layout, layout.varphi, half_length)
        tail = self._in_plane(layout, layout.varphi + PI, half_length)

        projection = make_dashed_line3d(
            tail,
            tip,
            colors.projection,
            thickness=layout.projection_thickness,
            dash_length=layout.dash_length,
            dash_gap=layout.dash_gap,
            anchor=half_length + layout.in_plane_arc_radius,
        )

        upper_atom = lab_to_scene(0.5 * layout.molecule_length * layout.molecule_axis)

        def drop_line(atom: np.ndarray, foot: np.ndarray) -> VGroup:
            direction = (foot - atom) / np.linalg.norm(foot - atom)
            return make_dashed_line3d(
                atom + layout.atom_radius * direction,
                foot,
                colors.projection,
                thickness=layout.drop_line_thickness,
                dash_length=0.5 * layout.dash_length,
                dash_gap=0.5 * layout.dash_gap,
            )

        return projection, drop_line(upper_atom, tip), drop_line(-upper_atom, tail)

    def _make_e_field(self, layout: SceneLayout, colors: SceneColors) -> tuple[VGroup, MathTex]:
        half_length = 0.5 * layout.field_length
        start = self._in_plane(layout, layout.field_angle + PI, half_length)
        end = self._in_plane(layout, layout.field_angle, half_length)
        arrow = make_double_arrow(start, end, colors.e_field, thickness=layout.field_thickness)

        label = MathTex(r"\vec{E}(t)", font_size=52, color=colors.e_field)
        label.move_to(self._in_plane(layout, layout.field_angle + PI, half_length + 0.55))
        return arrow, label

    def _make_theta_arc(self, layout: SceneLayout, colors: SceneColors) -> tuple[VGroup, MathTex]:
        r = layout.theta_arc_radius
        e1 = lab_to_scene(np.array([0.0, 0.0, 1.0]))
        e2 = lab_to_scene(np.array([np.cos(layout.varphi), np.sin(layout.varphi), 0.0]))

        # Arc angle t is measured from the z-axis towards the molecule, so
        # t = theta is the molecular axis.
        t_start = np.arcsin(layout.z_axis_thickness / r)
        t_end = layout.theta - np.arcsin(layout.bond_thickness / r)
        arc = make_arc_tube(ORIGIN, e1, e2, r, (t_start, t_end), layout.arc_thickness, colors.angle)

        mid = 0.5 * layout.theta
        bisector = np.cos(mid) * e1 + np.sin(mid) * e2
        leader = self._make_leader(layout, colors, bisector, r, layout.theta_label_distance, layout.theta_label_gap)

        label = MathTex(r"\theta", font_size=52, color=colors.angle)
        label.move_to(layout.theta_label_distance * bisector)
        return VGroup(arc, leader), label

    def _make_in_plane_arc(self, layout: SceneLayout, colors: SceneColors) -> tuple[VGroup, MathTex]:
        r = layout.in_plane_arc_radius
        e1 = lab_to_scene(np.array([1.0, 0.0, 0.0]))
        e2 = lab_to_scene(np.array([0.0, 1.0, 0.0]))

        ends = sorted(
            [
                (layout.field_angle, layout.field_thickness),
                (layout.varphi, layout.projection_thickness),
            ]
        )
        (t_min, r_min), (t_max, r_max) = ends
        t_range = (t_min + np.arcsin(r_min / r), t_max - np.arcsin(r_max / r))
        arc = make_arc_tube(ORIGIN, e1, e2, r, t_range, layout.arc_thickness, colors.angle)

        mid = 0.5 * (t_min + t_max)
        bisector = np.cos(mid) * e1 + np.sin(mid) * e2
        leader = self._make_leader(
            layout, colors, bisector, r, layout.in_plane_label_distance, layout.in_plane_label_gap
        )

        label = MathTex(r"\varphi - \Phi(t)", font_size=44, color=colors.angle)
        label.move_to(layout.in_plane_label_distance * bisector)
        return VGroup(arc, leader), label

    def _make_leader(
        self,
        layout: SceneLayout,
        colors: SceneColors,
        bisector: np.ndarray,
        arc_radius: float,
        label_distance: float,
        label_gap: float,
    ) -> Line3D:
        # Runs along the arc's bisector from the outside of the arc tube to
        # just short of the label.
        return Line3D(
            start=(arc_radius + layout.arc_thickness) * bisector,
            end=(label_distance - label_gap) * bisector,
            color=colors.angle,
            thickness=layout.leader_thickness,
        )
