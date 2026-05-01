from __future__ import annotations

import numpy as np
from manim import *

from base_core.physics.optical_centrifuge import OpticalCentrifuge, DELTA_DELAY_ARM
from base_core.quantities.models import Length

# Compile it by running
# manim -pql misc_scripts/animation_optical_centrifuge.py PhysicalOpticalCentrifuge3D
# manim -pqh misc_scripts/animation_optical_centrifuge.py PhysicalOpticalCentrifuge3D

def create_centrifuge_for_animation() -> OpticalCentrifuge:
    """
    Construct the physical optical centrifuge used by the animation.

    The right and left arm use almost equal chirps, with a small chirp mismatch.
    The actual relative delay is introduced through different propagation lengths
    z_R and z_L when evaluating the centrifuge field.
    """
    ''' right_arm = CircularChirpedPulse(
            1,
            CENTRAL_FREQUENCY,
            BETA_0,
            PHASE_0,
            T,
            CircularHandedness.RIGHT,
        )

        left_arm = CircularChirpedPulse(
            1,
            CENTRAL_FREQUENCY,
            BETA_0 + DELTA_BETA,
            PHASE_0,
            T,
            CircularHandedness.LEFT,
        )
    '''
    return OpticalCentrifuge()


class PhysicalOpticalCentrifuge3D(ThreeDScene):
    def construct(self):
        # ============================================================
        # Build physical model
        # ============================================================

        cfg = create_centrifuge_for_animation()

        # The right arm is the reference arm.
        # The left arm propagates an additional 1.5 mm.
        #
        # Your OpticalCentrifuge class internally uses t_ret = t - z/c.
        z_R_extra = Length(2* DELTA_DELAY_ARM)
        z_L_extra = DELTA_DELAY_ARM

        # ============================================================
        # Animation parameters
        # ============================================================

        t_min = -6.0
        t_max = 6.0
        t = ValueTracker(t_min)

        # This maps Manim pulse-frame distance xi to physical pulse time.
        #
        # The full visible pulse coordinate spans about propagation_length.
        # With 80 ps / Manim unit and propagation_length = 9, the displayed
        # physical time window is about 720 ps, which is a good visual match
        # for a 300 ps FWHM pulse.
        seconds_per_manim_unit = 80e-12

        # Molecule follows only above this local amplitude fraction.
        # This is based on sqrt(I / I_max), not I / I_max.
        capture_fraction = 0.25

        # Scene geometry
        propagation_length = 9.0
        droplet_center = np.array([2.2, 0.0, 0.0])
        molecule_center = droplet_center
        molecule_length = 1.45

        # Moving pulse geometry in Manim units
        pulse_start_x = -5.2
        pulse_end_x = droplet_center[0] + 1.4

        # Smooth centrifuge surface
        centrifuge_color = "#D18A45"
        ribbon_width = 1.35
        ribbon_opacity = 0.42

        # Performance / quality knobs
        centrifuge_resolution = (12, 120)
        sphere_resolution = (10, 5)
        droplet_resolution = (24, 12)

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
            """
            Center of the Gaussian/corkscrew pulse in the lab frame.
            This controls only the visual motion.
            """
            progress = (anim_time - t_min) / (t_max - t_min)
            return pulse_start_x + progress * (pulse_end_x - pulse_start_x)

        def pulse_coordinate(x: float, anim_time: float) -> float:
            """
            Co-moving pulse coordinate in Manim units.

            xi = x - x_c(t)
            """
            return x - pulse_center_x(anim_time)

        def pulse_time_from_xi(xi: float) -> float:
            """
            Map fixed spatial pulse coordinate xi to physical pulse time.

            If the corkscrew rotates the wrong way visually, flip the sign here.
            """
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
            """
            Use sqrt(I/Imax), so this behaves like electric-field amplitude.
            """
            intensity = float(cfg.intensity(pulse_time, z_R_extra, z_L_extra))
            return np.sqrt(max(intensity, 0.0) / intensity_norm)

        def field_angle_from_pulse_time(pulse_time: float) -> float:
            """
            Use the polarization angle returned by your OpticalCentrifuge class.
            This avoids manually reimplementing the phase-difference formula.
            """
            angle = cfg.polarization_angle(pulse_time, z_R_extra, z_L_extra)
            return float(angle)

        def normalized_amplitude_at_x(x: float, anim_time: float) -> float:
            xi = pulse_coordinate(x, anim_time)
            pulse_time = pulse_time_from_xi(xi)
            return normalized_field_amplitude_from_pulse_time(pulse_time)

        def field_angle_at_x(x: float, anim_time: float) -> float:
            xi = pulse_coordinate(x, anim_time)
            pulse_time = pulse_time_from_xi(xi)
            return field_angle_from_pulse_time(pulse_time)

        def field_direction_from_angle(theta: float) -> np.ndarray:
            """
            Visual convention:
            propagation is along Manim x.
            The E-field rotates in the Manim y-z plane.
            """
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
            """
            Return local centrifuge rotation frequency at the molecule in Hz.
            """
            xi = pulse_coordinate(droplet_center[0], anim_time)
            pulse_time = pulse_time_from_xi(xi)
            return float(cfg.centrifuge_frequency(pulse_time, z_R_extra, z_L_extra))

        def molecule_direction(anim_time: float) -> np.ndarray:
            """
            Molecule follows the local E-field only after the local field
            amplitude at the molecule exceeds capture_fraction.
            """
            if field_strength_at_molecule(anim_time) < capture_fraction:
                return np.array([0.0, 1.0, 0.0])

            return field_direction_at_molecule(anim_time)
        
        # ============================================================
        # Droplet design
        # ============================================================

        def blobby_radius(theta: float, phi: float, phase: float) -> float:
            """
            Smooth animated angular radius modulation.
            Small amplitudes keep the droplet helium-like and avoid folding.
            """
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
            """
            u = theta, azimuthal angle
            v = phi, polar angle
            """
            theta = u
            phi = v

            r = blobby_radius(theta, phi, phase)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            return center + np.array([x, y, z])

        # ============================================================
        # Build smooth centrifuge once in the pulse frame
        # ============================================================

        def build_static_centrifuge_profile() -> VGroup:
            """
            Smooth optical centrifuge:
              - no center axis
              - no rods
              - no ridge lines
              - only a smooth translucent twisted surface

            Built once in the pulse frame. Then translated by x_c(t).
            """
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
        # Helium droplet: blobby surface
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
        # Molecule: simple CS2-like linear molecule
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
        # Local E-field vector at molecule
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

        he_label = Text("He", font_size=36, color=TEAL_A)
        he_label.move_to(droplet_center + np.array([-0.25, -0.75, 0.85]))

        def rotation_frequency_label() -> VGroup:
            current_time = t.get_value()
            amp = field_strength_at_molecule(current_time)

            if amp < capture_fraction:
                frequency_ghz = 0.0
            else:
                frequency_hz = centrifuge_frequency_at_molecule(current_time)
                frequency_ghz = abs(frequency_hz) / 1e9

            # Reserve space for two digits before the decimal point.
            # Example:
            #  1.8 -> \phantom{0}1.8
            # 19.8 -> 19.8
            number_text = f"{frequency_ghz:4.1f}".replace(" ", r"\phantom{0}")

            if frequency_ghz < 10:
                number_text = rf"\phantom{{0}}{frequency_ghz:.1f}"
            else:
                number_text = rf"{frequency_ghz:.1f}"

            label = MathTex(
                rf"f_{{\mathrm{{mol}}}} = {number_text}\,\mathrm{{GHz}}",
                font_size=30,
                color=WHITE,
            )

            # Fixed screen position. Adjust this until it is visually above the molecule.
            label.move_to(UP * 2.75 + RIGHT * 1.15)

            box = BackgroundRectangle(
                label,
                color=BLACK,
                fill_opacity=0.35,
                buff=0.12,
            )

            return VGroup(box, label)


        frequency_label = always_redraw(rotation_frequency_label)
        self.add_fixed_in_frame_mobjects(frequency_label)
        

        self.add_fixed_orientation_mobjects(
            he_label,
        )

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