"""Coordinate helpers for the front-suspension pickup points."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Point(NamedTuple):
    """An inch-based point in the vehicle coordinate system.

    The x, y, and z axes are lateral, longitudinal, and vertical,
    respectively.
    """

    x: float
    y: float
    z: float

    def __add__(self, other: object) -> Point:
        """Return the component-wise sum with another point or vector.

        Returns:
            The summed point, or ``NotImplemented`` for an unsupported operand.

        """
        if not isinstance(other, Point):
            return NotImplemented

        return Point(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: object) -> Point:
        """Return the component-wise difference from another point or vector.

        Returns:
            The difference point, or ``NotImplemented`` for an unsupported operand.

        """
        if not isinstance(other, Point):
            return NotImplemented

        return Point(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __abs__(self) -> float:
        """Return this point's Euclidean length in inches.

        Returns:
            The Pythagorean length of the point's three components in inches.

        """
        return float(np.sqrt(self.x**2 + self.y**2 + self.z**2))


class Geometry(NamedTuple):
    """Fixed dimensions and angles for one side of the front suspension."""

    lower_in: float = 10
    lower_pickup_width_in: float = 24
    upper_in: float = 12
    inner_in: float = 7
    inner_angle_rad: float = float(np.deg2rad(20))
    outer_in: float = 12
    caster_rad: float = float(np.deg2rad(10.0))

    axle_length_in: float = 5
    axle_offset_in: float = 5

    wheel_radius_in: float = 22.1 / 2

    tie_rod_inward_in: float = 1
    tie_rod_up_in: float = 3
    tie_rod_forward_in: float = 8

    @property
    def caster_cos(self) -> float:
        """Cosine of the kingpin caster angle."""
        return float(np.cos(self.caster_rad))

    @property
    def caster_sin(self) -> float:
        """Sine of the kingpin caster angle."""
        return float(np.sin(self.caster_rad))

    def lower_pickup(self) -> Point:
        """Return the lower inboard pickup point for this suspension side.

        Returns:
            The lower pickup point in inches.

        """
        return Point(
            self.lower_pickup_width_in / 2.0,
            0.0,
            0.0,
        )

    def upper_pickup(self) -> Point:
        """Return the upper inboard pickup point relative to the lower one.

        Returns:
            The upper pickup point in inches.

        """
        inner_projection_in = self.inner_in * float(np.cos(self.inner_angle_rad))
        return self.lower_pickup() + Point(
            -self.inner_in * float(np.sin(self.inner_angle_rad)),
            inner_projection_in * self.caster_sin,
            inner_projection_in * self.caster_cos,
        )

    def lower_trunnion(self, lower_angle_rad: float) -> Point:
        """Return the lower outboard trunnion point at the supplied arm angle.

        Returns:
            The lower trunnion point in inches.

        """
        lower_projection_in = self.lower_in * float(np.sin(lower_angle_rad))
        return self.lower_pickup() + Point(
            self.lower_in * float(np.cos(lower_angle_rad)),
            lower_projection_in * self.caster_sin,
            lower_projection_in * self.caster_cos,
        )

    def upper_trunnion(self, lower_angle_rad: float) -> Point:
        """Return the upper outboard trunnion point at the supplied arm angle.

        Returns:
            The upper trunnion point in inches.

        """
        theta_outer_horizon = self.outer_horizon_rad(lower_angle_rad)
        outer_projection_in = self.outer_in * float(np.sin(theta_outer_horizon))
        return self.lower_trunnion(lower_angle_rad) + Point(
            -self.outer_in * float(np.cos(theta_outer_horizon)),
            outer_projection_in * self.caster_sin,
            outer_projection_in * self.caster_cos,
        )

    def outer_horizon_rad(self, lower_angle_rad: float) -> float:
        """Return the outer link angle from the lateral horizon.

        Returns:
            The outer link angle in radians.

        """
        trunnion_distance_in = abs(
            self.upper_pickup() - self.lower_trunnion(lower_angle_rad),
        )
        theta_inner_to_lower_rad = (
            self.inner_angle_rad + float(np.pi) / 2.0 - lower_angle_rad
        )
        theta_lower_trunnion_rad = np.arcsin(
            np.sin(theta_inner_to_lower_rad) / trunnion_distance_in * self.inner_in,
        )
        theta_outer_trunnion_rad = np.arccos(
            (trunnion_distance_in**2 + self.outer_in**2 - self.upper_in**2)
            / (2.0 * trunnion_distance_in * self.outer_in),
        )
        theta_lower_to_outer_rad = theta_lower_trunnion_rad + theta_outer_trunnion_rad
        return float(theta_lower_to_outer_rad - lower_angle_rad)

    def axle(self, lower_angle_rad: float) -> Point:
        """Return the axle point at the supplied lower control-arm angle.

        Returns:
            The axle point in inches.

        """
        theta_outer_horizon = self.outer_horizon_rad(lower_angle_rad)
        outer_projection_in = self.axle_offset_in * float(np.sin(theta_outer_horizon))
        return self.lower_trunnion(lower_angle_rad) + Point(
            -self.axle_offset_in * float(np.cos(theta_outer_horizon)),
            outer_projection_in * self.caster_sin,
            outer_projection_in * self.caster_cos,
        )

    def rotate_vector_about_upright(
        self,
        vector: Point,
        lower_angle_rad: float,
        toe_angle_rad: float,
    ) -> Point:
        """Rotate a vector about the lower-to-upper trunnion axis.

        Positive toe is a negative right-hand rotation around the upright so that
        the wheel moves toward negative longitudinal position.

        Returns:
            The toe-rotated vector.

        """
        lower_trunnion_point = self.lower_trunnion(lower_angle_rad)
        upright_vector = self.upper_trunnion(lower_angle_rad) - lower_trunnion_point
        upright_length_in = abs(upright_vector)
        upright_unit_x = upright_vector.x / upright_length_in
        upright_unit_y = upright_vector.y / upright_length_in
        upright_unit_z = upright_vector.z / upright_length_in
        toe_cos = float(np.cos(-toe_angle_rad))
        toe_sin = float(np.sin(-toe_angle_rad))
        upright_vector_dot_product = (
            upright_unit_x * vector.x
            + upright_unit_y * vector.y
            + upright_unit_z * vector.z
        )
        return Point(
            vector.x * toe_cos
            + (upright_unit_y * vector.z - upright_unit_z * vector.y) * toe_sin
            + upright_unit_x * upright_vector_dot_product * (1.0 - toe_cos),
            vector.y * toe_cos
            + (upright_unit_z * vector.x - upright_unit_x * vector.z) * toe_sin
            + upright_unit_y * upright_vector_dot_product * (1.0 - toe_cos),
            vector.z * toe_cos
            + (upright_unit_x * vector.y - upright_unit_y * vector.x) * toe_sin
            + upright_unit_z * upright_vector_dot_product * (1.0 - toe_cos),
        )

    def wheel_center(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the wheel center at the supplied lower control-arm angle.

        Returns:
            The wheel center point in inches.

        """
        theta_outer_horizon = self.outer_horizon_rad(lower_angle_rad)
        theta_axle_horizon = float(np.pi) / 2.0 - theta_outer_horizon
        axle_projection_in = self.axle_offset_in * float(np.sin(theta_axle_horizon))
        return self.axle(lower_angle_rad) + self.rotate_vector_about_upright(
            Point(
                self.axle_length_in * float(np.cos(theta_axle_horizon)),
                axle_projection_in * self.caster_sin,
                axle_projection_in * self.caster_cos,
            ),
            lower_angle_rad,
            toe_angle_rad,
        )

    def wheel_contact(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the tire contact point at the supplied lower control-arm angle.

        Returns:
            The wheel contact point in inches.

        """
        theta_outer_horizon = self.outer_horizon_rad(lower_angle_rad)
        return self.wheel_center(
            lower_angle_rad,
            toe_angle_rad,
        ) + self.rotate_vector_about_upright(
            Point(
                self.wheel_radius_in * float(np.cos(theta_outer_horizon)),
                0.0,
                -self.wheel_radius_in * float(np.sin(theta_outer_horizon)),
            ),
            lower_angle_rad,
            toe_angle_rad,
        )

    def tie_rod_point(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the outer tie-rod point in the wheel-disc coordinate frame.

        The tie-rod point is inboard along the wheel spindle, upward along the
        upright, and forward in the wheel-disc plane.

        Returns:
            The outer tie-rod point in inches.

        """
        axle_point = self.axle(lower_angle_rad)
        wheel_center_point = self.wheel_center(lower_angle_rad, toe_angle_rad)
        wheel_outward_vector = wheel_center_point - axle_point
        wheel_outward_length_in = abs(wheel_outward_vector)
        wheel_outward_unit_x = wheel_outward_vector.x / wheel_outward_length_in
        wheel_outward_unit_y = wheel_outward_vector.y / wheel_outward_length_in
        wheel_outward_unit_z = wheel_outward_vector.z / wheel_outward_length_in
        upright_vector = self.upper_trunnion(lower_angle_rad) - self.lower_trunnion(
            lower_angle_rad,
        )
        upright_length_in = abs(upright_vector)
        upright_unit_x = upright_vector.x / upright_length_in
        upright_unit_y = upright_vector.y / upright_length_in
        upright_unit_z = upright_vector.z / upright_length_in
        wheel_forward_vector = Point(
            wheel_outward_unit_y * upright_unit_z
            - wheel_outward_unit_z * upright_unit_y,
            wheel_outward_unit_z * upright_unit_x
            - wheel_outward_unit_x * upright_unit_z,
            wheel_outward_unit_x * upright_unit_y
            - wheel_outward_unit_y * upright_unit_x,
        )
        wheel_forward_length_in = abs(wheel_forward_vector)
        wheel_forward_unit_x = wheel_forward_vector.x / wheel_forward_length_in
        wheel_forward_unit_y = wheel_forward_vector.y / wheel_forward_length_in
        wheel_forward_unit_z = wheel_forward_vector.z / wheel_forward_length_in
        return wheel_center_point + Point(
            -self.tie_rod_inward_in * wheel_outward_unit_x
            + self.tie_rod_up_in * upright_unit_x
            + self.tie_rod_forward_in * wheel_forward_unit_x,
            -self.tie_rod_inward_in * wheel_outward_unit_y
            + self.tie_rod_up_in * upright_unit_y
            + self.tie_rod_forward_in * wheel_forward_unit_y,
            -self.tie_rod_inward_in * wheel_outward_unit_z
            + self.tie_rod_up_in * upright_unit_z
            + self.tie_rod_forward_in * wheel_forward_unit_z,
        )

    def instant_center(self, lower_angle_rad: float) -> Point:
        """Return the projected instant center at the supplied arm angle.

        Returns:
            The instant-center point in inches.

        """
        theta_outer_horizon = self.outer_horizon_rad(lower_angle_rad)
        theta_lower_to_outer_rad = theta_outer_horizon + lower_angle_rad
        upper_trunnion_point = self.upper_trunnion(lower_angle_rad)
        upper_pickup_point = self.upper_pickup()
        theta_upper_horizon = float(
            np.arctan2(
                (upper_trunnion_point.z - upper_pickup_point.z) / self.caster_cos,
                upper_trunnion_point.x - upper_pickup_point.x,
            ),
        )
        theta_outer_to_upper_rad = float(
            np.pi - theta_outer_horizon - theta_upper_horizon,
        )
        theta_instant_center_rad = float(
            np.pi - theta_outer_to_upper_rad - theta_lower_to_outer_rad,
        )
        instant_center_distance_in = self.outer_in * float(
            np.sin(theta_outer_to_upper_rad) / np.sin(theta_instant_center_rad),
        )
        instant_center_projection_in = instant_center_distance_in * float(
            np.sin(lower_angle_rad),
        )
        return self.lower_trunnion(lower_angle_rad) + Point(
            -instant_center_distance_in * float(np.cos(lower_angle_rad)),
            -instant_center_projection_in * self.caster_sin,
            -instant_center_projection_in * self.caster_cos,
        )

    def roll_center(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the x=0 intersection of the instant-center contact line.

        Returns:
            The roll-center point in inches.

        """
        instant_center_point = self.instant_center(lower_angle_rad)
        wheel_contact_point = self.wheel_contact(lower_angle_rad, toe_angle_rad)
        instant_center_to_contact_x_in = wheel_contact_point.x - instant_center_point.x
        interpolation_fraction = (
            -instant_center_point.x / instant_center_to_contact_x_in
        )
        return Point(
            0.0,
            instant_center_point.y
            + interpolation_fraction * (wheel_contact_point.y - instant_center_point.y),
            instant_center_point.z
            + interpolation_fraction * (wheel_contact_point.z - instant_center_point.z),
        )
