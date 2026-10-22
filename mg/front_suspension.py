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

    lower_in: float = 10.75
    lower_pickup_width_in: float = 18
    upper_in: float = 7.25
    inner_in: float = 9
    inner_angle_rad: float = float(np.deg2rad(-31.5))
    outer_in: float = 8.25
    caster_rad: float = float(np.deg2rad(3))

    kingpin_upper_inward_in: float = 0.75
    kingpin_lower_inward_in: float = -1

    axle_length_in: float = 2.5
    axle_kingpin_angle_rad: float = float(np.deg2rad(8))
    axle_offset_in: float = 3.75

    wheel_radius_in: float = 22.6 / 2

    tie_rod_inward_in: float = 2.125
    tie_rod_up_in: float = 1.5
    tie_rod_forward_in: float = 5

    tie_rod_link_in: float = 8
    steering_rod_y_in: float = -4.5
    steering_rod_z_in: float = 5

    coilover_lower_offset_in: float = 10.75 * 2.0 / 3.0
    coilover_upper_x_from_lower_pickup_in: float = float(
        -9.0 * np.sin(np.deg2rad(-31.5)) + 3.0,
    )
    coilover_upper_z_from_lower_pickup_in: float = float(
        9.0 * np.cos(np.deg2rad(-31.5)) * np.cos(np.deg2rad(3.0)) + 4.0,
    )

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

    def coilover_lower_mount(self, lower_angle_rad: float) -> Point:
        """Return the lower coilover mount at its offset along the lower arm.

        Args:
            lower_angle_rad: Lower control-arm angle in radians.

        Returns:
            Lower coilover mount point in inches.

        """
        lower_pickup_point = self.lower_pickup()
        lower_arm_vector = self.lower_trunnion(lower_angle_rad) - lower_pickup_point
        return lower_pickup_point + Point(
            lower_arm_vector.x * self.coilover_lower_offset_in / self.lower_in,
            lower_arm_vector.y * self.coilover_lower_offset_in / self.lower_in,
            lower_arm_vector.z * self.coilover_lower_offset_in / self.lower_in,
        )

    def coilover_upper_mount(self) -> Point:
        """Return the chassis-fixed upper coilover mount in the x-z plane.

        Returns:
            Upper coilover mount point in inches.

        """
        lower_pickup_point = self.lower_pickup()
        upper_pickup_point = self.upper_pickup()
        return Point(
            (lower_pickup_point.x + self.coilover_upper_x_from_lower_pickup_in),
            upper_pickup_point.y,
            (lower_pickup_point.z + self.coilover_upper_z_from_lower_pickup_in),
        )

    def coilover_length_in(self, lower_angle_rad: float) -> float:
        """Return the coilover mount-to-mount length.

        Args:
            lower_angle_rad: Lower control-arm angle in radians.

        Returns:
            Coilover length in inches.

        """
        return abs(
            self.coilover_upper_mount() - self.coilover_lower_mount(lower_angle_rad),
        )

    def coilover_angle_to_vertical_rad(self, lower_angle_rad: float) -> float:
        """Return the coilover angle away from vertical using the 3D centerline.

        Args:
            lower_angle_rad: Lower control-arm angle in radians.

        Returns:
            Coilover angle from vertical in radians.

        """
        coilover_vector = self.coilover_upper_mount() - self.coilover_lower_mount(
            lower_angle_rad,
        )
        horizontal_distance_in = float(
            np.hypot(coilover_vector.x, coilover_vector.y),
        )
        return float(np.arctan2(horizontal_distance_in, abs(coilover_vector.z)))

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

    def kingpin_axis(self, lower_angle_rad: float) -> tuple[Point, Point]:
        """Return two points on the kingpin axis at the trunnion heights.

        The measured trunnion offsets are perpendicular to the kingpin in the
        suspension plane. Positive offsets point inboard.

        Returns:
            The lower and upper kingpin-axis points in inches.

        """
        lower_trunnion_point = self.lower_trunnion(lower_angle_rad)
        upper_trunnion_point = self.upper_trunnion(lower_angle_rad)
        trunnion_vector = upper_trunnion_point - lower_trunnion_point
        trunnion_plane_z_in = trunnion_vector.z / self.caster_cos
        trunnion_distance_in = float(
            np.hypot(trunnion_vector.x, trunnion_plane_z_in),
        )
        trunnion_unit_x = trunnion_vector.x / trunnion_distance_in
        trunnion_unit_z = trunnion_plane_z_in / trunnion_distance_in
        offset_difference_in = (
            self.kingpin_upper_inward_in - self.kingpin_lower_inward_in
        )
        normal_projection = -offset_difference_in / trunnion_distance_in
        normal_perpendicular_component = float(
            np.sqrt(1.0 - normal_projection**2),
        )
        inward_candidate_1_x = (
            normal_projection * trunnion_unit_x
            - normal_perpendicular_component * trunnion_unit_z
        )
        inward_candidate_1_z = (
            normal_projection * trunnion_unit_z
            + normal_perpendicular_component * trunnion_unit_x
        )
        inward_candidate_2_x = (
            normal_projection * trunnion_unit_x
            + normal_perpendicular_component * trunnion_unit_z
        )
        inward_candidate_2_z = (
            normal_projection * trunnion_unit_z
            - normal_perpendicular_component * trunnion_unit_x
        )
        inward_unit_x, inward_unit_z = (
            (inward_candidate_1_x, inward_candidate_1_z)
            if inward_candidate_1_x < inward_candidate_2_x
            else (inward_candidate_2_x, inward_candidate_2_z)
        )
        inward_unit_vector = Point(
            inward_unit_x,
            inward_unit_z * self.caster_sin,
            inward_unit_z * self.caster_cos,
        )
        return (
            lower_trunnion_point
            + Point(
                inward_unit_vector.x * self.kingpin_lower_inward_in,
                inward_unit_vector.y * self.kingpin_lower_inward_in,
                inward_unit_vector.z * self.kingpin_lower_inward_in,
            ),
            upper_trunnion_point
            + Point(
                inward_unit_vector.x * self.kingpin_upper_inward_in,
                inward_unit_vector.y * self.kingpin_upper_inward_in,
                inward_unit_vector.z * self.kingpin_upper_inward_in,
            ),
        )

    def axle(self, lower_angle_rad: float) -> Point:
        """Return the axle kingpin rotation point at the supplied arm angle.

        Returns:
            The axle point in inches.

        """
        lower_kingpin_point, upper_kingpin_point = self.kingpin_axis(
            lower_angle_rad,
        )
        kingpin_vector = upper_kingpin_point - lower_kingpin_point
        kingpin_length_in = abs(kingpin_vector)
        return lower_kingpin_point + Point(
            self.axle_offset_in * kingpin_vector.x / kingpin_length_in,
            self.axle_offset_in * kingpin_vector.y / kingpin_length_in,
            self.axle_offset_in * kingpin_vector.z / kingpin_length_in,
        )

    def axle_direction(self, lower_angle_rad: float) -> Point:
        """Return the unit axle direction from kingpin toward the wheel center.

        Positive axle-to-kingpin angle drops the axle toward the lower end of
        the kingpin, relative to a perpendicular axle.

        Returns:
            The outward axle unit vector.

        """
        lower_kingpin_point, upper_kingpin_point = self.kingpin_axis(
            lower_angle_rad,
        )
        kingpin_vector = upper_kingpin_point - lower_kingpin_point
        kingpin_length_in = abs(kingpin_vector)
        kingpin_unit_vector = Point(
            kingpin_vector.x / kingpin_length_in,
            kingpin_vector.y / kingpin_length_in,
            kingpin_vector.z / kingpin_length_in,
        )
        kingpin_plane_z_in = kingpin_vector.z / self.caster_cos
        kingpin_plane_length_in = float(
            np.hypot(kingpin_vector.x, kingpin_plane_z_in),
        )
        inward_unit_vector = Point(
            -kingpin_plane_z_in / kingpin_plane_length_in,
            kingpin_vector.x / kingpin_plane_length_in * self.caster_sin,
            kingpin_vector.x / kingpin_plane_length_in * self.caster_cos,
        )
        axle_cos = float(np.cos(self.axle_kingpin_angle_rad))
        axle_sin = float(np.sin(self.axle_kingpin_angle_rad))
        return Point(
            -inward_unit_vector.x * axle_cos - kingpin_unit_vector.x * axle_sin,
            -inward_unit_vector.y * axle_cos - kingpin_unit_vector.y * axle_sin,
            -inward_unit_vector.z * axle_cos - kingpin_unit_vector.z * axle_sin,
        )

    def rotate_vector_about_kingpin(
        self,
        vector: Point,
        lower_angle_rad: float,
        toe_angle_rad: float,
    ) -> Point:
        """Rotate a vector about the lower-to-upper kingpin axis.

        Positive toe is a negative right-hand rotation around the kingpin so that
        the wheel moves toward negative longitudinal position.

        Returns:
            The toe-rotated vector.

        """
        lower_kingpin_point, upper_kingpin_point = self.kingpin_axis(
            lower_angle_rad,
        )
        kingpin_vector = upper_kingpin_point - lower_kingpin_point
        kingpin_length_in = abs(kingpin_vector)
        kingpin_unit_x = kingpin_vector.x / kingpin_length_in
        kingpin_unit_y = kingpin_vector.y / kingpin_length_in
        kingpin_unit_z = kingpin_vector.z / kingpin_length_in
        toe_cos = float(np.cos(-toe_angle_rad))
        toe_sin = float(np.sin(-toe_angle_rad))
        kingpin_vector_dot_product = (
            kingpin_unit_x * vector.x
            + kingpin_unit_y * vector.y
            + kingpin_unit_z * vector.z
        )
        return Point(
            vector.x * toe_cos
            + (kingpin_unit_y * vector.z - kingpin_unit_z * vector.y) * toe_sin
            + kingpin_unit_x * kingpin_vector_dot_product * (1.0 - toe_cos),
            vector.y * toe_cos
            + (kingpin_unit_z * vector.x - kingpin_unit_x * vector.z) * toe_sin
            + kingpin_unit_y * kingpin_vector_dot_product * (1.0 - toe_cos),
            vector.z * toe_cos
            + (kingpin_unit_x * vector.y - kingpin_unit_y * vector.x) * toe_sin
            + kingpin_unit_z * kingpin_vector_dot_product * (1.0 - toe_cos),
        )

    def wheel_center(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the wheel center at the supplied lower control-arm angle.

        Returns:
            The wheel center point in inches.

        """
        wheel_axis_vector = self.wheel_axis(lower_angle_rad, toe_angle_rad)
        return self.axle(lower_angle_rad) + Point(
            self.axle_length_in * wheel_axis_vector.x,
            self.axle_length_in * wheel_axis_vector.y,
            self.axle_length_in * wheel_axis_vector.z,
        )

    def wheel_axis(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the unit axle direction after the steering rotation.

        Returns:
            The wheel-axis unit vector toward the wheel center.

        """
        axle_direction = self.axle_direction(lower_angle_rad)
        return self.rotate_vector_about_kingpin(
            Point(
                axle_direction.x,
                axle_direction.y,
                axle_direction.z,
            ),
            lower_angle_rad,
            toe_angle_rad,
        )

    def wheel_contact(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the tire contact point at the supplied lower control-arm angle.

        Returns:
            The wheel contact point in inches.

        """
        wheel_axis_vector = self.wheel_axis(lower_angle_rad, toe_angle_rad)
        downward_axis_dot_product = -wheel_axis_vector.z
        downward_in_wheel_plane = Point(
            -downward_axis_dot_product * wheel_axis_vector.x,
            -downward_axis_dot_product * wheel_axis_vector.y,
            -1.0 - downward_axis_dot_product * wheel_axis_vector.z,
        )
        downward_in_wheel_plane_length_in = abs(downward_in_wheel_plane)
        return self.wheel_center(lower_angle_rad, toe_angle_rad) + Point(
            self.wheel_radius_in
            * downward_in_wheel_plane.x
            / downward_in_wheel_plane_length_in,
            self.wheel_radius_in
            * downward_in_wheel_plane.y
            / downward_in_wheel_plane_length_in,
            self.wheel_radius_in
            * downward_in_wheel_plane.z
            / downward_in_wheel_plane_length_in,
        )

    def tie_rod_point(self, lower_angle_rad: float, toe_angle_rad: float) -> Point:
        """Return the outer tie-rod point in the wheel-disc coordinate frame.

        The tie-rod point is inboard along the wheel spindle, upward along the
        kingpin projected into the wheel-disc plane, and forward in that plane.

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
        lower_kingpin_point, upper_kingpin_point = self.kingpin_axis(
            lower_angle_rad,
        )
        kingpin_vector = upper_kingpin_point - lower_kingpin_point
        kingpin_length_in = abs(kingpin_vector)
        kingpin_unit_x = kingpin_vector.x / kingpin_length_in
        kingpin_unit_y = kingpin_vector.y / kingpin_length_in
        kingpin_unit_z = kingpin_vector.z / kingpin_length_in
        kingpin_outward_dot_product = (
            kingpin_unit_x * wheel_outward_unit_x
            + kingpin_unit_y * wheel_outward_unit_y
            + kingpin_unit_z * wheel_outward_unit_z
        )
        wheel_up_vector = Point(
            kingpin_unit_x - kingpin_outward_dot_product * wheel_outward_unit_x,
            kingpin_unit_y - kingpin_outward_dot_product * wheel_outward_unit_y,
            kingpin_unit_z - kingpin_outward_dot_product * wheel_outward_unit_z,
        )
        wheel_up_length_in = abs(wheel_up_vector)
        wheel_up_unit_x = wheel_up_vector.x / wheel_up_length_in
        wheel_up_unit_y = wheel_up_vector.y / wheel_up_length_in
        wheel_up_unit_z = wheel_up_vector.z / wheel_up_length_in
        wheel_forward_vector = Point(
            wheel_outward_unit_y * wheel_up_unit_z
            - wheel_outward_unit_z * wheel_up_unit_y,
            wheel_outward_unit_z * wheel_up_unit_x
            - wheel_outward_unit_x * wheel_up_unit_z,
            wheel_outward_unit_x * wheel_up_unit_y
            - wheel_outward_unit_y * wheel_up_unit_x,
        )
        wheel_forward_length_in = abs(wheel_forward_vector)
        wheel_forward_unit_x = wheel_forward_vector.x / wheel_forward_length_in
        wheel_forward_unit_y = wheel_forward_vector.y / wheel_forward_length_in
        wheel_forward_unit_z = wheel_forward_vector.z / wheel_forward_length_in
        return wheel_center_point + Point(
            -self.tie_rod_inward_in * wheel_outward_unit_x
            + self.tie_rod_up_in * wheel_up_unit_x
            + self.tie_rod_forward_in * wheel_forward_unit_x,
            -self.tie_rod_inward_in * wheel_outward_unit_y
            + self.tie_rod_up_in * wheel_up_unit_y
            + self.tie_rod_forward_in * wheel_forward_unit_y,
            -self.tie_rod_inward_in * wheel_outward_unit_z
            + self.tie_rod_up_in * wheel_up_unit_z
            + self.tie_rod_forward_in * wheel_forward_unit_z,
        )

    def steering_rod_link_point(
        self,
        lower_angle_rad: float,
        toe_angle_rad: float,
    ) -> Point:
        """Return the tie-rod link point on the positive-x steering rod.

        The steering rod is parallel to the x axis at fixed longitudinal and
        vertical coordinates. The returned point is the inboard endpoint of the
        fixed-length tie-rod link.

        Returns:
            The steering-rod link point in inches.

        """
        tie_rod_point = self.tie_rod_point(lower_angle_rad, toe_angle_rad)
        steering_rod_link_point_x_in = tie_rod_point.x - float(
            np.sqrt(
                self.tie_rod_link_in**2
                - (tie_rod_point.y - self.steering_rod_y_in) ** 2
                - (tie_rod_point.z - self.steering_rod_z_in) ** 2,
            ),
        )
        return Point(
            steering_rod_link_point_x_in,
            self.steering_rod_y_in,
            self.steering_rod_z_in,
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
