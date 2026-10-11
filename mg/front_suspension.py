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

    lower_in: float = 12
    lower_pickup_width_in: float = 36
    upper_in: float = 9
    inner_in: float = 10
    inner_angle_rad: float = float(np.deg2rad(5))
    outer_in: float = 9
    caster_rad: float = float(np.deg2rad(3.0))

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
        d = abs(self.upper_pickup() - self.lower_trunnion(lower_angle_rad))
        theta_lower_inner = self.inner_angle_rad + float(np.pi) / 2.0 - lower_angle_rad

        theta_lower_d = np.arcsin(np.sin(theta_lower_inner) / d * self.inner_in)
        theta_outer_d = np.arccos(
            (d**2 + self.outer_in**2 - self.upper_in**2) / (2 * d * self.outer_in),
        )

        theta_lower_outer = theta_lower_d + theta_outer_d

        theta_outer_horizon = theta_lower_outer - lower_angle_rad
        outer_projection_in = self.lower_in * float(np.sin(theta_outer_horizon))
        return self.lower_trunnion(lower_angle_rad) + Point(
            -self.outer_in * float(np.cos(theta_outer_horizon)),
            outer_projection_in * self.caster_sin,
            outer_projection_in * self.caster_cos,
        )
