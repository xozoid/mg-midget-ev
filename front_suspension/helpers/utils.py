"""Angle and vector operations for suspension geometry."""

from __future__ import annotations

from functools import cache, cached_property
from operator import index
from typing import cast, NamedTuple, SupportsIndex

import numpy as np

DEFAULT_TOL = 0.01


class Angle(float):
    """An angle stored in radians, with angle-preserving arithmetic."""

    def __add__(self, other: float) -> Angle:
        """Add the operands.

        Returns:
            The resulting value.

        """
        result = float.__add__(self, other)
        return NotImplemented if result is NotImplemented else Angle(result)

    def __radd__(self, other: float) -> Angle:
        """Add the operands in reverse order.

        Returns:
            The resulting value.

        """
        return self + other

    def __sub__(self, other: float) -> Angle:
        """Subtract the operands.

        Returns:
            The resulting value.

        """
        result = float.__sub__(self, other)
        return NotImplemented if result is NotImplemented else Angle(result)

    def __rsub__(self, other: float) -> Angle:
        """Subtract the operands in reverse order.

        Returns:
            The resulting value.

        """
        result = float.__rsub__(self, other)
        return NotImplemented if result is NotImplemented else Angle(result)

    def __mul__(self, other: float) -> Angle:
        """Multiply by the operand.

        Returns:
            The resulting value.

        """
        result = float.__mul__(self, other)
        return NotImplemented if result is NotImplemented else Angle(result)

    def __rmul__(self, other: float) -> Angle:
        """Multiply by the operand in reverse order.

        Returns:
            The resulting value.

        """
        return self * other

    def __truediv__(self, other: float) -> Angle:
        """Divide by the operand.

        Returns:
            The resulting value.

        """
        result = float.__truediv__(self, other)
        return NotImplemented if result is NotImplemented else Angle(result)

    def __rtruediv__(self, other: float) -> Angle:
        """Divide the operands in reverse order.

        Returns:
            The resulting value.

        """
        result = float.__rtruediv__(self, other)
        return NotImplemented if result is NotImplemented else Angle(result)

    def __neg__(self) -> Angle:
        """Negate the angle.

        Returns:
            The resulting value.

        """
        return Angle(float.__neg__(self))

    def __pos__(self) -> Angle:
        """Return the positive angle.

        Returns:
            The resulting value.

        """
        return Angle(self)

    def __abs__(self) -> Angle:
        """Return the absolute magnitude.

        Returns:
            The resulting value.

        """
        return Angle(float.__abs__(self))

    @cached_property
    def deg(self) -> float:
        """The angle in degrees."""
        return np.rad2deg(self)

    @cached_property
    def sin(self) -> float:
        """The sine of the angle."""
        return np.sin(self)

    @classmethod
    def asin(cls, x: float) -> Angle:
        """Construct an angle from its sine.

        Returns:
            The resulting value.

        """
        return cls(np.arcsin(x))

    @cached_property
    def cos(self) -> float:
        """The cosine of the angle."""
        return np.cos(self)

    @classmethod
    def acos(cls, x: float) -> Angle:
        """Construct an angle from its cosine.

        Returns:
            The resulting value.

        """
        return cls(np.arccos(x))

    @cached_property
    def tan(self) -> float:
        """The tangent of the angle."""
        return np.tan(self)

    @classmethod
    def atan(cls, y: float, x: float) -> Angle:
        """Construct an angle from its Cartesian components.

        Returns:
            The resulting value.

        """
        return cls(np.arctan2(y, x))

    @classmethod
    def from_deg(cls, x: float) -> Angle:
        """Construct an angle from degrees.

        Returns:
            The resulting value.

        """
        return cls(np.deg2rad(x))


HALF_PI = Angle(np.pi / 2)


class Point(NamedTuple):
    """Chassis-fixed X, Y, Z coordinates in inches."""

    x: float
    y: float
    z: float

    def __add__(self, other: tuple[object, ...]) -> Point:
        """Add the operands.

        Returns:
            The resulting value.

        """
        values = cast("tuple[float, ...]", other)
        return Point(*(a + b for a, b in zip(self, values, strict=True)))

    def __sub__(self, other: Point) -> Point:
        """Subtract the operands.

        Returns:
            The resulting value.

        """
        return Point(*(a - b for a, b in zip(self, other, strict=True)))

    def __abs__(self) -> float:
        """Return the absolute magnitude.

        Returns:
            The resulting value.

        """
        return float(np.linalg.norm(self))

    def __mul__(self, scalar: float | SupportsIndex) -> Point:
        """Multiply by the operand.

        Returns:
            The resulting value.

        """
        factor = scalar if isinstance(scalar, float) else index(scalar)
        return Point(*(a * factor for a in self))

    def __truediv__(self, scalar: float) -> Point:
        """Divide by the operand.

        Returns:
            The resulting value.

        """
        return self * (1 / scalar)

    def __rmul__(self, scalar: float | SupportsIndex) -> Point:
        """Multiply the vector by a scalar.

        Returns:
            The scaled vector.

        """
        return self * scalar

    @property
    def unit(self) -> Point:
        """The unit vector."""
        return self / abs(self)

    def rotate(self, axis: Point, angle: Angle) -> Point:
        """Rotate this vector about an axis using the right-hand rule.

        Returns:
            The rotated vector.

        """
        direction = np.asarray(axis.unit)
        vector = np.asarray(self)
        result = (
            vector * angle.cos
            + np.cross(direction, vector) * angle.sin
            + direction * np.dot(direction, vector) * (1 - angle.cos)
        )
        return Point(*result)

    def __matmul__(self, other: Point) -> Point:
        """Calculate the cross product of two vectors.

        Returns:
            The cross product.

        """
        return Point(*np.cross(self, other))


@cache
def approx(a: float, b: float, *, tol: float = DEFAULT_TOL) -> bool:
    """Return whether two values differ by less than the tolerance.

    Returns:
        The resulting value.

    """
    return abs(a - b) < tol
