"""Helpers for plotting three-dimensional suspension geometry in 2D."""

from __future__ import annotations

import re
from typing import Any, cast, TYPE_CHECKING

import numpy as np

from mg.front_suspension.utils import Point

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_PROJECTION_AXES = 2
_MIN_POLYGON_POINTS = 3


class PlotUtils:
    """Draw points projected onto two signed coordinate axes, such as ``-YZ``."""

    def __init__(self, ax: Axes, projection: str, **kwargs: object) -> None:
        """Set up an axes, projection, and line options.

        Raises:
            ValueError: If the projection is invalid.

        """
        axes = re.findall(r"[+-]?[XYZ]", projection.upper())
        if (
            "".join(axes) != projection.upper()
            or len(axes) != _PROJECTION_AXES
            or axes[0][-1] == axes[1][-1]
        ):
            msg = f"Invalid projection: {projection!r}"
            raise ValueError(msg)
        self.ax = ax
        self.axes = axes
        self.kwargs = kwargs
        self._label_used = False

    def _project(self, point: Point) -> tuple[float, float]:
        def component(axis: str) -> float:
            sign = -1 if axis.startswith("-") else 1
            return sign * point["XYZ".index(axis[-1])]

        return component(self.axes[0]), component(self.axes[1])

    def _draw(self, points: list[Point], **kwargs: object) -> None:
        coordinates = [self._project(point) for point in points]
        options = dict(self.kwargs)
        if self._label_used:
            options.pop("label", None)
        options.update(kwargs)
        self.ax.plot(*zip(*coordinates, strict=True), **cast("Any", options))
        self._label_used = True

    def line(self, start: Point, end: Point, **kwargs: object) -> None:
        """Draw a line between two points."""
        self._draw([start, end], **kwargs)

    def poly(self, *points: Point, **kwargs: object) -> None:
        """Draw a closed polygon through the points.

        Raises:
            ValueError: If fewer than three points are supplied.

        """
        if len(points) < _MIN_POLYGON_POINTS:
            msg = "A polygon needs at least three points"
            raise ValueError(msg)
        self._draw([*points, points[0]], **kwargs)

    def ellipse(
        self,
        center: Point,
        axis: Point,
        radius: float,
        **kwargs: object,
    ) -> None:
        """Project a circular wheel whose plane is normal to ``axis``.

        Raises:
            ValueError: If the radius is negative or the axis is zero.

        """
        if radius < 0 or abs(axis) == 0:
            msg = "Wheel radius must be nonnegative and axis nonzero"
            raise ValueError(msg)
        normal = np.asarray(axis.unit)
        reference = np.eye(3)[np.argmin(np.abs(normal))]
        first = np.cross(normal, reference)
        first /= np.linalg.norm(first)
        second = np.cross(normal, first)
        angles = np.linspace(0, 2 * np.pi, 181)
        points = [
            center + Point(*(radius * (first * np.cos(a) + second * np.sin(a))))
            for a in angles
        ]
        self._draw(points, **kwargs)
