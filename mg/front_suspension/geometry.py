"""Front-suspension states for prescribed wheel travel and rack displacement.

Origin is x=0 is centerline, y=0 is well wall, z=0 is ground at rest
All lengths are inches
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from mg.front_suspension.constraints import BUMP
from mg.front_suspension.plot_utils import PlotUtils
from mg.front_suspension.utils import Angle, DEFAULT_TOL, HALF_PI, Point

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes


@dataclass(frozen=True)
class SuspensionState:
    """A solved suspension configuration in chassis coordinates."""

    lp: Point
    up: Point
    lf: Point
    uf: Point
    rack: Point
    tie_rod: Point

    wheel_r: float
    wheel_center: Point
    _wheel_contact_z: float

    @cached_property
    def wheel_contact(self) -> Point:
        """The wheel contact point."""
        return Point(self.wheel_center.x, self.wheel_center.y, self._wheel_contact_z)

    axle_center: Point
    axle_axis: Point  # unit vector originating at axle_center, pointing outward
    kp_axis: Point  # unit vector originating at axle_center, pointing upward

    @cached_property
    def _instant_center_homogeneous(self) -> np.ndarray:
        lower = np.cross([self.lp.x, self.lp.z, 1], [self.lf.x, self.lf.z, 1])
        upper = np.cross([self.up.x, self.up.z, 1], [self.uf.x, self.uf.z, 1])
        return np.cross(lower, upper)

    @cached_property
    def instant_center(self) -> Point:
        """The front-view arm intersection, on the lower arm's line.

        Raises:
            ValueError: If the projected arms have no unique finite intersection.

        """
        instant = self._instant_center_homogeneous
        if np.isclose(instant[2], 0):
            msg = "Instant center has no unique finite position"
            raise ValueError(msg)
        x_in = float(instant[0] / instant[2])
        z_in = float(instant[1] / instant[2])
        arm = self.lf - self.lp
        # NOTE: Use the larger projected component to avoid dividing by a
        # nearly vertical or horizontal arm component.
        fraction = (
            (x_in - self.lp.x) / arm.x
            if abs(arm.x) >= abs(arm.z)
            else (z_in - self.lp.z) / arm.z
        )
        return Point(x_in, self.lp.y + fraction * arm.y, z_in)

    @cached_property
    def roll_center(self) -> Point:
        """The symmetric roll center from the front-view arm intersections.

        Raises:
            ValueError: If the roll center is undefined.

        """
        instant = self._instant_center_homogeneous
        contact = self.wheel_contact
        denominator = contact.x * instant[2] - instant[0]
        # NOTE: Homogeneous coordinates also handle parallel arms, whose
        # instant center is at infinity but whose roll center can be finite.
        if np.isclose(denominator, 0):
            msg = "Roll center is undefined for this suspension geometry"
            raise ValueError(msg)
        height_in = (contact.x * instant[1] - contact.z * instant[0]) / denominator
        return Point(0, contact.y, float(height_in))

    @cached_property
    def kpi(self) -> Angle:
        """The kingpin inclination."""
        return Angle.atan(-self.kp_axis.x, self.kp_axis.z)

    @cached_property
    def camber(self) -> Angle:
        """The wheel camber angle."""
        return -Angle.atan(self.axle_axis.z, self.axle_axis.x)

    @cached_property
    def caster(self) -> Angle:
        """The kingpin caster angle."""
        return Angle.atan(-self.kp_axis.y, self.kp_axis.z)

    @cached_property
    def toe(self) -> Angle:
        # toe-in: toe > 0
        """The wheel toe angle."""
        return Angle.atan(self.axle_axis.y, self.axle_axis.x)

    def plot(
        self,
        ax: Axes,
        projection: str,
        *,
        roll_center_construction: bool = True,
        **kwargs: object,
    ) -> None:
        """Plot the suspension on the requested projection."""
        plt = PlotUtils(ax, projection, **kwargs)
        plt.poly(self.lp, self.lf, self.uf, self.up)
        lower = float(np.dot(self.lf - self.axle_center, self.kp_axis))
        upper = float(np.dot(self.uf - self.axle_center, self.kp_axis))
        plt.line(
            self.axle_center + lower * self.kp_axis,
            self.axle_center + upper * self.kp_axis,
            color="blue",
        )
        plt.line(self.axle_center, self.wheel_center, linestyle=":")
        plt.ellipse(self.wheel_center, self.axle_axis, self.wheel_r)
        plt.line(self.rack, self.tie_rod, linestyle=":")
        plt.line(self.roll_center, self.roll_center, marker="o", linestyle="none")
        # NOTE: Parallel arms have an instant center at infinity, so there
        # is no finite construction point to draw.
        if roll_center_construction and not np.isclose(
            self._instant_center_homogeneous[2],
            0,
        ):
            instant = self.instant_center
            plt.line(self.lp, instant, linestyle="--")
            plt.line(self.up, instant, linestyle="--")
            plt.line(instant, self.wheel_contact, linestyle="--")
            plt.line(instant, instant, marker="o", linestyle="none")

    @cached_property
    def world(self) -> SuspensionState:
        """The state translated to ground level."""
        offset = Point(0, 0, self._wheel_contact_z)
        return replace(
            self,
            lp=self.lp - offset,
            up=self.up - offset,
            lf=self.lf - offset,
            uf=self.uf - offset,
            rack=self.rack - offset,
            tie_rod=self.tie_rod - offset,
            wheel_center=self.wheel_center - offset,
            _wheel_contact_z=0.0,
            axle_center=self.axle_center - offset,
        )


@dataclass(frozen=True)
class FourBar:
    # Inputs
    """A planar four-bar linkage."""

    inner_elevation: Angle

    inner: float
    lower: float
    upper: float
    outer: float

    @cached_property
    def il(self) -> Point:
        """The inner lower pivot."""
        return Point(0, 0, 0)

    @cached_property
    def iu(self) -> Point:
        """The inner upper pivot."""
        return Point(
            self.inner * self.inner_elevation.cos,
            self.inner * self.inner_elevation.sin,
            0,
        )

    @cached_property
    def ol(self) -> Point:
        """The outer lower pivot."""
        return Point(self.lower, 0, 0)

    @cached_property
    def ou(self) -> Point:
        """The outer upper pivot.

        Raises:
            ValueError: If the links cannot intersect.

        """
        delta = self.ol - self.iu
        distance = abs(delta)
        if (
            distance == 0
            or distance > self.upper + self.outer
            or distance < abs(self.upper - self.outer)
        ):
            msg = "Four-bar links do not intersect"
            raise ValueError(msg)
        along = (self.upper**2 - self.outer**2 + distance**2) / (2 * distance)
        height = np.sqrt(max(0.0, self.upper**2 - along**2))
        return Point(
            self.iu.x + (along * delta.x - height * delta.y) / distance,
            self.iu.y + (along * delta.y + height * delta.x) / distance,
            0,
        )


@dataclass(frozen=True)
class GeometryFixed:
    """Fixed dimensions and angles for one side of the front suspension."""

    # Positive caster tilts the top of the wheel towards the driver
    caster: Angle = field(default_factory=lambda: Angle.from_deg(1.8))

    # lp = lower pickup
    lp_x: float = 18.194 / 2
    lp_y: float = 13.827
    lp_z: float = 7.998

    # up = upper pickup
    up_x: float = 13.800
    up_z: float = 15.668

    arm_lower: float = 10.75
    arm_upper: float = 7.25
    arm_outer: float = 8.228

    arm_tie_rod: float = 8

    rack_x_rest: float = 27 / 2
    rack_y: float = 18.555
    rack_z: float = 12.997
    steering: float = 0

    # w[x,y,z] is in the wheel's coordinate system
    # all vectors parallel to global when all angles are zero
    tie_rod_wx: float = -2.125
    tie_rod_wy: float = 5
    tie_rod_wz: float = 1.5

    wheel_r: float = 22.6 / 2

    # kp[x,y,z] is in the kingpin's coordinate system
    kp_uf_kpx: float = 0.75
    kp_lf_kpx: float = -1

    axle_kpz: float = 3.682

    axle_length: float = 3.093
    axle_rotation_four_bar: Angle = field(
        default_factory=lambda: Angle.from_deg(-96.48),
    )

    @cached_property
    def lp(self) -> Point:
        """The lower pickup point."""
        return Point(self.lp_x, self.lp_y, self.lp_z)

    @cached_property
    def four_bar_axis(self) -> Point:
        # Y-axis rotated -caster about X-axis
        """The four-bar rotation axis."""
        return Point(0, -self.caster.cos, -self.caster.sin)

    @cached_property
    def up_y(self) -> float:
        # Positive caster tilts the top of the wheel towards the driver (-Y)
        """The upper pickup longitudinal coordinate."""
        return self.lp_y - (self.up_z - self.lp_z) * self.caster.tan

    @cached_property
    def up(self) -> Point:
        """The upper pickup point."""
        return Point(self.up_x, self.up_y, self.up_z)

    @cached_property
    def arm_inner(self) -> float:
        """The distance between inner pickups."""
        return abs(self.up - self.lp)

    @cached_property
    def rack(self) -> Point:
        """The steering rack pivot."""
        return Point(self.rack_x_rest + self.steering, self.rack_y, self.rack_z)

    @cached_property
    def kp_rotation_four_bar(self) -> Angle:
        """The kingpin angle relative to the four-bar linkage.

        Raises:
            ValueError: If the kingpin offsets exceed the fulcrum separation.

        """
        x = self.kp_uf_kpx - self.kp_lf_kpx
        if abs(x) > self.arm_outer:
            msg = "Kingpin offsets exceed fulcrum separation"
            raise ValueError(msg)
        return Angle.asin(x / self.arm_outer)

    def plot(
        self,
        path: Path,
        *,
        droop: float = BUMP[0],
        bump: float = BUMP[1],
        roll_center_construction: bool = False,
    ) -> None:
        """Save suspension geometry plots at rest, droop, and bump."""
        states = {
            "rest": (0, "green"),
            "droop": (droop, "orange"),
            "bump": (bump, "red"),
        }
        fig, ax = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
        for label, (height, color) in states.items():
            state = Solver.at(self, height)
            state.plot(
                ax[0, 0],
                "XZ",
                label=label,
                color=color,
                roll_center_construction=roll_center_construction,
            )
            state.plot(
                ax[0, 1],
                "-YZ",
                label=label,
                color=color,
                roll_center_construction=roll_center_construction,
            )
            state.plot(
                ax[0, 2],
                "-XY",
                label=label,
                color=color,
                roll_center_construction=roll_center_construction,
            )
            state.world.plot(
                ax[1, 0],
                "XZ",
                label=label,
                color=color,
                roll_center_construction=roll_center_construction,
            )
            state.world.plot(
                ax[1, 1],
                "-YZ",
                label=label,
                color=color,
                roll_center_construction=roll_center_construction,
            )
            state.world.plot(
                ax[1, 2],
                "-XY",
                label=label,
                color=color,
                roll_center_construction=roll_center_construction,
            )

        for axes, title, xlabel, ylabel in (
            (ax[0, 0], "Chassis: front view", "X (in)", "Z (in)"),
            (ax[0, 1], "Chassis: side view", "-Y (in)", "Z (in)"),
            (ax[0, 2], "Chassis: top view", "-X (in)", "Y (in)"),
            (ax[1, 0], "Ground: front view", "X (in)", "Z (in)"),
            (ax[1, 1], "Ground: side view", "-Y (in)", "Z (in)"),
            (ax[1, 2], "Ground: top view", "-X (in)", "Y (in)"),
        ):
            axes.set_title(title)
            axes.set_xlabel(xlabel)
            axes.set_ylabel(ylabel)
            axes.set_aspect("equal", adjustable="box")
            axes.grid(visible=True, alpha=0.3)
            axes.legend()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"Saved {path}")


@dataclass(frozen=True)
class Solver:
    """Solve suspension geometry for arm elevation and steering."""

    g: GeometryFixed

    # Inputs
    # Positive is bump
    _arm_lower_elevation: Angle = field(default_factory=lambda: Angle(0))
    # Positive is left turn
    _swivel: Angle = field(default_factory=lambda: Angle(0))

    @cached_property
    def _four_bar(self) -> FourBar:
        inner_elevation = Angle.atan(
            (self.g.up.z - self.g.lp.z) / self.g.caster.cos,
            self.g.up.x - self.g.lp.x,
        )
        return FourBar(
            inner_elevation - self._arm_lower_elevation,
            self.g.arm_inner,
            self.g.arm_lower,
            self.g.arm_upper,
            self.g.arm_outer,
        )

    def _from_four_bar(self, point: Point) -> Point:
        a = self._arm_lower_elevation
        x = point.x * a.cos - point.y * a.sin
        z = point.x * a.sin + point.y * a.cos
        # Positive caster tilts the top of the wheel towards the driver (-Y)
        return self.g.lp + Point(x, -z * self.g.caster.sin, z * self.g.caster.cos)

    # lf = lower fulcrum
    @cached_property
    def lf(self) -> Point:
        """The lower fulcrum point."""
        return self._from_four_bar(self._four_bar.ol)

    # uf = upper fulcrum
    @cached_property
    def uf(self) -> Point:
        """The upper fulcrum point."""
        return self._from_four_bar(self._four_bar.ou)

    @cached_property
    def outer_axis(self) -> Point:
        """The unit vector along the outer link."""
        return (self.uf - self.lf).unit

    @cached_property
    def _kpz(self) -> Point:
        return self.outer_axis.rotate(
            self.g.four_bar_axis,
            self.g.kp_rotation_four_bar,
        )

    @cached_property
    def _kpx(self) -> Point:
        return self._kpz.rotate(self.g.four_bar_axis, -HALF_PI)

    @property
    def kp_axis(self) -> Point:
        """The kingpin axis."""
        return self._kpz

    @cached_property
    def _kp0(self) -> Point:
        return self.lf - self.g.kp_lf_kpx * self._kpx

    @cached_property
    def axle_center(self) -> Point:
        """The axle center at the kingpin."""
        return self._kp0 + self.g.axle_kpz * self._kpz

    @cached_property
    def _wx(self) -> Point:
        wx_straight = self._kpz.rotate(
            self.g.four_bar_axis,
            self.g.axle_rotation_four_bar,
        )
        return wx_straight.rotate(self.kp_axis, self._swivel)

    @cached_property
    def _wy(self) -> Point:
        return self._wz @ self._wx

    @cached_property
    def _wz(self) -> Point:
        wz_straight = self._kpz.rotate(
            self.g.four_bar_axis,
            self.g.axle_rotation_four_bar + HALF_PI,
        )
        return wz_straight.rotate(self.kp_axis, self._swivel)

    @property
    def axle_axis(self) -> Point:
        """The outward axle axis."""
        return self._wx

    @cached_property
    def _w0(self) -> Point:
        return self.axle_center + self.g.axle_length * self._wx

    @property
    def wheel_center(self) -> Point:
        """The wheel center."""
        return self._w0

    @cached_property
    def wheel_contact_z(self) -> float:
        """The wheel contact height."""
        return self.wheel_center.z - self.g.wheel_r

    @cached_property
    def tie_rod(self) -> Point:
        """The outer tie-rod pivot."""
        dwx = self._wx * self.g.tie_rod_wx
        dwy = self._wy * self.g.tie_rod_wy
        dwz = self._wz * self.g.tie_rod_wz
        return self._w0 + dwx + dwy + dwz

    @cached_property
    def arm_tie_rod_derived(self) -> float:
        """The distance between tie-rod pivots."""
        return abs(self.g.rack - self.tie_rod)

    @classmethod
    def at(cls, fixed: GeometryFixed, bump: float) -> SuspensionState:
        """Solve for a wheel contact height in chassis coordinates.

        Returns:
            The solved suspension state.

        Raises:
            ValueError: If no state satisfies the constraints.

        """

        def residual(angles: np.ndarray) -> list[float]:
            g = cls(
                fixed,
                _arm_lower_elevation=Angle(angles[0]),
                _swivel=Angle(angles[1]),
            )
            return [g.arm_tie_rod_derived - fixed.arm_tie_rod, g.wheel_contact_z - bump]

        limits = np.deg2rad([20, 30])
        result = least_squares(residual, [0.0, 0.0], bounds=(-limits, limits))
        if not result.success or np.max(np.abs(result.fun)) >= DEFAULT_TOL:
            msg = f"No suspension state found for wheel contact height {bump}"
            raise ValueError(msg)
        return cls(
            fixed,
            _arm_lower_elevation=Angle(result.x[0]),
            _swivel=Angle(result.x[1]),
        ).suspension_state

    @cached_property
    def suspension_state(self) -> SuspensionState:
        """The solved suspension state."""
        return SuspensionState(
            lp=self.g.lp,
            up=self.g.up,
            lf=self.lf,
            uf=self.uf,
            rack=self.g.rack,
            tie_rod=self.tie_rod,
            wheel_r=self.g.wheel_r,
            wheel_center=self.wheel_center,
            _wheel_contact_z=self.wheel_contact_z,
            axle_center=self.axle_center,
            axle_axis=self.axle_axis,
            kp_axis=self.kp_axis,
        )
