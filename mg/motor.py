"""Motor and controller."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from mg.utils import LBFT_TO_NM


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    """Linear interpolation clamped to the curve endpoints.

    Args:
        xs: Monotonically increasing x breakpoints.
        ys: y values at each x breakpoint.
        x: Query x value.

    Returns:
        Interpolated y value, clamped at the curve endpoints.

    """
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])


@dataclass
class Motor:
    """Motor specifications."""

    name: str
    """Descriptive motor and controller name."""

    mass_kg: float
    """Motor and controller assembly mass in kilograms."""

    nominal_voltage_v: float
    """Nominal operating voltage in volts."""

    min_voltage_v: float
    """Minimum pack voltage in volts."""

    max_voltage_v: float
    """Maximum pack voltage in volts."""

    peak_pack_current_a: float
    """Peak pack current draw in amps."""

    continuous_power_kw: float
    """Continuous rated power output in kilowatts."""

    max_rpm: float
    """Maximum motor speed in revolutions per minute."""

    torque_curve_rpm: list[float]
    """RPM breakpoints for the torque and efficiency curves."""

    torque_curve_nm: list[float]
    """Torque in Newton-metres at each RPM breakpoint."""

    efficiency_curve_pct: list[float]
    """Motor efficiency as a percentage at each RPM breakpoint."""

    today_price_usd: float
    """Current sourced price for the motor and controller kit in USD."""

    @property
    def peak_torque_nm(self) -> float:
        """Peak torque in Newton-metres, taken as the maximum on the torque curve."""
        return max(self.torque_curve_nm)

    @property
    def peak_torque_lbft(self) -> float:
        """Peak torque in pound-feet."""
        return self.peak_torque_nm / LBFT_TO_NM

    @property
    def peak_power_kw(self) -> float:
        """Peak power in kilowatts, derived from the torque curve."""
        peak = 0.0
        for rpm, torque_nm in zip(
            self.torque_curve_rpm,
            self.torque_curve_nm,
            strict=True,
        ):
            if rpm > 0:
                omega = rpm * 2.0 * math.pi / 60.0
                power = torque_nm * omega / 1000.0
                peak = max(peak, power)
        return peak

    @property
    def assumed_cruise_efficiency(self) -> float:
        """Best motor efficiency from the curve as a fraction (0-1)."""
        return max(self.efficiency_curve_pct) / 100.0

    def available_torque_nm(self, rpm: float) -> float:
        """Available torque at the given RPM by interpolating the torque curve.

        Args:
            rpm: Motor speed in revolutions per minute.

        Returns:
            Available torque in Newton-metres.

        """
        return _interp(self.torque_curve_rpm, self.torque_curve_nm, rpm)

    def efficiency_at_rpm(self, rpm: float) -> float:
        """Motor efficiency at the given RPM by interpolating the efficiency curve.

        Args:
            rpm: Motor speed in revolutions per minute.

        Returns:
            Motor efficiency as a fraction (0-1).

        """
        return _interp(self.torque_curve_rpm, self.efficiency_curve_pct, rpm) / 100.0


HYPER9_144V = Motor(
    name="NetGain HyPer 9HV",
    nominal_voltage_v=144.0,
    min_voltage_v=90.0,
    max_voltage_v=180.0,
    peak_pack_current_a=500.0,
    mass_kg=63.5,
    continuous_power_kw=38.0,
    max_rpm=8000.0,
    torque_curve_rpm=[0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0],
    torque_curve_nm=[220.7, 220.5, 220.4, 219.5, 175.3, 137.7, 98.4],
    efficiency_curve_pct=[0.0, 88.0, 93.1, 94.9, 93.8, 92.3, 91.7],
    today_price_usd=5600.0,
)
