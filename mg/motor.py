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


def _peak_power_from_curve(rpms: list[float], torques_nm: list[float]) -> float:
    """Return the peak mechanical power in kW from paired RPM/torque lists.

    Args:
        rpms: Motor speed breakpoints in revolutions per minute.
        torques_nm: Torque values in Newton-metres at each breakpoint.

    Returns:
        Peak power in kilowatts.

    """
    peak = 0.0
    for rpm, torque_nm in zip(rpms, torques_nm, strict=True):
        if rpm > 0:
            omega = rpm * 2.0 * math.pi / 60.0
            peak = max(peak, torque_nm * omega / 1000.0)
    return peak


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

    max_rpm: float
    """Maximum motor speed in revolutions per minute."""

    peak_curve_rpm: list[float]
    """RPM breakpoints for the peak torque and efficiency curves."""

    peak_curve_nm: list[float]
    """Peak torque in Newton-metres at each RPM breakpoint."""

    peak_efficiency_pct: list[float]
    """Motor efficiency as a percentage at each peak-curve RPM breakpoint."""

    continuous_curve_rpm: list[float]
    """RPM breakpoints for the continuous torque and efficiency curves."""

    continuous_curve_nm: list[float]
    """Continuous torque in Newton-metres at each RPM breakpoint."""

    continuous_efficiency_pct: list[float]
    """Motor efficiency as a percentage at each continuous-curve RPM breakpoint."""

    today_price_usd: float
    """Current sourced price for the motor and controller kit in USD."""

    @property
    def peak_torque_nm(self) -> float:
        """Peak torque in Newton-metres, taken as the maximum on the peak curve."""
        return max(self.peak_curve_nm)

    @property
    def peak_torque_lbft(self) -> float:
        """Peak torque in pound-feet."""
        return self.peak_torque_nm / LBFT_TO_NM

    @property
    def peak_power_kw(self) -> float:
        """Peak power in kilowatts, derived from the peak torque curve."""
        return _peak_power_from_curve(self.peak_curve_rpm, self.peak_curve_nm)

    @property
    def continuous_power_kw(self) -> float:
        """Continuous power in kilowatts, derived from the continuous torque curve."""
        return _peak_power_from_curve(
            self.continuous_curve_rpm,
            self.continuous_curve_nm,
        )

    @property
    def assumed_cruise_efficiency(self) -> float:
        """Best motor efficiency from the continuous curve as a fraction (0-1)."""
        return max(self.continuous_efficiency_pct) / 100.0

    def available_torque_nm(self, rpm: float) -> float:
        """Available peak torque at the given RPM, interpolated from the peak curve.

        Args:
            rpm: Motor speed in revolutions per minute.

        Returns:
            Available torque in Newton-metres.

        """
        return _interp(self.peak_curve_rpm, self.peak_curve_nm, rpm)

    def continuous_torque_nm(self, rpm: float) -> float:
        """Continuous torque at the given RPM, interpolated from the continuous curve.

        Args:
            rpm: Motor speed in revolutions per minute.

        Returns:
            Continuous torque in Newton-metres.

        """
        return _interp(self.continuous_curve_rpm, self.continuous_curve_nm, rpm)

    def efficiency_at_rpm(self, rpm: float) -> float:
        """Motor efficiency at the given RPM, interpolated from the peak curve.

        Args:
            rpm: Motor speed in revolutions per minute.

        Returns:
            Motor efficiency as a fraction (0-1).

        """
        return _interp(self.peak_curve_rpm, self.peak_efficiency_pct, rpm) / 100.0

    def continuous_efficiency_at_rpm(self, rpm: float) -> float:
        """Motor efficiency at the given RPM, interpolated from the continuous curve.

        Args:
            rpm: Motor speed in revolutions per minute.

        Returns:
            Motor efficiency as a fraction (0-1).

        """
        return (
            _interp(self.continuous_curve_rpm, self.continuous_efficiency_pct, rpm)
            / 100.0
        )


CURTIS_AC50 = Motor(
    name="HPEVS AC-50 / Curtis 1238-7601 w/ 1238SE-7971",
    mass_kg=52.0 + 7,
    nominal_voltage_v=96.0,
    min_voltage_v=72.0,
    max_voltage_v=130.0,
    peak_pack_current_a=650.0,
    max_rpm=8000.0,
    # Peak curve: 650 A at 96 V supply (derived from datasheet peaks)
    peak_curve_rpm=[0.0, 3400.0, 5000.0, 6000, 7000, 8000],
    peak_curve_nm=[163.4, 148.8, 80, 52, 35, 25.2],
    peak_efficiency_pct=[0, 87.0, 90.5, 85.6, 83.8, 83.5],
    continuous_curve_rpm=[1000, 2000, 3000, 4000, 5000, 6000, 7000],
    continuous_curve_nm=[35, 50, 55, 68, 57, 43, 31],
    continuous_efficiency_pct=[79.5, 87.2, 89.9, 89.9, 87.5, 85.2, 84.4],
    today_price_usd=2400.0 + 2200,
)


HYPER9_120V = Motor(
    name="NetGain HyPer 9",
    mass_kg=63.5,
    nominal_voltage_v=120.0,
    min_voltage_v=72.0,
    max_voltage_v=132.0,
    peak_pack_current_a=500.0,
    max_rpm=8000.0,
    # Peak curve: 850 A at 120 V supply
    peak_curve_rpm=[0, 1200, 2400, 3600, 4800, 6000, 7200],
    peak_curve_nm=[235.1, 235, 234.8, 234.6, 204.5, 161.8, 120.6],
    peak_efficiency_pct=[0, 87.4, 92.8, 94.8, 94.3, 92.9, 91.9],
    # Continuous curve: 350 A at 108 V supply
    continuous_curve_rpm=[0.0, 1000.0, 2500.0, 4000.0, 5500.0, 7000.0, 8500.0],
    continuous_curve_nm=[88.8, 88.7, 88.5, 88.4, 72.1, 55.8, 44.7],
    continuous_efficiency_pct=[0.0, 91.7, 95.4, 96.2, 95, 93.1, 91.1],
    today_price_usd=5400.0,
)

HYPER9_144V = Motor(
    name="NetGain HyPer 9HV",
    mass_kg=63.5,
    nominal_voltage_v=144.0,
    min_voltage_v=90.0,
    max_voltage_v=180.0,
    peak_pack_current_a=500.0,
    max_rpm=8500.0,
    # Peak curve: 600 A at 144 V supply
    peak_curve_rpm=[0.0, 1200.0, 2400.0, 3600.0, 4800.0, 6000.0, 7200.0],
    peak_curve_nm=[220.7, 220.5, 220.4, 219.5, 175.3, 137.7, 98.4],
    peak_efficiency_pct=[0.0, 88.0, 93.1, 94.9, 93.8, 92.3, 91.7],
    # Continuous curve: 250 A at 156 V supply
    continuous_curve_rpm=[0.0, 1000.0, 2500.0, 4000.0, 5500.0, 7000.0, 8500.0],
    continuous_curve_nm=[83.9, 83.8, 83.7, 83.5, 74.2, 57.9, 46.5],
    continuous_efficiency_pct=[0.0, 91.7, 95.4, 96.2, 95.4, 93.6, 91.6],
    today_price_usd=5600.0,
)

MOTORS: list[Motor] = [CURTIS_AC50, HYPER9_120V, HYPER9_144V]
