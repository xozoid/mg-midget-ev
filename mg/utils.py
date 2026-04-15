"""Physics constants and unit-conversion utilities."""

from __future__ import annotations

G = 9.81
AIR_DENSITY = 1.225
LBFT_TO_NM = 1.35582
INCH_TO_M = 0.0254
MPH_TO_MPS = 0.44704
MPS_TO_MPH = 1.0 / MPH_TO_MPS
HP_PER_KW = 1.34102


def mph_to_mps(mph: float) -> float:
    """Convert miles per hour to metres per second.

    Args:
        mph: Speed in miles per hour.

    Returns:
        Speed in metres per second.

    """
    return mph * MPH_TO_MPS


def mps_to_mph(mps: float) -> float:
    """Convert metres per second to miles per hour.

    Args:
        mps: Speed in metres per second.

    Returns:
        Speed in miles per hour.

    """
    return mps * MPS_TO_MPH


def grade_force_n(total_mass_kg: float, grade_percent: float) -> float:
    """Gravitational force component along a graded road.

    Args:
        total_mass_kg: Total vehicle mass in kilograms.
        grade_percent: Road grade as a percentage (positive = uphill).

    Returns:
        Grade resistance force in Newtons.

    """
    return total_mass_kg * G * (grade_percent / 100.0)


def wheel_power_kw(force_n: float, speed_mph: float) -> float:
    """Mechanical power at the wheels.

    Args:
        force_n: Tractive or resistive force in Newtons.
        speed_mph: Vehicle speed in miles per hour.

    Returns:
        Power in kilowatts.

    """
    return force_n * mph_to_mps(speed_mph) / 1000.0
