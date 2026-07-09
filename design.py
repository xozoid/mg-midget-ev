"""MG Midget EV - design study and pack comparison."""

from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from mg.battery import BatteryPack, CELLS
from mg.motor import MOTORS
from mg.utils import wheel_power_kw
from mg.vehicle import OperatingLimitError, Vehicle

if TYPE_CHECKING:
    from mg.motor import Motor

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
MAX_WEIGHT_LB: float = 2200.0
MIN_RANGE_MILES: float = 60.0
MIN_MAX_SPEED_MPH: float = 60.0
PEAK_HILL_GRADE_PCT: float = 20.0
PEAK_HILL_CHECK_SPEEDS_MPH: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
CONTINUOUS_HILL_GRADE_PCT: float = 6.0
CONTINUOUS_HILL_CHECK_SPEED_MPH: float = 45.0
CRUISE_SPEED_MPH: float = 35.0
TRANSMISSION_MASS_KG: float = 30.0
DESIGN_SCORE_TOP_SPEED_MPH: float = 70.0
DESIGN_SCORE_BATTERY_C_RATE: float = 1.0
DESIGN_SCORE_COST_SCALE_USD: float = 1000.0

# ---------------------------------------------------------------------------
# Combinations
# Each motor entry maps to the series cell count that targets the motor
# nominal voltage with LFP cells at 3.2 V nominal.
# ---------------------------------------------------------------------------
PARALLEL_STRINGS: list[int] = [1, 2]


@dataclass(frozen=True)
class Drivetrain:
    """Drivetrain option for the design sweep."""

    name: str
    glider_mass_delta_kg: float
    transmission_efficiency: float
    gear_ratios: dict[Literal[1, 2, 3, 4], float]


@dataclass(frozen=True)
class DesignRow:
    """One evaluated design option."""

    motor_short: str
    drivetrain: str
    config: str
    weight_lb: float
    range_miles: float
    max_speed_mph: float
    continuous_hill_discharge_c: float
    total_cost_usd: float
    cell_cost_usd: float
    design_margin_score: float
    passed: bool
    verdict: str


DRIVETRAINS: list[Drivetrain] = [
    Drivetrain(
        name="4-speed",
        glider_mass_delta_kg=0.0,
        transmission_efficiency=0.95,
        gear_ratios={
            1: 3.513,
            2: 2.170,
            3: 1.378,
            4: 1.00,
        },
    ),
    Drivetrain(
        name="direct",
        glider_mass_delta_kg=-TRANSMISSION_MASS_KG,
        transmission_efficiency=0.99,
        gear_ratios={1: 1.0},
    ),
]


def _series_for_motor(motor: Motor, cell_v: float) -> int:
    """Return the series cell count that targets the motor's nominal voltage.

    Args:
        motor: Motor whose nominal voltage is used.
        cell_v: Nominal cell voltage

    Returns:
        Number of LFP cells (3.2 V each) to wire in series.

    """
    return round(motor.nominal_voltage_v / cell_v)


def _flat_max_speed_mph(vehicle: Vehicle) -> float:
    """Estimate flat-road maximum speed in miles per hour.

    Returns:
        Maximum flat-road speed before gearing, continuous tractive force, or
        current limits are exceeded.

    """
    redline_speed_mph = max(
        vehicle.speed_mph_at_motor_rpm(vehicle.motor.max_rpm, gear)
        for gear in vehicle.gear_ratios
    )

    def can_hold_speed(speed_mph: float) -> bool:
        if (
            _best_continuous_force_at_speed_n(
                vehicle,
                speed_mph,
            )
            < vehicle.road_load_forces(speed_mph, 0.0)["total_n"]
        ):
            return False
        try:
            vehicle.cruise_case(speed_mph, 0.0)
        except OperatingLimitError:
            return False
        return True

    low_mph = 0.0
    high_mph = redline_speed_mph
    for _ in range(40):
        mid_mph = (low_mph + high_mph) / 2.0
        if can_hold_speed(mid_mph):
            low_mph = mid_mph
        else:
            high_mph = mid_mph
    return low_mph


def _best_continuous_force_at_speed_n(vehicle: Vehicle, speed_mph: float) -> float:
    """Return the best continuous tractive force available at road speed.

    Args:
        vehicle: Vehicle configuration to evaluate.
        speed_mph: Road speed in miles per hour.

    Returns:
        Best available continuous wheel force in Newtons, capped by traction.

    """
    traction_limit_n = vehicle.traction_limited_force_n()
    best_force_n = 0.0

    for gear in vehicle.gear_ratios:
        rpm = vehicle.motor_rpm(speed_mph, gear)
        if rpm > vehicle.motor.max_rpm:
            continue
        force_n = min(
            vehicle.continuous_drive_force_n(gear, rpm),
            traction_limit_n,
        )
        best_force_n = max(best_force_n, force_n)

    return best_force_n


def _best_peak_force_at_speed_n(vehicle: Vehicle, speed_mph: float) -> float:
    """Return the best peak tractive force available at road speed.

    Args:
        vehicle: Vehicle configuration to evaluate.
        speed_mph: Road speed in miles per hour.

    Returns:
        Best available peak wheel force in Newtons, capped by traction.

    """
    traction_limit_n = vehicle.traction_limited_force_n()
    best_force_n = 0.0

    for gear in vehicle.gear_ratios:
        rpm = vehicle.motor_rpm(speed_mph, gear)
        if rpm > vehicle.motor.max_rpm:
            continue
        force_n = min(vehicle.drive_force_n(gear, rpm), traction_limit_n)
        best_force_n = max(best_force_n, force_n)

    return best_force_n


def _hill_force_error(
    vehicle: Vehicle,
    speed_mph: float,
    grade_pct: float,
    *,
    continuous: bool,
) -> str:
    """Return the hill force-limit error, if any.

    Args:
        vehicle: Vehicle configuration to evaluate.
        speed_mph: Road speed in miles per hour.
        grade_pct: Road grade as a percentage.
        continuous: Use continuous motor torque when ``True``. Use peak motor
            torque when ``False``.

    Returns:
        Empty string when wheel force is sufficient; otherwise, a short error
        suitable for the design table.

    """
    required_force_n = vehicle.road_load_forces(speed_mph, grade_pct)["total_n"]
    if continuous:
        available_force_n = _best_continuous_force_at_speed_n(vehicle, speed_mph)
    else:
        available_force_n = _best_peak_force_at_speed_n(vehicle, speed_mph)
    if available_force_n >= required_force_n:
        return ""
    return f"needs {required_force_n:.0f}N, has {available_force_n:.0f}N"


def _cruise_discharge_c_unchecked(
    vehicle: Vehicle,
    speed_mph: float,
    grade_pct: float,
) -> float:
    """Return cruise discharge C-rate without enforcing component limits.

    Args:
        vehicle: Vehicle configuration to evaluate.
        speed_mph: Road speed in miles per hour.
        grade_pct: Road grade as a percentage.

    Returns:
        Pack discharge C-rate for the operating point.

    """
    forces = vehicle.road_load_forces(speed_mph, grade_pct)
    p_wheel_kw = wheel_power_kw(forces["total_n"], speed_mph)
    total_efficiency = (
        vehicle.driveline_efficiency * vehicle.motor.assumed_cruise_efficiency
    )
    p_batt_kw = (p_wheel_kw / max(total_efficiency, 1e-6)) + (
        vehicle.accessory_power_kw
    )
    pack_current_a = (p_batt_kw * 1000.0) / vehicle.battery.nominal_voltage_v
    return max(pack_current_a, 0.0) / vehicle.battery.capacity_ah


def _design_margin_score(
    *,
    weight_lb: float,
    max_speed_mph: float,
    continuous_hill_discharge_c: float,
    total_cost_usd: float,
) -> float:
    """Return the non-gating design margin score.

    Args:
        weight_lb: Vehicle weight in pounds.
        max_speed_mph: Estimated flat-road maximum speed in miles per hour.
        continuous_hill_discharge_c: Discharge C-rate at the continuous hill
            check operating point.
        total_cost_usd: Estimated parts cost in US dollars.

    Returns:
        Dimensionless score. Higher is better.

    """
    weight_margin_pct = ((MAX_WEIGHT_LB - weight_lb) / MAX_WEIGHT_LB) * 100.0
    speed_margin_pct = (
        (max_speed_mph - DESIGN_SCORE_TOP_SPEED_MPH) / DESIGN_SCORE_TOP_SPEED_MPH
    ) * 100.0
    battery_margin_pct = (
        (DESIGN_SCORE_BATTERY_C_RATE - continuous_hill_discharge_c)
        / DESIGN_SCORE_BATTERY_C_RATE
    ) * 100.0
    cost_penalty = total_cost_usd / DESIGN_SCORE_COST_SCALE_USD
    return weight_margin_pct + speed_margin_pct + battery_margin_pct - cost_penalty


def _print_table(rows: list[DesignRow], title: str) -> None:
    """Print a formatted comparison table.

    Args:
        rows: Design rows to print.
        title: Section heading printed above the table.

    """
    col = [14, 10, 18, 9, 9, 9, 9, 9, 10, 8, 0]
    headers = [
        "Motor",
        "Drive",
        "Pack",
        "Wt (lb)",
        "Range mi",
        "Max mph",
        "6% BatC",
        "Cost $",
        "Cells $",
        "Score",
        "Result",
    ]
    header_line = "  ".join(
        h.ljust(w) if w else h for h, w in zip(headers, col, strict=True)
    )
    sep = "-" * max(len(header_line), 60)
    print(f"\n{title}")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print(
            f"  {row.motor_short:<{col[0]}}  "
            f"{row.drivetrain:<{col[1]}}  "
            f"{row.config:<{col[2]}}  "
            f"{row.weight_lb:>{col[3]}.0f}  "
            f"{row.range_miles:>{col[4]}.1f}  "
            f"{row.max_speed_mph:>{col[5]}.1f}  "
            f"{row.continuous_hill_discharge_c:>{col[6]}.1f}  "
            f"{row.total_cost_usd:>{col[7]},.0f}  "
            f"{row.cell_cost_usd:>{col[8]},.0f}  "
            f"{row.design_margin_score:>{col[9]}.1f}  "
            f"{row.verdict}",
        )
    print(sep)


if __name__ == "__main__":
    all_rows: list[DesignRow] = []

    for motor, cell, parallel, drivetrain in itertools.product(
        MOTORS,
        CELLS,
        PARALLEL_STRINGS,
        DRIVETRAINS,
    ):
        series = _series_for_motor(motor, cell.nominal_voltage_v)
        pack = BatteryPack(
            cell=cell,
            series_cells=series,
            parallel_strings=parallel,
        )
        vehicle = Vehicle(
            battery=pack,
            motor=motor,
            glider_mass_kg=Vehicle.glider_mass_kg + drivetrain.glider_mass_delta_kg,
            transmission_efficiency=drivetrain.transmission_efficiency,
            gear_ratios=drivetrain.gear_ratios,
        )
        cost = vehicle.build_cost_model()

        weight_lb = vehicle.total_mass_kg * 2.20462
        weight_ok = weight_lb <= MAX_WEIGHT_LB
        max_speed_mph = _flat_max_speed_mph(vehicle)
        speed_ok = max_speed_mph >= MIN_MAX_SPEED_MPH

        continuous_hill_discharge_c = _cruise_discharge_c_unchecked(
            vehicle,
            CONTINUOUS_HILL_CHECK_SPEED_MPH,
            CONTINUOUS_HILL_GRADE_PCT,
        )
        peak_hill_error = ""
        continuous_hill_error = ""
        for hill_speed_mph in PEAK_HILL_CHECK_SPEEDS_MPH:
            peak_hill_error = _hill_force_error(
                vehicle,
                hill_speed_mph,
                PEAK_HILL_GRADE_PCT,
                continuous=False,
            )
            if peak_hill_error:
                peak_hill_error = f"{hill_speed_mph:.0f}mph {peak_hill_error}"
                break
        try:
            cruise = vehicle.cruise_case(
                CONTINUOUS_HILL_CHECK_SPEED_MPH,
                CONTINUOUS_HILL_GRADE_PCT,
            )
            continuous_hill_discharge_c = cruise.discharge_c
        except OperatingLimitError as exc:
            continuous_hill_error = str(exc)
        if not continuous_hill_error:
            continuous_hill_error = _hill_force_error(
                vehicle,
                CONTINUOUS_HILL_CHECK_SPEED_MPH,
                CONTINUOUS_HILL_GRADE_PCT,
                continuous=True,
            )

        range_miles = 0.0
        flat_error = ""
        try:
            cruise = vehicle.cruise_case(CRUISE_SPEED_MPH, 0.0)
            range_miles = cruise.range_miles
        except OperatingLimitError as exc:
            flat_error = str(exc)
        range_ok = range_miles >= MIN_RANGE_MILES
        passed = (
            weight_ok
            and speed_ok
            and not peak_hill_error
            and not continuous_hill_error
            and range_ok
            and not flat_error
        )

        config = f"{cell.model} {series}s{parallel}p"
        motor_short = motor.name.split("/")[0].strip().split()[-1]
        total_cost_usd = cost.subtotal_parts_usd()
        design_margin_score = _design_margin_score(
            weight_lb=weight_lb,
            max_speed_mph=max_speed_mph,
            continuous_hill_discharge_c=continuous_hill_discharge_c,
            total_cost_usd=total_cost_usd,
        )

        fails = []
        if not weight_ok:
            fails.append(f"wt {weight_lb:.0f}lb")
        if not speed_ok:
            fails.append(f"top {max_speed_mph:.0f}mph")
        if peak_hill_error:
            fails.append(f"20% peak: {peak_hill_error}")
        if continuous_hill_error:
            fails.append(f"6% cont: {continuous_hill_error}")
        if not range_ok and not flat_error:
            fails.append(f"range {range_miles:.0f}mi")
        if flat_error:
            fails.append(f"limit: {flat_error}")
        verdict = "PASS" if passed else ("FAIL (" + ", ".join(fails) + ")")

        all_rows.append(
            DesignRow(
                motor_short=motor_short,
                drivetrain=drivetrain.name,
                config=config,
                weight_lb=weight_lb,
                range_miles=range_miles,
                max_speed_mph=max_speed_mph,
                continuous_hill_discharge_c=continuous_hill_discharge_c,
                total_cost_usd=total_cost_usd,
                cell_cost_usd=pack.cell_cost_usd,
                design_margin_score=design_margin_score,
                passed=passed,
                verdict=verdict,
            ),
        )

    _print_table(all_rows, "=== ALL COMBINATIONS ===")

    passing = [r for r in all_rows if r.passed]
    passing.sort(key=operator.attrgetter("design_margin_score"), reverse=True)
    if passing:
        _print_table(passing, "=== PASSING - RANKED BY DESIGN MARGIN SCORE ===")
    else:
        print("\nNo combinations passed all criteria.")

    print(
        f"\nCriteria: weight <= {MAX_WEIGHT_LB:.0f} lb | "
        f"top speed >= {MIN_MAX_SPEED_MPH:.0f} mph | "
        f"range >= {MIN_RANGE_MILES:.0f} mi @ {CRUISE_SPEED_MPH:.0f} mph | "
        f"peak grade >= {PEAK_HILL_GRADE_PCT:.0f}% @ "
        f"{PEAK_HILL_CHECK_SPEEDS_MPH[0]:.0f}-"
        f"{PEAK_HILL_CHECK_SPEEDS_MPH[-1]:.0f} mph | "
        f"continuous grade >= {CONTINUOUS_HILL_GRADE_PCT:.0f}% @ "
        f"{CONTINUOUS_HILL_CHECK_SPEED_MPH:.0f} mph",
    )
    print(
        "Score: weight margin % + "
        f"top-speed margin % above {DESIGN_SCORE_TOP_SPEED_MPH:.0f} mph + "
        f"C-rate margin % below {DESIGN_SCORE_BATTERY_C_RATE:.1f} C - "
        f"cost/${DESIGN_SCORE_COST_SCALE_USD:,.0f}",
    )
