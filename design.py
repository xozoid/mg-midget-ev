"""MG Midget EV - design study and pack comparison."""

from __future__ import annotations

import itertools
import operator
from typing import TYPE_CHECKING

from mg.battery import BatteryPack, CELLS
from mg.motor import MOTORS
from mg.vehicle import OperatingLimitError, Vehicle

if TYPE_CHECKING:
    from mg.motor import Motor

# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
MAX_WEIGHT_LB: float = 2200.0
MIN_RANGE_MILES: float = 60.0
MIN_HILL_GRADE_PCT: float = 25.0
HILL_CHECK_SPEED_MPH: float = 5.0
CRUISE_SPEED_MPH: float = 35.0

# ---------------------------------------------------------------------------
# Combinations
# Each motor entry maps to the series cell count that targets the motor
# nominal voltage with LFP cells at 3.2 V nominal.
# ---------------------------------------------------------------------------
PARALLEL_STRINGS: list[int] = [1, 2]


def _series_for_motor(motor: Motor, cell_v: float) -> int:
    """Return the series cell count that targets the motor's nominal voltage.

    Args:
        motor: Motor whose nominal voltage is used.
        cell_v: Nominal cell voltage

    Returns:
        Number of LFP cells (3.2 V each) to wire in series.

    """
    return round(motor.nominal_voltage_v / cell_v)


def _max_grade_at_speed(vehicle: Vehicle, speed_mph: float) -> float:
    """Return the highest grade achievable at the given speed across all gears.

    Args:
        vehicle: Vehicle to evaluate.
        speed_mph: Road speed in miles per hour.

    Returns:
        Maximum grade in percent that the vehicle can sustain.

    """
    best = 0.0
    for gear in (1, 2, 3, 4):
        result = vehicle.gradeability_at_speed(speed_mph, gear)
        if result.valid:
            best = max(best, result.max_grade_percent)
    return best


def _print_table(
    rows: list[tuple[str, str, float, float, float, float, float, bool, str]],
    title: str,
) -> None:
    """Print a formatted comparison table.

    Args:
        rows: Each row is (motor_short, config, weight_lb, range_mi,
            max_grade, total_cost, cell_cost, passed, verdict).
        title: Section heading printed above the table.

    """
    col = [14, 18, 9, 9, 9, 9, 10, 0]
    headers = [
        "Motor",
        "Pack",
        "Wt (lb)",
        "Range mi",
        "Grade %",
        "Cost $",
        "Cells $",
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
    for (
        motor_short,
        config,
        weight_lb,
        range_mi,
        grade,
        total_cost,
        cell_cost,
        _,
        verdict,
    ) in rows:
        print(
            f"  {motor_short:<{col[0]}}  "
            f"{config:<{col[1]}}  "
            f"{weight_lb:>{col[2]}.0f}  "
            f"{range_mi:>{col[3]}.1f}  "
            f"{grade:>{col[4]}.1f}  "
            f"{total_cost:>{col[5]},.0f}  "
            f"{cell_cost:>{col[6]},.0f}  "
            f"{verdict}",
        )
    print(sep)


if __name__ == "__main__":
    all_rows: list[tuple[str, str, float, float, float, float, float, bool, str]] = []

    for motor, cell, parallel in itertools.product(MOTORS, CELLS, PARALLEL_STRINGS):
        series = _series_for_motor(motor, cell.nominal_voltage_v)
        pack = BatteryPack(
            cell=cell,
            series_cells=series,
            parallel_strings=parallel,
        )
        vehicle = Vehicle(battery=pack, motor=motor)
        cost = vehicle.build_cost_model()

        weight_lb = vehicle.total_mass_kg * 2.20462
        weight_ok = weight_lb <= MAX_WEIGHT_LB

        max_grade = _max_grade_at_speed(vehicle, HILL_CHECK_SPEED_MPH)
        hill_ok = max_grade >= MIN_HILL_GRADE_PCT

        range_miles = 0.0
        error_msg = ""
        try:
            cruise = vehicle.cruise_case(CRUISE_SPEED_MPH, 0.0)
            range_miles = cruise.range_miles
        except OperatingLimitError as exc:
            error_msg = str(exc)

        range_ok = range_miles >= MIN_RANGE_MILES
        passed = weight_ok and hill_ok and range_ok and not error_msg

        config = f"{cell.model} {series}s{parallel}p"
        motor_short = motor.name.split("/")[0].strip().split()[-1]

        fails = []
        if not weight_ok:
            fails.append(f"wt {weight_lb:.0f}lb")
        if not hill_ok:
            fails.append(f"hill {max_grade:.0f}%")
        if not range_ok and not error_msg:
            fails.append(f"range {range_miles:.0f}mi")
        if error_msg:
            fails.append(f"limit: {error_msg}")
        verdict = "PASS" if passed else ("FAIL (" + ", ".join(fails) + ")")

        all_rows.append(
            (
                motor_short,
                config,
                weight_lb,
                range_miles,
                max_grade,
                cost.subtotal_parts_usd(),
                pack.cell_cost_usd,
                passed,
                verdict,
            ),
        )

    _print_table(all_rows, "=== ALL COMBINATIONS ===")

    passing = [r for r in all_rows if r[7]]
    passing.sort(key=operator.itemgetter(5))
    if passing:
        _print_table(passing, "=== PASSING - RANKED BY COST (LOWEST FIRST) ===")
    else:
        print("\nNo combinations passed all criteria.")

    print(
        f"\nCriteria: weight <= {MAX_WEIGHT_LB:.0f} lb | "
        f"range >= {MIN_RANGE_MILES:.0f} mi @ {CRUISE_SPEED_MPH:.0f} mph | "
        f"grade >= {MIN_HILL_GRADE_PCT:.0f}% @ {HILL_CHECK_SPEED_MPH:.0f} mph",
    )
