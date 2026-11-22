"""Evaluate front-suspension assembly tolerance with adjustable static toe."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, fields, replace
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

from front_suspension.candidates import CANDIDATES
from front_suspension.helpers.constraints import STATIC_TOE
from front_suspension.helpers.geometry import Solver
from front_suspension.helpers.sweeper import (
    SuspensionSweep,
    SuspensionTargetTests,
    SuspensionTargetValues,
)
from front_suspension.helpers.utils import Angle

if TYPE_CHECKING:
    from front_suspension.helpers.geometry import GeometryFixed

# The upper pickup's Y coordinate is derived from caster and cannot vary
# independently in the current geometry model.
ADJUSTABLE_DIMENSIONS = ("up_x", "up_z", "arm_upper", "arm_lower")
ALIGNMENT_TOLERANCE = 1e-7


class Trial(NamedTuple):
    """One assembly scenario."""

    name: str
    geometry: GeometryFixed
    offsets_in: tuple[float, ...]


def evaluate(trial: Trial) -> dict[str, object]:
    """Align and evaluate one assembly scenario.

    Returns:
        A CSV row containing offsets, alignment, target values and outcomes.

    """
    row: dict[str, object] = {
        "scenario": trial.name,
        **{
            f"{name}_offset_in": offset
            for name, offset in zip(
                ADJUSTABLE_DIMENSIONS,
                trial.offsets_in,
                strict=True,
            )
        },
        "status": "alignment_unsolvable",
        "pass": False,
    }
    geometry = replace(
        trial.geometry,
        **dict(
            zip(
                ADJUSTABLE_DIMENSIONS,
                (
                    getattr(trial.geometry, name) + offset
                    for name, offset in zip(
                        ADJUSTABLE_DIMENSIONS,
                        trial.offsets_in,
                        strict=True,
                    )
                ),
                strict=True,
            ),
        ),
    )
    target_toe_rad = (float(STATIC_TOE[0]) + float(STATIC_TOE[1])) / 2

    def residual(angles: np.ndarray) -> list[float]:
        """Calculate alignment errors.

        Returns:
            Wheel height error in inches and toe error in radians.

        """
        state = Solver(
            geometry,
            _arm_lower_elevation=Angle(angles[0]),
            _swivel=Angle(angles[1]),
        )
        return [
            state.wheel_contact_z,
            float(state.suspension_state.toe) - target_toe_rad,
        ]

    try:  # ruff: ignore[too-many-statements-in-try-clause]
        limits_rad = np.deg2rad([20, 30])
        result = least_squares(residual, [0.0, 0.0], bounds=(-limits_rad, limits_rad))
        if not result.success or np.max(np.abs(result.fun)) > ALIGNMENT_TOLERANCE:
            return row
        aligned = Solver(
            geometry,
            _arm_lower_elevation=Angle(result.x[0]),
            _swivel=Angle(result.x[1]),
        )
        length_in = aligned.arm_tie_rod_derived
        row["tie_rod_length_in"] = length_in
        row["tie_rod_adjustment_in"] = length_in - trial.geometry.arm_tie_rod
        geometry = replace(geometry, arm_tie_rod=length_in)
        rest = Solver.at(geometry, 0)
        row["static_toe_deg"] = rest.toe.deg
        if not STATIC_TOE[0].deg <= rest.toe.deg <= STATIC_TOE[1].deg:
            row["status"] = "toe_out_of_range"
            return row
        sweep = SuspensionSweep(geometry)
        _ = sweep.steering
        values = sweep.target_values
        tests = sweep.target_tests
    except ValueError:
        row["status"] = "unsolvable"
        return row
    row.update(asdict(values))
    row["status"] = "evaluated"
    row["pass"] = tests.hard_constraints_passed
    for name, passed in asdict(tests).items():
        row[f"{name}_pass"] = passed
    return row


def failed_target(row: dict[str, object], name: str) -> bool:
    """Check a recorded target result, excluding targets that were not evaluated.

    Returns:
        Whether the target was evaluated and failed.

    """
    result = row.get(f"{name}_pass")
    return result is not None and not bool(result)


def report_individual_failures(rows: list[dict[str, object]]) -> None:
    """Print failed one-dimension scenarios and their hard targets."""
    print("One-dimension scenarios that failed:")
    for row in rows:
        if not row["pass"]:
            failed = [
                name
                for name in (
                    "static_camber",
                    "static_toe",
                    "bump_camber",
                    "droop_camber",
                    "bump_steer",
                    "wheel_center",
                )
                if failed_target(row, name)
            ]
            print(f"  {row['scenario']}: {row['status']} {', '.join(failed)}")


def main() -> None:
    """Run a reproducible sensitivity study for a numbered candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=int, help="1-based candidate ID")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0625,
        metavar="IN",
        help="independent uniform dimension offset, +/- inches (default: 0.0625)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        metavar="N",
        help="number of combined random scenarios (default: 100)",
    )
    parser.add_argument("--output", type=Path, help="CSV output path")
    args = parser.parse_args()
    if args.samples < 0 or args.tolerance < 0:
        parser.error("samples and tolerance amust be nonnegative ")

    if not 1 <= args.candidate <= len(CANDIDATES):
        parser.error(f"candidate must be between 1 and {len(CANDIDATES)}")
    geometry: GeometryFixed = CANDIDATES[args.candidate - 1]
    zero = (0.0,) * len(ADJUSTABLE_DIMENSIONS)
    trials = [Trial("baseline", geometry, zero)]
    for index, name in enumerate(ADJUSTABLE_DIMENSIONS):
        for sign, label in ((-1, "minus"), (1, "plus")):
            offsets = list(zero)
            offsets[index] = sign * args.tolerance
            trials.append(Trial(f"{name}_{label}", geometry, tuple(offsets)))
    rng = np.random.default_rng()
    for index in range(args.samples):
        offsets = tuple(
            float(v) for v in rng.uniform(-args.tolerance, args.tolerance, len(zero))
        )
        trials.append(Trial(f"random_{index + 1:04d}", geometry, offsets))

    output = args.output or Path(
        f"out/front_suspension/sensitivity_{args.candidate:02d}.csv",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "scenario",
        *(f"{name}_offset_in" for name in ADJUSTABLE_DIMENSIONS),
        "status",
        "tie_rod_length_in",
        "tie_rod_adjustment_in",
        *(field.name for field in fields(SuspensionTargetValues)),
        *(f"{field.name}_pass" for field in fields(SuspensionTargetTests)),
        "pass",
    ]
    with (
        output.open("w", encoding="utf-8", newline="") as handle,
        Pool(processes=max(1, (os.cpu_count() or 0) - 2)) as pool,
    ):
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        rows: list[dict[str, object]] = []
        for row in tqdm(pool.imap(evaluate, trials, chunksize=1), total=len(trials)):
            writer.writerow(row)
            rows.append(row)
    random_rows = [row for row in rows if str(row["scenario"]).startswith("random_")]
    print(
        f"Candidate {args.candidate}: +/- {args.tolerance:g} in "
        "independent dimension offsets ",
    )
    print(f"Baseline: {rows[0]['status']}, {'PASS' if rows[0]['pass'] else 'FAIL'}")
    random_passed = sum(bool(row["pass"]) for row in random_rows)
    print(f"Random scenarios passing hard targets: {random_passed}/{len(random_rows)}")
    for status in ("alignment_unsolvable", "toe_out_of_range", "unsolvable"):
        print(f"{status}: {sum(row['status'] == status for row in random_rows)}")
    for name in (
        "static_camber",
        "static_toe",
        "bump_camber",
        "droop_camber",
        "bump_steer",
        "wheel_center",
    ):
        failures = sum(failed_target(row, name) for row in random_rows)
        print(f"{name} failures: {failures}")
    for name in ("camber_slope", "bump_camber_monotonic", "roll_center"):
        failures = sum(failed_target(row, name) for row in random_rows)
        print(f"{name} soft-target failures: {failures}")
    report_individual_failures(rows[1 : 1 + 2 * len(ADJUSTABLE_DIMENSIONS)])
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
