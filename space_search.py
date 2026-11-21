"""Search suspension dimensions and write target measurements to CSV."""

from __future__ import annotations

import csv
import os
from dataclasses import fields, replace
from functools import cached_property
from itertools import product
from math import prod
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple, NotRequired, TypedDict

import numpy as np
from tqdm import tqdm

from mg.front_suspension.constraints import BUMP, STEERING
from mg.front_suspension.geometry import GeometryFixed, Solver, SuspensionState
from mg.front_suspension.sweeper import SuspensionSweep, SuspensionTargetValues

SearchRow = TypedDict(
    "SearchRow",
    {
        "arm_tie_rod": float,
        "arm_lower": float,
        "arm_upper": float,
        "up_x": float,
        "up_z": float,
        "solvable": bool,
        "pass": bool,
        "static_camber_deg": NotRequired[float],
        "static_toe_deg": NotRequired[float],
        "camber_slope_deg_per_in": NotRequired[float],
        "bump_camber_increase_deg": NotRequired[float],
        "bump_camber_max_deg": NotRequired[float],
        "droop_camber_max_deg": NotRequired[float],
        "bump_steer_pp_deg": NotRequired[float],
        "wheel_center_x_in": NotRequired[float],
        "roll_center_pp_in": NotRequired[float],
    },
)


class SearchStates(NamedTuple):
    """Sampled suspension states for the search."""

    bump: list[SuspensionState]
    steering: list[SuspensionState]


class SearchSweep(SuspensionSweep):
    """Sweep outward from rest using the previous solution as an initial guess."""

    @cached_property
    def states(self) -> SearchStates:
        """Solve both ranges, sharing the centered rest state.

        Returns:
            States in the original sample order.

        """
        centered = replace(self.geometry, steering=0)
        # Test the steering limits first so impossible designs fail early.
        extremes = {
            value: Solver.solve(replace(centered, steering=value), 0)
            for value in dict.fromkeys(
                (min(self.steering_range), max(self.steering_range)),
            )
            if value != 0
        }
        rest = Solver.solve(centered, 0)

        def sample(values: list[float], *, steering: bool) -> list[SuspensionState]:
            """Solve each side from rest, then restore the requested order.

            Returns:
                States in the original sample order.

            """
            solved = {0.0: rest, **(extremes if steering else {})}
            for direction in (-1, 1):
                previous = rest
                for value in sorted((v for v in values if v * direction > 0), key=abs):
                    if value in solved:
                        previous = solved[value]
                        continue
                    fixed = replace(centered, steering=value) if steering else centered
                    previous = Solver.solve(
                        fixed,
                        0 if steering else value,
                        initial=previous,
                    )
                    solved[value] = previous
            return [solved[value].suspension_state for value in values]

        steering = sample(self.steering_range, steering=True)
        bump = sample(self.bump_range, steering=False)
        return SearchStates(bump=bump, steering=steering)

    @cached_property
    def bump(self) -> list[SuspensionState]:
        """States over wheel travel with the rack centered."""
        return self.states.bump

    @cached_property
    def steering(self) -> list[SuspensionState]:
        """States over rack travel at rest height."""
        return self.states.steering


def evaluate(combination: tuple[float, ...]) -> SearchRow:
    """Evaluate one combination of suspension dimensions.

    Returns:
        Input dimensions, solvability, target measurements, and pass status.

    """
    arm_tie_rod, arm_lower, arm_upper, up_x, up_z = combination
    fixed = GeometryFixed(
        arm_tie_rod=arm_tie_rod,
        arm_lower=arm_lower,
        arm_upper=arm_upper,
        up_x=up_x,
        up_z=up_z,
    )
    row: SearchRow = {
        "arm_tie_rod": arm_tie_rod,
        "arm_lower": arm_lower,
        "arm_upper": arm_upper,
        "up_x": up_x,
        "up_z": up_z,
        "solvable": False,
        "pass": False,
    }
    try:
        sweep = SearchSweep(
            fixed,
            bump_range=[
                BUMP[0],
                BUMP[0] * 0.01,
                0,
                BUMP[1] * 0.01,
                BUMP[1],
            ],
            steering_range=[
                STEERING[0],
                STEERING[0] * 0.01,
                0,
                STEERING[1] * 0.01,
                STEERING[1],
            ],
        )
        _ = sweep.steering
        values = sweep.target_values
    except ValueError:
        # Unsolvable geometries still need a row in the search results.
        pass
    else:
        row["static_camber_deg"] = values.static_camber_deg
        row["static_toe_deg"] = values.static_toe_deg
        row["camber_slope_deg_per_in"] = values.camber_slope_deg_per_in
        row["bump_camber_increase_deg"] = values.bump_camber_increase_deg
        row["bump_camber_max_deg"] = values.bump_camber_max_deg
        row["droop_camber_max_deg"] = values.droop_camber_max_deg
        row["bump_steer_pp_deg"] = values.bump_steer_pp_deg
        row["wheel_center_x_in"] = values.wheel_center_x_in
        row["roll_center_pp_in"] = values.roll_center_pp_in
        row["solvable"] = True
        row["pass"] = sweep.target_tests.hard_constraints_passed
    return row


def main() -> None:
    """Evaluate every combination of suspension dimensions."""
    n = 15
    spaces = {
        "arm_tie_rod": np.linspace(7.5, 8.5, n),
        "arm_lower": np.linspace(9, 12, n),
        "arm_upper": np.linspace(5, 9, n),
        "up_x": np.linspace(11, 15, n),
        "up_z": np.linspace(14, 17, n),
    }
    target_names = [field.name for field in fields(SuspensionTargetValues)]
    columns = [*spaces, "solvable", *target_names, "pass"]
    combinations = product(*spaces.values())

    n_solvable = 0
    n_passing = 0

    with (
        Path("space_search.csv").open("w", encoding="utf-8", newline="") as output,
        Pool(processes=max(1, (os.cpu_count() or 0) - 2)) as pool,
    ):
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        results = pool.imap(evaluate, combinations, chunksize=16)
        for row in tqdm(results, total=prod(map(len, spaces.values()))):
            if row["solvable"]:
                n_solvable += 1
            if row["pass"]:
                n_passing += 1
            writer.writerow(row)

    print(f"Solvable: {n_solvable:,}")
    print(f"Passing: {n_passing:,}")


if __name__ == "__main__":
    main()
