"""Sweep front-suspension travel and steering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style

from mg.front_suspension.constraints import (
    BUMP,
    BUMP_CAMBER_MAX,
    BUMP_STEER_PP,
    DROOP_CAMBER_MAX,
    ROLL_CENTER_PP,
    STATIC_CAMBER,
    STATIC_CAMBER_SLOPE,
    STEERING,
    WHEEL_CENTER_X,
)
from mg.front_suspension.geometry import GeometryFixed, Solver

if TYPE_CHECKING:
    from pathlib import Path

    from mg.front_suspension.geometry import SuspensionState


N = 31


@dataclass(frozen=True)
class SuspensionTargetValues:
    """Measured values used to evaluate the suspension targets."""

    static_camber_deg: float
    camber_slope_deg_per_in: float
    bump_camber_increase_deg: float
    bump_camber_max_deg: float
    droop_camber_max_deg: float
    bump_steer_pp_deg: float
    wheel_center_x_in: float
    roll_center_pp_in: float


@dataclass(frozen=True)
class SuspensionTargetTests:
    """Pass or fail results for the suspension targets."""

    static_camber: bool
    camber_slope: bool
    bump_camber_monotonic: bool
    bump_camber: bool
    droop_camber: bool
    bump_steer: bool
    wheel_center: bool
    roll_center: bool

    @property
    def hard_constraints_passed(self) -> bool:
        """Whether all hard design constraints pass."""
        return all(
            (
                self.static_camber,
                self.bump_camber,
                self.droop_camber,
                self.bump_steer,
                self.wheel_center,
                self.roll_center,
            ),
        )


@dataclass(frozen=True)
class SuspensionSweep:
    """Evaluate suspension geometry over travel and steering ranges."""

    geometry: GeometryFixed

    bump_range: list[float] = field(
        default_factory=lambda: list(np.linspace(*BUMP, N, dtype=float)),
    )
    steering_range: list[float] = field(
        default_factory=lambda: list(np.linspace(*STEERING, N, dtype=float)),
    )

    @cached_property
    def bump(self) -> list[SuspensionState]:
        """States over wheel travel with the rack centered."""
        g = replace(self.geometry, steering=0)
        return [Solver.at(g, v) for v in self.bump_range]

    @cached_property
    def steering(self) -> list[SuspensionState]:
        """States over rack travel at rest height."""
        return [
            Solver.at(replace(self.geometry, steering=v), 0)
            for v in self.steering_range
        ]

    @cached_property
    def camber(self) -> list[float]:
        """Camber angles in degrees over wheel travel."""
        return [s.camber.deg for s in self.bump]

    @cached_property
    def bump_steer(self) -> list[float]:
        """Toe angles in degrees over wheel travel."""
        return [s.toe.deg for s in self.bump]

    @cached_property
    def lateral_scrub(self) -> list[float]:
        """Lateral contact-patch positions in inches over wheel travel."""
        return [s.wheel_contact.x for s in self.bump]

    @cached_property
    def roll_center(self) -> list[float]:
        """Roll-center heights above ground in inches over wheel travel."""
        return [s.roll_center.z - s.wheel_contact.z for s in self.bump]

    @cached_property
    def target_values(self) -> SuspensionTargetValues:
        """Measure the sampled sweep over the target travel range.

        Raises:
            ValueError: If the travel samples do not cover the target range.

        """
        travel_in = np.asarray(self.bump_range)
        if not len(travel_in) or min(travel_in) > BUMP[0] or max(travel_in) < BUMP[1]:
            msg = "Suspension samples must cover the target travel range."
            raise ValueError(msg)
        order = np.argsort(travel_in)
        travel_in = travel_in[order]
        if np.any(np.diff(travel_in) <= 0):
            msg = "Suspension travel samples must be distinct."
            raise ValueError(msg)
        camber_deg = np.asarray(self.camber)[order]
        toe_deg = np.asarray(self.bump_steer)[order]
        center_x_in = np.asarray(self.lateral_scrub)[order]
        roll_center_in = np.asarray(self.roll_center)[order]

        # NOTE: Interpolated endpoints keep extra travel outside the target range
        # from changing the result and include rest when it was not sampled.
        def interval_values(
            values: np.ndarray,
            start_in: float,
            end_in: float,
        ) -> np.ndarray:
            """Return ordered samples and interpolated boundaries in an interval.

            Returns:
                The values within the requested travel interval.

            """
            inside = (travel_in > start_in) & (travel_in < end_in)
            return np.concatenate(
                (
                    [np.interp(start_in, travel_in, values)],
                    values[inside],
                    [np.interp(end_in, travel_in, values)],
                ),
            )

        static_camber_deg = float(np.interp(0, travel_in, camber_deg))
        lower = int(np.searchsorted(travel_in, 0, side="left")) - 1
        upper = int(np.searchsorted(travel_in, 0, side="right"))
        slope_deg_per_in = (camber_deg[upper] - camber_deg[lower]) / (
            travel_in[upper] - travel_in[lower]
        )
        bump_camber_deg = interval_values(camber_deg, 0, BUMP[1])
        droop_camber_deg = interval_values(camber_deg, BUMP[0], 0)
        toe_range_deg = interval_values(toe_deg, *BUMP)
        static_center_x_in = float(np.interp(0, travel_in, center_x_in))
        return SuspensionTargetValues(
            static_camber_deg=static_camber_deg,
            camber_slope_deg_per_in=float(slope_deg_per_in),
            bump_camber_increase_deg=float(max(0.0, np.max(np.diff(bump_camber_deg)))),
            bump_camber_max_deg=float(np.max(bump_camber_deg)),
            droop_camber_max_deg=float(np.max(droop_camber_deg)),
            bump_steer_pp_deg=float(np.ptp(toe_range_deg)),
            wheel_center_x_in=static_center_x_in,
            roll_center_pp_in=float(np.ptp(interval_values(roll_center_in, *BUMP))),
        )

    @cached_property
    def target_tests(self) -> SuspensionTargetTests:
        """Evaluate the sampled sweep against the suspension targets."""
        values = self.target_values
        return SuspensionTargetTests(
            static_camber=bool(
                STATIC_CAMBER[0].deg
                <= values.static_camber_deg
                <= STATIC_CAMBER[1].deg,
            ),
            camber_slope=bool(
                STATIC_CAMBER_SLOPE[0].deg
                <= values.camber_slope_deg_per_in
                <= STATIC_CAMBER_SLOPE[1].deg,
            ),
            bump_camber_monotonic=values.bump_camber_increase_deg <= 0,
            bump_camber=values.bump_camber_max_deg <= BUMP_CAMBER_MAX.deg,
            droop_camber=values.droop_camber_max_deg <= DROOP_CAMBER_MAX.deg,
            bump_steer=values.bump_steer_pp_deg <= BUMP_STEER_PP.deg,
            roll_center=values.roll_center_pp_in <= ROLL_CENTER_PP,
            wheel_center=WHEEL_CENTER_X[0]
            <= values.wheel_center_x_in
            <= WHEEL_CENTER_X[1],
        )

    def report(self) -> None:
        """Print target values and pass or fail results for the sweep."""
        values = self.target_values
        tests = self.target_tests

        def line(
            passed: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
            name: str,
            actual: str,
            target: str,
        ) -> None:
            """Print one color-coded target result."""
            color = Fore.GREEN if passed else Fore.RED
            status = "PASS" if passed else "FAIL"
            actual_len = len(actual)
            actual = re.sub(
                r"-?\d+(?:\.\d+)?",
                rf"{Fore.BLUE}\g<0>{Style.RESET_ALL}",
                actual,
            ) + " " * (15 - actual_len)
            target = re.sub(
                r"-?\d+(?:\.\d+)?",
                rf"{Fore.MAGENTA}\g<0>{Style.RESET_ALL}",
                target,
            )
            print(
                f"{color}[{status}]{Style.RESET_ALL} "
                f"{name:30}: {actual} (target {target})",
            )

        line(
            tests.static_camber,
            "Static camber",
            f"{values.static_camber_deg:6.3f} deg",
            f"{STATIC_CAMBER[0].deg:.3f} to {STATIC_CAMBER[1].deg:.3f} deg",
        )
        line(
            tests.camber_slope,
            "Static camber slope (soft)",
            f"{values.camber_slope_deg_per_in:6.3f} deg/in",
            f"{STATIC_CAMBER_SLOPE[0].deg:.3f} to "
            f"{STATIC_CAMBER_SLOPE[1].deg:.3f} deg/in",
        )
        line(
            tests.bump_camber_monotonic,
            "Bump camber increase (soft)",
            f"{values.bump_camber_increase_deg:6.3f} deg",
            "0 deg",
        )
        line(
            tests.bump_camber,
            "Maximum bump camber",
            f"{values.bump_camber_max_deg:6.3f} deg",
            f"<= {BUMP_CAMBER_MAX.deg:.3f} deg",
        )
        line(
            tests.droop_camber,
            "Maximum droop camber",
            f"{values.droop_camber_max_deg:6.3f} deg",
            f"<= {DROOP_CAMBER_MAX.deg:.3f} deg",
        )
        line(
            tests.bump_steer,
            "Bump steer peak-to-peak",
            f"{values.bump_steer_pp_deg:6.3f} deg",
            f"<= {BUMP_STEER_PP.deg:.3f} deg",
        )
        line(
            tests.roll_center,
            "Roll center peak-to-peak",
            f"{values.roll_center_pp_in:6.3f} in",
            f"<= {ROLL_CENTER_PP:.3f} in",
        )
        line(
            tests.wheel_center,
            "Static wheel center X",
            f"{values.wheel_center_x_in:6.3f} in",
            f"{WHEEL_CENTER_X[0]:.3f} to {WHEEL_CENTER_X[1]:.3f} in",
        )

    def plot(
        self,
        path: Path,
    ) -> None:
        """Save suspension responses to bump travel and steering rack movement."""
        fig, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
        axes[0, 0].plot(self.bump_range, self.camber)
        axes[0, 0].set_title("Camber")
        axes[0, 0].set_ylabel("Camber (deg)")

        axes[0, 1].plot(self.bump_range, self.bump_steer)
        axes[0, 1].set_title("Bump steer")
        axes[0, 1].set_ylabel("Toe (deg)")

        axes[1, 0].plot(self.bump_range, self.lateral_scrub)
        axes[1, 0].set_title("Contact patch lateral position")
        axes[1, 0].set_ylabel("Lateral position (in)")

        axes[1, 1].plot(self.steering_range, [s.toe.deg for s in self.steering])
        axes[1, 1].set_title("Steering response")
        axes[1, 1].set_ylabel("Toe (deg)")

        roll_ax = axes[2, 0]
        roll_ax.plot(self.bump_range, self.roll_center)
        roll_ax.set_title("Roll center")
        roll_ax.set_ylabel("Height above ground (in)")
        roll_ax.set_xlabel("Bump travel (in)")
        roll_midpoint_in = (min(self.roll_center) + max(self.roll_center)) / 2
        rc_min = roll_midpoint_in - ROLL_CENTER_PP / 2
        rc_max = roll_midpoint_in + ROLL_CENTER_PP / 2
        roll_ax.axhspan(
            rc_min,
            rc_max,
            color="tab:green",
            alpha=0.08,
            label="Roll center peak-to-peak limit",
        )
        roll_ax.axhline(rc_min, color="tab:red", linestyle="--")
        roll_ax.axhline(rc_max, color="tab:red", linestyle="--")
        roll_ax.axvline(0, color="0.5", linestyle=":", linewidth=0.8)
        roll_ax.legend(fontsize="small")
        axes[2, 1].set_visible(False)

        camber_ax = axes[0, 0]
        # NOTE: The camber ceiling changes at rest height; static targets only
        # apply at zero travel, not across the full sweep.
        for start_in, end_in, limit_deg, label in (
            (BUMP[0], 0, DROOP_CAMBER_MAX.deg, "Droop camber maximum"),
            (0, BUMP[1], BUMP_CAMBER_MAX.deg, "Bump camber maximum"),
        ):
            camber_ax.plot(
                [start_in, end_in],
                [limit_deg, limit_deg],
                color="tab:red",
                linestyle="--",
                label=label,
            )
        camber_ax.plot(
            [0, 0],
            [angle.deg for angle in STATIC_CAMBER],
            color="tab:green",
            linewidth=4,
            label="Static camber target",
        )
        order = np.argsort(self.bump_range)
        travel_in = np.asarray(self.bump_range)[order]
        camber_deg = np.asarray(self.camber)[order]
        if travel_in[0] <= 0 <= travel_in[-1]:
            static_camber_deg = float(np.interp(0, travel_in, camber_deg))
            slope_travel_in = np.array(
                [max(travel_in[0], -0.5), min(travel_in[-1], 0.5)],
            )
            for index, slope in enumerate(STATIC_CAMBER_SLOPE):
                camber_ax.plot(
                    slope_travel_in,
                    static_camber_deg + slope.deg * slope_travel_in,
                    color="tab:orange",
                    linestyle=":",
                    label="Static camber slope target" if index == 0 else None,
                )
        camber_ylim = camber_ax.get_ylim()
        for start_in, end_in, limit_deg in (
            (BUMP[0], 0, DROOP_CAMBER_MAX.deg),
            (0, BUMP[1], BUMP_CAMBER_MAX.deg),
        ):
            if camber_ylim[1] > limit_deg:
                camber_ax.fill_between(
                    [start_in, end_in],
                    limit_deg,
                    camber_ylim[1],
                    color="tab:red",
                    alpha=0.08,
                )
        camber_ax.set_ylim(camber_ylim)

        toe_ax = axes[0, 1]
        toe_midpoint_deg = (min(self.bump_steer) + max(self.bump_steer)) / 2
        toe_min_deg = toe_midpoint_deg - BUMP_STEER_PP.deg / 2
        toe_max_deg = toe_midpoint_deg + BUMP_STEER_PP.deg / 2
        for limit_deg in (toe_min_deg, toe_max_deg):
            toe_ax.axhline(limit_deg, color="tab:red", linestyle="--")
        toe_ax.axhspan(
            toe_min_deg,
            toe_max_deg,
            color="tab:green",
            alpha=0.08,
            label="Bump steer peak-to-peak limit",
        )

        axes[1, 0].plot(
            [0, 0],
            WHEEL_CENTER_X,
            color="tab:green",
            linewidth=4,
            label="Static wheel center target",
        )
        for ax in (camber_ax, toe_ax, axes[1, 0]):
            ax.axvline(0, color="0.5", linestyle=":", linewidth=0.8)
            ax.legend(fontsize="small")

        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xlabel("Bump travel (in)")
        axes[1, 1].set_xlabel("Steering rack displacement (in)")
        for ax in axes.flat:
            ax.grid(visible=True, alpha=0.3)
        fig.savefig(path, dpi=200)
        plt.close(fig)
