"""Sweep front-suspension travel and steering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from mg.front_suspension.constraints import (
    BUMP,
    BUMP_CAMBER_MAX,
    BUMP_STEER_PP,
    DROOP_CAMBER_MAX,
    STATIC_CAMBER,
    STEERING,
    WHEEL_CENTER_X,
)
from mg.front_suspension.geometry import GeometryFixed, Solver

if TYPE_CHECKING:
    from pathlib import Path

    from mg.front_suspension.geometry import SuspensionState


N = 31


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

    def plot(
        self,
        path: Path,
    ) -> None:
        """Save suspension responses to bump travel and steering rack movement."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
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
