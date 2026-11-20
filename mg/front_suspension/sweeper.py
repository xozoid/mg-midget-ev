"""Sweep front-suspension travel and steering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from mg.front_suspension.constraints import MIN_BUMP, MIN_DROOP, MIN_LEFT, MIN_RIGHT
from mg.front_suspension.geometry import GeometryFixed, Solver

if TYPE_CHECKING:
    from pathlib import Path

    from mg.front_suspension.geometry import SuspensionState


N = 31


@dataclass(frozen=True)
class SuspensionSweep:
    """Evaluate suspension geometry over travel and steering ranges."""

    geometry: GeometryFixed

    bump_range: list[float] = field(
        default_factory=lambda: list(np.linspace(MIN_DROOP, MIN_BUMP, N, dtype=float)),
    )
    steering_range: list[float] = field(
        default_factory=lambda: list(np.linspace(MIN_LEFT, MIN_RIGHT, N, dtype=float)),
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

        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set_xlabel("Bump travel (in)")
        axes[1, 1].set_xlabel("Steering rack displacement (in)")
        for ax in axes.flat:
            ax.grid(visible=True, alpha=0.3)
        fig.savefig(path, dpi=200)
