"""Plot front-suspension travel in chassis and ground coordinates."""

from __future__ import annotations

from pathlib import Path

from mg.front_suspension.geometry import GeometryFixed
from mg.front_suspension.sweeper import SuspensionSweep


def main() -> None:
    """Plot suspension geometry for the requested wheel travel."""
    fixed = GeometryFixed()
    fixed.plot(Path("front_suspension_geometry.png"))

    sweeper = SuspensionSweep(fixed)
    sweeper.plot(Path("front_suspension_sweep.png"))


if __name__ == "__main__":
    main()
