"""Plot the passing front-suspension search candidates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mg.front_suspension.sweeper import SuspensionSweep

if TYPE_CHECKING:
    from mg.front_suspension.geometry import GeometryFixed


def main() -> None:
    """Save geometry and sweep plots for each candidate."""
    candidates: list[GeometryFixed] = []
    output = Path("out")
    output.mkdir(parents=True, exist_ok=True)
    for index, fixed in enumerate(candidates, start=1):
        fixed.plot(output / f"candidate_{index:02d}_geometry.png")
        sweeper = SuspensionSweep(fixed)
        print(f"Candidate {index:02d}: {fixed}")
        sweeper.report()
        sweeper.plot(output / f"candidate_{index:02d}_sweep.png")


if __name__ == "__main__":
    main()
