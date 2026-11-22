"""Plot selected front-suspension candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from front_suspension.candidates import CANDIDATES
from front_suspension.helpers.sweeper import SuspensionSweep


def main() -> None:
    """Save geometry and sweep plots for selected candidates."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("index", nargs="?", type=int, help="1-based candidate index")
    selection.add_argument("--all", action="store_true", help="plot every candidate")
    args = parser.parse_args()

    if args.all:
        selected = enumerate(CANDIDATES, start=1)
    elif 1 <= args.index <= len(CANDIDATES):
        selected = [(args.index, CANDIDATES[args.index - 1])]
    else:
        parser.error(f"index must be between 1 and {len(CANDIDATES)}")

    output = Path("out/front_suspension")
    output.mkdir(parents=True, exist_ok=True)
    for index, fixed in selected:
        fixed.plot(output / f"candidate_{index:02d}_geometry.png")
        sweeper = SuspensionSweep(fixed)
        print(f"Candidate {index:02d}: {fixed}")
        sweeper.report()
        sweeper.plot(output / f"candidate_{index:02d}_sweep.png")


if __name__ == "__main__":
    main()
