"""Plot front-suspension travel in chassis and ground coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mg.front_suspension.geometry import GeometryFixed, Solver


def main() -> None:
    """Plot suspension geometry for the requested wheel travel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("front_suspension_travel.png"),
    )
    args = parser.parse_args()

    fixed = GeometryFixed()
    states = {
        "rest": (0, "green"),
        "droop": (-2, "orange"),
        "bump": (2.5, "red"),
    }
    fig, ax = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    for label, (height, color) in states.items():
        state = Solver.at(fixed, height)
        state.plot(ax[0, 0], "XZ", label=label, color=color)
        state.plot(ax[0, 1], "-YZ", label=label, color=color)
        state.plot(ax[0, 2], "-XY", label=label, color=color)
        state.world.plot(ax[1, 0], "XZ", label=label, color=color)
        state.world.plot(ax[1, 1], "-YZ", label=label, color=color)
        state.world.plot(ax[1, 2], "-XY", label=label, color=color)

    for axes, title, xlabel, ylabel in (
        (ax[0, 0], "Chassis: front view", "X (in)", "Z (in)"),
        (ax[0, 1], "Chassis: side view", "-Y (in)", "Z (in)"),
        (ax[0, 2], "Chassis: top view", "-X (in)", "Y (in)"),
        (ax[1, 0], "Ground: front view", "X (in)", "Z (in)"),
        (ax[1, 1], "Ground: side view", "-Y (in)", "Z (in)"),
        (ax[1, 2], "Ground: top view", "-X (in)", "Y (in)"),
    ):
        axes.set_title(title)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_aspect("equal", adjustable="box")
        axes.grid(visible=True, alpha=0.3)
        axes.legend()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
