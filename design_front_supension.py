"""Plot front-suspension pickup and trunnion locations in the x-z plane."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from mg.front_suspension import Geometry

if TYPE_CHECKING:
    from collections.abc import Sequence


def plot_xz(
    geometry: Geometry,
    lower_angles_rad: Sequence[float],
    lower_angle_rest_rad: float,
    output_path: Path,
) -> None:
    """Plot three four-bar positions and all point paths in the x-z plane.

    Args:
        geometry: Fixed dimensions for the modeled suspension side.
        lower_angles_rad: Sweep of lower control-arm angles in radians.
        lower_angle_rest_rad: Lower control-arm rest angle in radians.
        output_path: PNG file that receives the finished plot.

    """
    lower_pickup = geometry.lower_pickup()
    upper_pickup = geometry.upper_pickup()
    figure, axis = plt.subplots(layout="constrained")
    lower_trunnion_path = [
        geometry.lower_trunnion(lower_angle_rad) for lower_angle_rad in lower_angles_rad
    ]
    upper_trunnion_path = [
        geometry.upper_trunnion(lower_angle_rad) for lower_angle_rad in lower_angles_rad
    ]
    point_paths = (
        ("Lower trunnion", lower_trunnion_path, "tab:green"),
        ("Upper trunnion", upper_trunnion_path, "tab:red"),
    )
    for point_name, point_path, color in point_paths:
        axis.plot(
            [point.x for point in point_path],
            [point.z for point in point_path],
            color=color,
            linestyle=":",
            linewidth=2.0,
            label=f"{point_name} path",
            zorder=1,
        )

    state_angles = (
        (lower_angles_rad[0], "Minimum", "tab:blue"),
        (lower_angle_rest_rad, "Rest", "tab:gray"),
        (lower_angles_rad[-1], "Maximum", "tab:orange"),
    )
    for lower_angle_rad, state_name, color in state_angles:
        lower_trunnion = geometry.lower_trunnion(lower_angle_rad)
        upper_trunnion = geometry.upper_trunnion(lower_angle_rad)
        points = (
            lower_pickup,
            upper_pickup,
            upper_trunnion,
            lower_trunnion,
            lower_pickup,
        )
        x_in = [point.x for point in points]
        z_in = [point.z for point in points]
        angle_deg = float(np.rad2deg(lower_angle_rad))
        axis.plot(
            x_in,
            z_in,
            color=color,
            linewidth=2.5,
            label=f"{state_name}: {angle_deg:.1f} degrees",
            zorder=2,
        )
        axis.scatter(x_in[:-1], z_in[:-1], color=color, zorder=3)

    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Lateral x (in)")
    axis.set_ylabel("Vertical z (in)")
    axis.set_title("Front suspension four-bar and point paths: x-z view")
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=200)

    plt.close(figure)


def main() -> int:
    """Create the front-suspension x-z plot.

    Returns:
        Zero after saving the plot.

    """
    lower_angle_rest_deg = -5.0
    lower_angle_min_deg = lower_angle_rest_deg - 5.0
    lower_angle_max_deg = lower_angle_rest_deg + 5.0
    lower_angles_rad = [
        float(np.deg2rad(angle_deg))
        for angle_deg in np.linspace(lower_angle_min_deg, lower_angle_max_deg, 101)
    ]
    plot_xz(
        Geometry(),
        lower_angles_rad,
        float(np.deg2rad(lower_angle_rest_deg)),
        Path("front_suspension_xz.png"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
