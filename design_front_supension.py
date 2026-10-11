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
    lower_pickup_point = geometry.lower_pickup()
    upper_pickup_point = geometry.upper_pickup()
    figure, (axis_geometry, axis_camber, axis_wheel_contact) = plt.subplots(
        ncols=3,
        figsize=(24.0, 8.0),
        gridspec_kw={"width_ratios": (2.0, 1.0, 1.0)},
        layout="constrained",
    )
    lower_trunnion_points = [
        geometry.lower_trunnion(lower_angle_rad) for lower_angle_rad in lower_angles_rad
    ]
    upper_trunnion_points = [
        geometry.upper_trunnion(lower_angle_rad) for lower_angle_rad in lower_angles_rad
    ]
    wheel_contact_points = [
        geometry.wheel_contact(lower_angle_rad) for lower_angle_rad in lower_angles_rad
    ]
    ride_heights_in = [
        -wheel_contact_point.z for wheel_contact_point in wheel_contact_points
    ]
    theta_outer_horizon_deg = [
        float(
            np.rad2deg(geometry.outer_horizon_rad(lower_angle_rad) - np.pi / 2.0),
        )
        for lower_angle_rad in lower_angles_rad
    ]
    geometry_point_paths = (
        ("Lower trunnion", lower_trunnion_points, "tab:green"),
        ("Upper trunnion", upper_trunnion_points, "tab:red"),
        ("Wheel contact", wheel_contact_points, "tab:purple"),
    )
    for point_name, geometry_point_path, path_color in geometry_point_paths:
        axis_geometry.plot(
            [point.x for point in geometry_point_path],
            [point.z for point in geometry_point_path],
            color=path_color,
            linestyle=":",
            linewidth=2.0,
            label=f"{point_name} path",
            zorder=1,
        )
        axis_geometry.plot(
            [-point.x for point in geometry_point_path],
            [point.z for point in geometry_point_path],
            color=path_color,
            linestyle=":",
            linewidth=2.0,
            zorder=1,
        )

    state_angles = (
        (lower_angles_rad[0], "Minimum", "tab:orange"),
        (lower_angle_rest_rad, "Rest", "tab:green"),
        (lower_angles_rad[-1], "Maximum", "tab:red"),
    )
    wheel_contact_rest_x_in = geometry.wheel_contact(lower_angle_rest_rad).x
    for lower_angle_rad, state_name, state_color in state_angles:
        lower_trunnion_point = geometry.lower_trunnion(lower_angle_rad)
        upper_trunnion_point = geometry.upper_trunnion(lower_angle_rad)
        axle_point = geometry.axle(lower_angle_rad)
        wheel_center_point = geometry.wheel_center(lower_angle_rad)
        wheel_contact_point = geometry.wheel_contact(lower_angle_rad)
        wheel_far_edge_point = wheel_center_point + (
            wheel_center_point - wheel_contact_point
        )
        four_bar_points = (
            lower_pickup_point,
            upper_pickup_point,
            upper_trunnion_point,
            lower_trunnion_point,
            lower_pickup_point,
        )
        four_bar_x_in = [point.x for point in four_bar_points]
        four_bar_z_in = [point.z for point in four_bar_points]
        lower_angle_deg = float(np.rad2deg(lower_angle_rad))
        axis_geometry.plot(
            four_bar_x_in,
            four_bar_z_in,
            color=state_color,
            linewidth=2.5,
            label=f"{state_name}: {lower_angle_deg:.1f} degrees",
            zorder=2,
        )
        axis_geometry.scatter(
            four_bar_x_in[:-1],
            four_bar_z_in[:-1],
            color=state_color,
            zorder=3,
        )
        axis_geometry.plot(
            [-x_in for x_in in four_bar_x_in],
            four_bar_z_in,
            color=state_color,
            linewidth=2.0,
            zorder=2,
        )
        axis_geometry.scatter(
            [-x_in for x_in in four_bar_x_in[:-1]],
            four_bar_z_in[:-1],
            color=state_color,
            zorder=3,
        )
        axis_geometry.plot(
            [axle_point.x, wheel_center_point.x],
            [axle_point.z, wheel_center_point.z],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        axis_geometry.plot(
            [-axle_point.x, -wheel_center_point.x],
            [axle_point.z, wheel_center_point.z],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        axis_geometry.plot(
            [wheel_contact_point.x, wheel_far_edge_point.x],
            [
                wheel_contact_point.z,
                wheel_far_edge_point.z,
            ],
            color=state_color,
            linewidth=2.0,
            zorder=2,
        )
        axis_geometry.plot(
            [-wheel_contact_point.x, -wheel_far_edge_point.x],
            [wheel_contact_point.z, wheel_far_edge_point.z],
            color=state_color,
            linewidth=2.0,
            zorder=2,
        )
        axis_camber.scatter(
            -wheel_contact_point.z,
            float(
                np.rad2deg(
                    geometry.outer_horizon_rad(lower_angle_rad) - np.pi / 2.0,
                ),
            ),
            color=state_color,
            zorder=2,
        )
        axis_wheel_contact.scatter(
            -wheel_contact_point.z,
            wheel_contact_point.x - wheel_contact_rest_x_in,
            color=state_color,
            zorder=2,
        )

    axis_geometry.set_aspect("equal", adjustable="box")
    axis_geometry.set_xlabel("Lateral position (in)")
    axis_geometry.set_ylabel("Height (in)")
    axis_geometry.set_title(
        "Front suspension four-bar, axle, wheel, and point paths: x-z view",
    )
    axis_geometry.legend(loc="best")
    axis_camber.plot(
        ride_heights_in,
        theta_outer_horizon_deg,
        color="tab:blue",
        zorder=1,
    )
    axis_camber.set_xlabel("Ride height (in)")
    axis_camber.set_ylabel("Camber (degrees)")
    axis_camber.set_title("Camber versus ride height")
    axis_camber.grid()
    axis_wheel_contact.plot(
        ride_heights_in,
        [
            wheel_contact_point.x - wheel_contact_rest_x_in
            for wheel_contact_point in wheel_contact_points
        ],
        color="tab:purple",
        zorder=1,
    )
    axis_wheel_contact.set_xlabel("Ride height (in)")
    axis_wheel_contact.set_ylabel("Contact-patch lateral displacement from rest (in)")
    axis_wheel_contact.set_title("Wheel contact lateral position versus ride height")
    axis_wheel_contact.grid()
    figure.savefig(output_path, dpi=200)

    plt.close(figure)


def main() -> int:
    """Create the front-suspension x-z plot.

    Returns:
        Zero after saving the plot.

    """
    lower_angle_rest_deg = -5.0
    lower_angle_min_deg = lower_angle_rest_deg - 15.0
    lower_angle_max_deg = lower_angle_rest_deg + 15.0
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
