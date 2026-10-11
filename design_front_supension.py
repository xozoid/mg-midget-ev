"""Plot front-suspension pickup and trunnion locations in the x-z plane."""

from __future__ import annotations

from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import root_scalar

from mg.front_suspension import Geometry

if TYPE_CHECKING:
    from collections.abc import Sequence


def plot_xz(
    geometry: Geometry,
    lower_angles_rad: Sequence[float],
    toe_angles_rad: Sequence[float],
    toe_angles_right_turn_rad: Sequence[float],
    toe_angles_left_turn_rad: Sequence[float],
    lower_angle_max_bump_rad: float,
    toe_angle_max_bump_rad: float,
    lower_angle_rest_rad: float,
    toe_angle_rest_rad: float,
    lower_angle_max_drop_rad: float,
    toe_angle_max_drop_rad: float,
    geometry_output_path: Path,
    ride_height_output_path: Path,
) -> None:
    """Plot three four-bar positions and all point paths in the x-z plane.

    Args:
        geometry: Fixed dimensions for the modeled suspension side.
        lower_angles_rad: Sweep of lower control-arm angles in radians.
        toe_angles_rad: Toe angles corresponding to ``lower_angles_rad``.
        toe_angles_right_turn_rad: Toe angles with a +5 degree right turn at rest.
        toe_angles_left_turn_rad: Toe angles with a -5 degree left turn at rest.
        lower_angle_max_bump_rad: Lower control-arm angle at maximum bump.
        toe_angle_max_bump_rad: Toe angle at maximum bump.
        lower_angle_rest_rad: Lower control-arm rest angle in radians.
        toe_angle_rest_rad: Toe angle at rest.
        lower_angle_max_drop_rad: Lower control-arm angle at maximum drop.
        toe_angle_max_drop_rad: Toe angle at maximum drop.
        geometry_output_path: PNG file for the two geometry views.
        ride_height_output_path: PNG file for the ride-height analysis plots.

    """
    lower_pickup_point = geometry.lower_pickup()
    upper_pickup_point = geometry.upper_pickup()
    geometry_figure, (
        (axis_geometry, axis_geometry_yz),
        (axis_geometry_contact, axis_geometry_contact_yz),
    ) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(20.0, 16.0),
        layout="constrained",
    )
    ride_height_figure, ride_height_axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.0, 10.0),
        layout="constrained",
    )
    axis_camber, axis_wheel_contact, axis_roll_center, axis_future = (
        ride_height_axes.flatten()
    )
    lower_trunnion_points = [
        geometry.lower_trunnion(lower_angle_rad) for lower_angle_rad in lower_angles_rad
    ]
    upper_trunnion_points = [
        geometry.upper_trunnion(lower_angle_rad) for lower_angle_rad in lower_angles_rad
    ]
    wheel_contact_points = list(
        starmap(
            geometry.wheel_contact,
            zip(lower_angles_rad, toe_angles_rad, strict=True),
        ),
    )
    roll_center_points = list(
        starmap(
            geometry.roll_center,
            zip(lower_angles_rad, toe_angles_rad, strict=True),
        ),
    )
    wheel_center_points = list(
        starmap(
            geometry.wheel_center,
            zip(lower_angles_rad, toe_angles_rad, strict=True),
        ),
    )
    tie_rod_points = list(
        starmap(
            geometry.tie_rod_point,
            zip(lower_angles_rad, toe_angles_rad, strict=True),
        ),
    )
    steering_rod_link_points = list(
        starmap(
            geometry.steering_rod_link_point,
            zip(lower_angles_rad, toe_angles_rad, strict=True),
        ),
    )
    ride_heights_in = [
        -wheel_contact_point.z for wheel_contact_point in wheel_contact_points
    ]
    theta_outer_horizon_deg = [
        float(
            np.rad2deg(geometry.outer_horizon_rad(lower_angle_rad) - np.pi / 2.0),
        )
        for lower_angle_rad in lower_angles_rad
    ]
    roll_center_heights_above_ground_in = [
        roll_center_point.z - wheel_center_point.z + geometry.wheel_radius_in
        for roll_center_point, wheel_center_point in zip(
            roll_center_points,
            wheel_center_points,
            strict=True,
        )
    ]
    geometry_point_paths = (
        ("Lower trunnion", lower_trunnion_points, "tab:green"),
        ("Upper trunnion", upper_trunnion_points, "tab:red"),
        ("Wheel contact", wheel_contact_points, "tab:purple"),
        ("Tie-rod point", tie_rod_points, "tab:brown"),
        ("Steering-rod link point", steering_rod_link_points, "tab:cyan"),
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
        axis_geometry_contact.plot(
            [point.x for point in geometry_point_path],
            [
                point.z - wheel_contact_point.z
                for point, wheel_contact_point in zip(
                    geometry_point_path,
                    wheel_contact_points,
                    strict=True,
                )
            ],
            color=path_color,
            linestyle=":",
            linewidth=2.0,
            label=f"{point_name} path",
            zorder=1,
        )
    axis_geometry_yz.plot(
        [tie_rod_point.y for tie_rod_point in tie_rod_points],
        [tie_rod_point.z for tie_rod_point in tie_rod_points],
        color="tab:brown",
        linestyle=":",
        linewidth=2.0,
        label="Tie-rod point path",
        zorder=1,
    )
    axis_geometry_contact_yz.plot(
        [tie_rod_point.y for tie_rod_point in tie_rod_points],
        [
            tie_rod_point.z - wheel_contact_point.z
            for tie_rod_point, wheel_contact_point in zip(
                tie_rod_points,
                wheel_contact_points,
                strict=True,
            )
        ],
        color="tab:brown",
        linestyle=":",
        linewidth=2.0,
        label="Tie-rod point path",
        zorder=1,
    )
    axis_geometry_yz.plot(
        [
            steering_rod_link_point.y
            for steering_rod_link_point in steering_rod_link_points
        ],
        [
            steering_rod_link_point.z
            for steering_rod_link_point in steering_rod_link_points
        ],
        color="tab:cyan",
        linestyle=":",
        linewidth=2.0,
        label="Steering-rod link point path",
        zorder=1,
    )
    axis_geometry_contact_yz.plot(
        [
            steering_rod_link_point.y
            for steering_rod_link_point in steering_rod_link_points
        ],
        [
            steering_rod_link_point.z - wheel_contact_point.z
            for steering_rod_link_point, wheel_contact_point in zip(
                steering_rod_link_points,
                wheel_contact_points,
                strict=True,
            )
        ],
        color="tab:cyan",
        linestyle=":",
        linewidth=2.0,
        label="Steering-rod link point path",
        zorder=1,
    )

    state_angles = (
        (
            lower_angle_max_bump_rad,
            toe_angle_max_bump_rad,
            "Maximum bump",
            "tab:orange",
        ),
        (lower_angle_rest_rad, toe_angle_rest_rad, "Rest", "tab:green"),
        (
            lower_angle_max_drop_rad,
            toe_angle_max_drop_rad,
            "Maximum drop",
            "tab:red",
        ),
    )
    wheel_ellipse_xz_width_in = (
        2.0 * geometry.wheel_radius_in * abs(np.sin(toe_angle_rest_rad))
    )
    wheel_ellipse_yz_width_in = (
        2.0 * geometry.wheel_radius_in * abs(np.cos(toe_angle_rest_rad))
    )
    wheel_contact_rest_x_in = geometry.wheel_contact(
        lower_angle_rest_rad,
        toe_angle_rest_rad,
    ).x
    for lower_angle_rad, toe_angle_rad, state_name, state_color in state_angles:
        lower_trunnion_point = geometry.lower_trunnion(lower_angle_rad)
        upper_trunnion_point = geometry.upper_trunnion(lower_angle_rad)
        instant_center_point = geometry.instant_center(lower_angle_rad)
        axle_point = geometry.axle(lower_angle_rad)
        wheel_center_point = geometry.wheel_center(lower_angle_rad, toe_angle_rad)
        wheel_contact_point = geometry.wheel_contact(lower_angle_rad, toe_angle_rad)
        roll_center_point = geometry.roll_center(lower_angle_rad, toe_angle_rad)
        tie_rod_point = geometry.tie_rod_point(lower_angle_rad, toe_angle_rad)
        steering_rod_link_point = geometry.steering_rod_link_point(
            lower_angle_rad,
            toe_angle_rad,
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
        axis_geometry.scatter(
            [instant_center_point.x],
            [instant_center_point.z],
            color="tab:blue",
            zorder=4,
        )
        axis_geometry.plot(
            [instant_center_point.x, wheel_contact_point.x],
            [instant_center_point.z, wheel_contact_point.z],
            color="tab:blue",
            linewidth=1.5,
            zorder=1,
        )
        axis_geometry.scatter(
            [roll_center_point.x],
            [roll_center_point.z],
            color="tab:blue",
            marker="s",
            label="Roll center" if state_name == "Maximum bump" else None,
            zorder=4,
        )
        axis_geometry.scatter(
            [tie_rod_point.x],
            [tie_rod_point.z],
            color=state_color,
            marker="D",
            zorder=3,
        )
        axis_geometry.scatter(
            [steering_rod_link_point.x],
            [steering_rod_link_point.z],
            color=state_color,
            marker="X",
            zorder=3,
        )
        axis_geometry.plot(
            [instant_center_point.x, lower_trunnion_point.x],
            [instant_center_point.z, lower_trunnion_point.z],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        axis_geometry.plot(
            [instant_center_point.x, upper_trunnion_point.x],
            [instant_center_point.z, upper_trunnion_point.z],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        axis_geometry.plot(
            [axle_point.x, wheel_center_point.x],
            [axle_point.z, wheel_center_point.z],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        axis_geometry.add_patch(
            Ellipse(
                (wheel_center_point.x, wheel_center_point.z),
                width=wheel_ellipse_xz_width_in,
                height=2.0 * geometry.wheel_radius_in,
                angle=float(
                    np.rad2deg(-geometry.outer_horizon_rad(lower_angle_rad)) - 90.0,
                ),
                edgecolor=state_color,
                facecolor="none",
                linewidth=2.0,
                zorder=2,
            ),
        )
        axis_geometry_contact.plot(
            four_bar_x_in,
            [point.z - wheel_contact_point.z for point in four_bar_points],
            color=state_color,
            linewidth=2.5,
            label=f"{state_name}: {lower_angle_deg:.1f} degrees",
            zorder=2,
        )
        axis_geometry_contact.scatter(
            four_bar_x_in[:-1],
            [point.z - wheel_contact_point.z for point in four_bar_points[:-1]],
            color=state_color,
            zorder=3,
        )
        axis_geometry_contact.scatter(
            [instant_center_point.x],
            [instant_center_point.z - wheel_contact_point.z],
            color="tab:blue",
            zorder=4,
        )
        axis_geometry_contact.plot(
            [instant_center_point.x, wheel_contact_point.x],
            [instant_center_point.z - wheel_contact_point.z, 0.0],
            color="tab:blue",
            linewidth=1.5,
            zorder=1,
        )
        axis_geometry_contact.scatter(
            [roll_center_point.x],
            [roll_center_point.z - wheel_contact_point.z],
            color="tab:blue",
            marker="s",
            label="Roll center" if state_name == "Maximum bump" else None,
            zorder=4,
        )
        axis_geometry_contact.scatter(
            [tie_rod_point.x],
            [tie_rod_point.z - wheel_contact_point.z],
            color=state_color,
            marker="D",
            zorder=3,
        )
        axis_geometry_contact.scatter(
            [steering_rod_link_point.x],
            [steering_rod_link_point.z - wheel_contact_point.z],
            color=state_color,
            marker="X",
            zorder=3,
        )
        axis_geometry_contact.plot(
            [instant_center_point.x, lower_trunnion_point.x],
            [
                instant_center_point.z - wheel_contact_point.z,
                lower_trunnion_point.z - wheel_contact_point.z,
            ],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        axis_geometry_contact.plot(
            [instant_center_point.x, upper_trunnion_point.x],
            [
                instant_center_point.z - wheel_contact_point.z,
                upper_trunnion_point.z - wheel_contact_point.z,
            ],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        axis_geometry_contact.plot(
            [axle_point.x, wheel_center_point.x],
            [
                axle_point.z - wheel_contact_point.z,
                wheel_center_point.z - wheel_contact_point.z,
            ],
            color=state_color,
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        axis_geometry_contact.add_patch(
            Ellipse(
                (
                    wheel_center_point.x,
                    wheel_center_point.z - wheel_contact_point.z,
                ),
                width=wheel_ellipse_xz_width_in,
                height=2.0 * geometry.wheel_radius_in,
                angle=float(
                    np.rad2deg(-geometry.outer_horizon_rad(lower_angle_rad)) - 90.0,
                ),
                edgecolor=state_color,
                facecolor="none",
                linewidth=2.0,
                zorder=2,
            ),
        )
        axis_geometry_yz.add_patch(
            Ellipse(
                (wheel_center_point.y, wheel_center_point.z),
                width=wheel_ellipse_yz_width_in,
                height=2.0 * geometry.wheel_radius_in,
                edgecolor=state_color,
                facecolor="none",
                linewidth=2.0,
                label=f"{state_name}: {lower_angle_deg:.1f} degrees",
                zorder=2,
            ),
        )
        axis_geometry_yz.scatter(
            [roll_center_point.y],
            [roll_center_point.z],
            color="tab:blue",
            marker="s",
            label="Roll center" if state_name == "Maximum bump" else None,
            zorder=4,
        )
        axis_geometry_yz.scatter(
            [tie_rod_point.y],
            [tie_rod_point.z],
            color=state_color,
            marker="D",
            zorder=3,
        )
        axis_geometry_yz.scatter(
            [steering_rod_link_point.y],
            [steering_rod_link_point.z],
            color=state_color,
            marker="X",
            zorder=3,
        )
        axis_geometry_contact_yz.add_patch(
            Ellipse(
                (
                    wheel_center_point.y,
                    geometry.wheel_radius_in,
                ),
                width=wheel_ellipse_yz_width_in,
                height=2.0 * geometry.wheel_radius_in,
                edgecolor=state_color,
                facecolor="none",
                linewidth=2.0,
                label=f"{state_name}: {lower_angle_deg:.1f} degrees",
                zorder=2,
            ),
        )
        axis_geometry_contact_yz.scatter(
            [roll_center_point.y],
            [
                roll_center_point.z - wheel_center_point.z + geometry.wheel_radius_in,
            ],
            color="tab:blue",
            marker="s",
            label="Roll center" if state_name == "Maximum bump" else None,
            zorder=4,
        )
        axis_geometry_contact_yz.scatter(
            [tie_rod_point.y],
            [tie_rod_point.z - wheel_contact_point.z],
            color=state_color,
            marker="D",
            zorder=3,
        )
        axis_geometry_contact_yz.scatter(
            [steering_rod_link_point.y],
            [steering_rod_link_point.z - wheel_contact_point.z],
            color=state_color,
            marker="X",
            zorder=3,
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
        axis_roll_center.scatter(
            -wheel_contact_point.z,
            roll_center_point.z - wheel_center_point.z + geometry.wheel_radius_in,
            color=state_color,
            zorder=2,
        )
        axis_future.scatter(
            -wheel_contact_point.z,
            float(np.rad2deg(toe_angle_rad)),
            color=state_color,
            zorder=2,
        )

    axis_geometry.set_aspect("equal", adjustable="box")
    axis_geometry.set_xlabel("Lateral position (in)")
    axis_geometry.set_ylabel("Height (in)")
    axis_geometry.set_title("Front suspension - fixed chassis (x-z view)")
    axis_geometry.legend(loc="best")
    axis_geometry_yz.set_aspect("equal", adjustable="box")
    axis_geometry_yz.set_xlabel("Longitudinal position (in)")
    axis_geometry_yz.set_ylabel("Height (in)")
    axis_geometry_yz.set_title("Front suspension - fixed chassis (y-z view)")
    axis_geometry_yz.legend(loc="best")
    axis_geometry_yz.set_xlim(
        left=min(-15.0, axis_geometry_yz.get_xlim()[0]),
        right=max(15.0, axis_geometry_yz.get_xlim()[1]),
    )
    axis_geometry_contact.set_aspect("equal", adjustable="box")
    axis_geometry_contact.set_xlabel("Lateral position (in)")
    axis_geometry_contact.set_ylabel("Height above contact patch (in)")
    axis_geometry_contact.set_title("Front suspension - fixed ground (x-z view)")
    axis_geometry_contact.legend(loc="best")
    axis_geometry_contact_yz.set_aspect("equal", adjustable="box")
    axis_geometry_contact_yz.set_xlabel("Longitudinal position (in)")
    axis_geometry_contact_yz.set_ylabel("Height above ground (in)")
    axis_geometry_contact_yz.set_title("Front suspension - fixed ground (y-z view)")
    axis_geometry_contact_yz.legend(loc="best")
    axis_geometry_contact_yz.set_xlim(
        left=min(-15.0, axis_geometry_contact_yz.get_xlim()[0]),
        right=max(15.0, axis_geometry_contact_yz.get_xlim()[1]),
    )
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
    axis_roll_center.plot(
        ride_heights_in,
        roll_center_heights_above_ground_in,
        color="tab:blue",
        zorder=1,
    )
    axis_roll_center.set_xlabel("Ride height (in)")
    axis_roll_center.set_ylabel("Roll-center height above ground (in)")
    axis_roll_center.set_title("Roll-center height above ground versus ride height")
    axis_roll_center.grid()
    axis_future.plot(
        ride_heights_in,
        np.rad2deg(toe_angles_rad),
        color="tab:blue",
        label="Straight ahead",
        zorder=1,
    )
    axis_future.plot(
        ride_heights_in,
        np.rad2deg(toe_angles_right_turn_rad),
        color="tab:orange",
        label="Right turn",
        zorder=1,
    )
    axis_future.plot(
        ride_heights_in,
        np.rad2deg(toe_angles_left_turn_rad),
        color="tab:green",
        label="Left turn",
        zorder=1,
    )
    axis_future.set_xlabel("Ride height (in)")
    axis_future.set_ylabel("Toe angle (degrees)")
    axis_future.set_title("Bump steer: toe versus ride height")
    axis_future.grid()
    axis_future.legend(loc="best")
    geometry_figure.savefig(geometry_output_path, dpi=200)
    ride_height_figure.savefig(ride_height_output_path, dpi=200)

    plt.close(geometry_figure)
    plt.close(ride_height_figure)


def geometry_is_valid(
    geometry: Geometry,
    lower_angle_rad: float,
    toe_angle_rad: float,
) -> bool:
    """Return whether all modeled points are finite at a lower-arm angle.

    Args:
        geometry: Fixed dimensions for the modeled suspension side.
        lower_angle_rad: Lower control-arm angle to evaluate in radians.
        toe_angle_rad: Toe angle to evaluate in radians.

    Returns:
        Whether the suspension geometry can be evaluated at the supplied angle.

    """
    try:
        with np.errstate(all="raise"):
            geometry_points = (
                geometry.lower_pickup(),
                geometry.upper_pickup(),
                geometry.lower_trunnion(lower_angle_rad),
                geometry.upper_trunnion(lower_angle_rad),
                geometry.instant_center(lower_angle_rad),
                geometry.roll_center(lower_angle_rad, toe_angle_rad),
                geometry.axle(lower_angle_rad),
                geometry.wheel_center(lower_angle_rad, toe_angle_rad),
                geometry.wheel_contact(lower_angle_rad, toe_angle_rad),
                geometry.tie_rod_point(lower_angle_rad, toe_angle_rad),
                geometry.steering_rod_link_point(lower_angle_rad, toe_angle_rad),
            )
    except FloatingPointError:
        return False

    return bool(
        np.all(
            np.isfinite(
                [
                    coordinate
                    for geometry_point in geometry_points
                    for coordinate in geometry_point
                ],
            ),
        ),
    )


def toe_angle_for_fixed_steering_rod_point(
    geometry: Geometry,
    lower_angle_rad: float,
    steering_rod_link_point_x_in: float,
    toe_angle_initial_rad: float,
) -> float:
    """Return toe that keeps the steering-rod link point at a target x value.

    Args:
        geometry: Fixed dimensions for the modeled suspension side.
        lower_angle_rad: Lower control-arm angle to evaluate in radians.
        steering_rod_link_point_x_in: Fixed steering-rod link-point x coordinate.
        toe_angle_initial_rad: Initial toe angle supplied to the scalar solver.

    Returns:
        Toe angle in radians that maintains the steering-rod link point.

    Raises:
        ValueError: If the scalar toe solve does not converge.

    """
    toe_angle_solution = root_scalar(
        lambda toe_angle_rad: geometry.steering_rod_link_point(
            lower_angle_rad,
            toe_angle_rad,
        ).x
        - steering_rod_link_point_x_in,
        method="secant",
        x0=toe_angle_initial_rad,
        x1=toe_angle_initial_rad + float(np.deg2rad(0.1)),
        xtol=1e-10,
    )
    if not toe_angle_solution.converged:
        message = "Toe calculation did not converge for the steering-rod link point."
        raise ValueError(message)

    return float(toe_angle_solution.root)


def main() -> int:
    """Create the front-suspension x-z plot.

    Returns:
        Zero after saving the plot.

    """
    geometry = Geometry()
    valid_lower_angles_rad = [
        float(lower_angle_rad)
        for lower_angle_rad in np.linspace(-np.pi / 4.0, np.pi / 4.0, 301)
        if geometry_is_valid(geometry, float(lower_angle_rad), 0.0)
    ]
    ride_height_rest_in = 7.0
    ride_height_max_bump_in = ride_height_rest_in - 2.0
    ride_height_max_drop_in = ride_height_rest_in + 3.0
    ride_heights_in = [
        -geometry.wheel_contact(lower_angle_rad, 0.0).z
        for lower_angle_rad in valid_lower_angles_rad
    ]
    ride_height_order = np.argsort(ride_heights_in)
    sorted_ride_heights_in = np.asarray(ride_heights_in)[ride_height_order]
    sorted_lower_angles_rad = np.asarray(valid_lower_angles_rad)[ride_height_order]
    lower_angle_max_bump_rad = float(
        np.interp(
            ride_height_max_bump_in,
            sorted_ride_heights_in,
            sorted_lower_angles_rad,
        ),
    )
    lower_angle_rest_rad = float(
        np.interp(
            ride_height_rest_in,
            sorted_ride_heights_in,
            sorted_lower_angles_rad,
        ),
    )
    lower_angle_max_drop_rad = float(
        np.interp(
            ride_height_max_drop_in,
            sorted_ride_heights_in,
            sorted_lower_angles_rad,
        ),
    )
    toe_angle_rest_rad = float(np.deg2rad(-3.0))
    steering_rod_link_point_rest_x_in = geometry.steering_rod_link_point(
        lower_angle_rest_rad,
        toe_angle_rest_rad,
    ).x
    steering_rod_length_rest_in = 2.0 * steering_rod_link_point_rest_x_in
    print(f"Resting steering rod length: {steering_rod_length_rest_in:.3f} in")
    toe_angle_right_turn_rest_rad = toe_angle_rest_rad + float(np.deg2rad(5.0))
    toe_angle_left_turn_rest_rad = toe_angle_rest_rad - float(np.deg2rad(5.0))
    toe_angle_max_bump_rad = toe_angle_for_fixed_steering_rod_point(
        geometry,
        lower_angle_max_bump_rad,
        steering_rod_link_point_rest_x_in,
        toe_angle_rest_rad,
    )
    toe_angle_max_drop_rad = toe_angle_for_fixed_steering_rod_point(
        geometry,
        lower_angle_max_drop_rad,
        steering_rod_link_point_rest_x_in,
        toe_angle_rest_rad,
    )
    lower_angles_rad = [
        lower_angle_rad
        for lower_angle_rad in valid_lower_angles_rad
        if ride_height_max_bump_in
        <= -geometry.wheel_contact(lower_angle_rad, 0.0).z
        <= ride_height_max_drop_in
    ]
    toe_angles_rad = [
        toe_angle_for_fixed_steering_rod_point(
            geometry,
            lower_angle_rad,
            steering_rod_link_point_rest_x_in,
            toe_angle_rest_rad,
        )
        for lower_angle_rad in lower_angles_rad
    ]
    steering_rod_link_point_right_turn_rest_x_in = geometry.steering_rod_link_point(
        lower_angle_rest_rad,
        toe_angle_right_turn_rest_rad,
    ).x
    toe_angles_right_turn_rad = [
        toe_angle_for_fixed_steering_rod_point(
            geometry,
            lower_angle_rad,
            steering_rod_link_point_right_turn_rest_x_in,
            toe_angle_right_turn_rest_rad,
        )
        for lower_angle_rad in lower_angles_rad
    ]
    steering_rod_link_point_left_turn_rest_x_in = geometry.steering_rod_link_point(
        lower_angle_rest_rad,
        toe_angle_left_turn_rest_rad,
    ).x
    toe_angles_left_turn_rad = [
        toe_angle_for_fixed_steering_rod_point(
            geometry,
            lower_angle_rad,
            steering_rod_link_point_left_turn_rest_x_in,
            toe_angle_left_turn_rest_rad,
        )
        for lower_angle_rad in lower_angles_rad
    ]
    plot_xz(
        geometry,
        lower_angles_rad,
        toe_angles_rad,
        toe_angles_right_turn_rad,
        toe_angles_left_turn_rad,
        lower_angle_max_bump_rad,
        toe_angle_max_bump_rad,
        lower_angle_rest_rad,
        toe_angle_rest_rad,
        lower_angle_max_drop_rad,
        toe_angle_max_drop_rad,
        Path("front_suspension_geometries.png"),
        Path("front_suspension_ride_height.png"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
