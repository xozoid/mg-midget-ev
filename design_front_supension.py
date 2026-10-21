"""Plot front-suspension pickup and trunnion locations in the x-z plane."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style
from matplotlib.patches import Ellipse
from scipy.optimize import root_scalar

from mg.front_suspension import Geometry, Point

if TYPE_CHECKING:
    from collections.abc import Sequence


TARGET_STATIC_CAMBER_DEG = -0.5
STATIC_CAMBER_TOLERANCE_DEG = 0.25
TARGET_MAX_BUMP_CAMBER_DEG = 0.0
TARGET_MAX_DROOP_CAMBER_DEG = 0.5
TARGET_MAX_TOE_PEAK_TO_PEAK_DEG = 1.0
TARGET_MAX_ROLL_CENTER_PEAK_TO_PEAK_IN = 2.0
TARGET_WHEEL_CENTER_REST_X_IN = 22.984638682953
WHEEL_CENTER_X_TOLERANCE_IN = 0.25
CAMBER_ERROR_REFERENCE_DEG = 0.5


@dataclass(frozen=True)
class SuspensionTargetReport:
    """Measured suspension targets, their constraint violations, and score."""

    static_camber_deg: float
    static_camber_distance_deg: float
    max_bump_camber_deg: float
    bump_camber_distance_deg: float
    max_droop_camber_deg: float
    droop_camber_distance_deg: float
    straight_toe_peak_to_peak_deg: float
    right_turn_toe_peak_to_peak_deg: float
    left_turn_toe_peak_to_peak_deg: float
    max_toe_peak_to_peak_deg: float
    toe_peak_to_peak_distance_deg: float
    roll_center_peak_to_peak_in: float
    roll_center_peak_to_peak_distance_in: float
    wheel_center_rest_x_in: float
    wheel_center_x_distance_in: float
    aggregate_score_pct: float


def _values_between_travel_limits(
    travel_in: Sequence[float],
    values: Sequence[float],
    travel_min_in: float,
    travel_max_in: float,
) -> np.ndarray:
    """Return values in a travel interval, including interpolated endpoints.

    Args:
        travel_in: Suspension travel relative to rest, in inches.
        values: Values sampled at ``travel_in``.
        travel_min_in: Inclusive lower travel boundary, in inches.
        travel_max_in: Inclusive upper travel boundary, in inches.

    Returns:
        Values in the requested interval.

    Raises:
        ValueError: If the samples do not cover the requested travel interval.

    """
    sorted_indices = np.argsort(travel_in)
    sorted_travel_in = np.asarray(travel_in)[sorted_indices]
    sorted_values = np.asarray(values)[sorted_indices]
    if sorted_travel_in[0] > travel_min_in or sorted_travel_in[-1] < travel_max_in:
        message = "Suspension samples do not cover the requested travel interval."
        raise ValueError(message)

    interval_values = sorted_values[
        (sorted_travel_in >= travel_min_in) & (sorted_travel_in <= travel_max_in)
    ]
    return np.concatenate(
        (
            [np.interp(travel_min_in, sorted_travel_in, sorted_values)],
            interval_values,
            [np.interp(travel_max_in, sorted_travel_in, sorted_values)],
        ),
    )


def evaluate_suspension_targets(
    travel_in: Sequence[float],
    camber_deg: Sequence[float],
    straight_toe_deg: Sequence[float],
    right_turn_toe_deg: Sequence[float],
    left_turn_toe_deg: Sequence[float],
    roll_center_heights_in: Sequence[float],
    wheel_center_rest_x_in: float,
) -> SuspensionTargetReport:
    """Evaluate the front-suspension camber and bump-steer targets.

    Positive travel is bump and negative travel is droop. Constraint distances
    are zero when their target is met and otherwise report the amount by which
    the constraint is exceeded.

    Args:
        travel_in: Suspension travel relative to rest, in inches.
        camber_deg: Camber angles sampled at ``travel_in``.
        straight_toe_deg: Straight-ahead toe angles sampled at ``travel_in``.
        right_turn_toe_deg: Right-turn toe angles sampled at ``travel_in``.
        left_turn_toe_deg: Left-turn toe angles sampled at ``travel_in``.
        roll_center_heights_in: Roll-center heights above ground at ``travel_in``.
        wheel_center_rest_x_in: Wheel-center lateral position at rest, in inches.

    Returns:
        Measured targets, constraint distances, and an aggregate score.

    """
    travel_order = np.argsort(travel_in)
    static_camber_deg = float(
        np.interp(
            0.0,
            np.asarray(travel_in)[travel_order],
            np.asarray(camber_deg)[travel_order],
        ),
    )
    bump_camber_deg = _values_between_travel_limits(
        travel_in,
        camber_deg,
        0.0,
        2.0,
    )
    droop_camber_deg = _values_between_travel_limits(
        travel_in,
        camber_deg,
        -2.0,
        0.0,
    )
    max_bump_camber_deg = float(np.max(bump_camber_deg))
    max_droop_camber_deg = float(np.max(droop_camber_deg))
    straight_toe_peak_to_peak_deg = float(np.ptp(straight_toe_deg))
    right_turn_toe_peak_to_peak_deg = float(np.ptp(right_turn_toe_deg))
    left_turn_toe_peak_to_peak_deg = float(np.ptp(left_turn_toe_deg))
    max_toe_peak_to_peak_deg = max(
        straight_toe_peak_to_peak_deg,
        right_turn_toe_peak_to_peak_deg,
        left_turn_toe_peak_to_peak_deg,
    )
    static_camber_distance_deg = max(
        0.0,
        abs(static_camber_deg - TARGET_STATIC_CAMBER_DEG) - STATIC_CAMBER_TOLERANCE_DEG,
    )
    bump_camber_distance_deg = max(
        0.0,
        max_bump_camber_deg - TARGET_MAX_BUMP_CAMBER_DEG,
    )
    droop_camber_distance_deg = max(
        0.0,
        max_droop_camber_deg - TARGET_MAX_DROOP_CAMBER_DEG,
    )
    toe_peak_to_peak_distance_deg = max(
        0.0,
        max_toe_peak_to_peak_deg - TARGET_MAX_TOE_PEAK_TO_PEAK_DEG,
    )
    roll_center_peak_to_peak_in = float(np.ptp(roll_center_heights_in))
    roll_center_peak_to_peak_distance_in = max(
        0.0,
        roll_center_peak_to_peak_in - TARGET_MAX_ROLL_CENTER_PEAK_TO_PEAK_IN,
    )
    wheel_center_x_distance_in = abs(
        wheel_center_rest_x_in - TARGET_WHEEL_CENTER_REST_X_IN,
    )
    wheel_center_x_excess_in = max(
        0.0,
        wheel_center_x_distance_in - WHEEL_CENTER_X_TOLERANCE_IN,
    )
    normalized_error = (
        static_camber_distance_deg / CAMBER_ERROR_REFERENCE_DEG
        + bump_camber_distance_deg / CAMBER_ERROR_REFERENCE_DEG
        + droop_camber_distance_deg / CAMBER_ERROR_REFERENCE_DEG
        + toe_peak_to_peak_distance_deg / TARGET_MAX_TOE_PEAK_TO_PEAK_DEG
        + roll_center_peak_to_peak_distance_in / TARGET_MAX_ROLL_CENTER_PEAK_TO_PEAK_IN
        + wheel_center_x_excess_in / WHEEL_CENTER_X_TOLERANCE_IN
    ) / 6.0

    return SuspensionTargetReport(
        static_camber_deg=static_camber_deg,
        static_camber_distance_deg=static_camber_distance_deg,
        max_bump_camber_deg=max_bump_camber_deg,
        bump_camber_distance_deg=bump_camber_distance_deg,
        max_droop_camber_deg=max_droop_camber_deg,
        droop_camber_distance_deg=droop_camber_distance_deg,
        straight_toe_peak_to_peak_deg=straight_toe_peak_to_peak_deg,
        right_turn_toe_peak_to_peak_deg=right_turn_toe_peak_to_peak_deg,
        left_turn_toe_peak_to_peak_deg=left_turn_toe_peak_to_peak_deg,
        max_toe_peak_to_peak_deg=max_toe_peak_to_peak_deg,
        toe_peak_to_peak_distance_deg=toe_peak_to_peak_distance_deg,
        roll_center_peak_to_peak_in=roll_center_peak_to_peak_in,
        roll_center_peak_to_peak_distance_in=roll_center_peak_to_peak_distance_in,
        wheel_center_rest_x_in=wheel_center_rest_x_in,
        wheel_center_x_distance_in=wheel_center_x_distance_in,
        aggregate_score_pct=100.0 / (1.0 + normalized_error),
    )


def print_suspension_target_report(report: SuspensionTargetReport) -> None:
    """Print a color-coded, compact target report for one design."""
    static_camber_passed = np.isclose(report.static_camber_distance_deg, 0.0)
    bump_camber_passed = report.max_bump_camber_deg < TARGET_MAX_BUMP_CAMBER_DEG
    droop_camber_passed = report.max_droop_camber_deg < TARGET_MAX_DROOP_CAMBER_DEG
    toe_passed = report.max_toe_peak_to_peak_deg < TARGET_MAX_TOE_PEAK_TO_PEAK_DEG
    roll_center_passed = (
        report.roll_center_peak_to_peak_in < TARGET_MAX_ROLL_CENTER_PEAK_TO_PEAK_IN
    )
    wheel_center_passed = (
        report.wheel_center_x_distance_in <= WHEEL_CENTER_X_TOLERANCE_IN
    )
    design_passed = all(
        (
            static_camber_passed,
            bump_camber_passed,
            droop_camber_passed,
            toe_passed,
            roll_center_passed,
            wheel_center_passed,
        ),
    )

    def status_line(*, passed: bool, text: str) -> str:
        """Return a red or green status line for terminal output.

        Returns:
            A terminal line with a colorama status color.

        """
        color = Fore.GREEN if passed else Fore.RED
        label = "PASS" if passed else "FAIL"
        return f"{color}[{label}] {text}{Style.RESET_ALL}"

    print("\n=== FRONT SUSPENSION ===")
    print(
        status_line(
            passed=static_camber_passed,
            text="Static camber "
            f"{report.static_camber_deg:+.1f} deg "
            f"(target {TARGET_STATIC_CAMBER_DEG:+.1f} +/- "
            f"{STATIC_CAMBER_TOLERANCE_DEG:.2f} deg, "
            f"off {report.static_camber_distance_deg:.1f} deg)",
        ),
    )
    print(
        status_line(
            passed=bump_camber_passed,
            text="Bump camber "
            f"{report.max_bump_camber_deg:+.1f} deg through +2.0 in "
            f"(must be negative, off {report.bump_camber_distance_deg:.1f} deg)",
        ),
    )
    print(
        status_line(
            passed=droop_camber_passed,
            text="Droop camber "
            f"{report.max_droop_camber_deg:+.1f} deg through -2.0 in "
            f"(limit {TARGET_MAX_DROOP_CAMBER_DEG:+.1f} deg, "
            f"off {report.droop_camber_distance_deg:.1f} deg)",
        ),
    )
    print(
        status_line(
            passed=toe_passed,
            text="Toe P2P "
            f"straight {report.straight_toe_peak_to_peak_deg:.1f} deg, "
            f"right {report.right_turn_toe_peak_to_peak_deg:.1f} deg, "
            f"left {report.left_turn_toe_peak_to_peak_deg:.1f} deg "
            f"(limit {TARGET_MAX_TOE_PEAK_TO_PEAK_DEG:.1f} deg, "
            f"off {report.toe_peak_to_peak_distance_deg:.1f} deg)",
        ),
    )
    print(
        status_line(
            passed=roll_center_passed,
            text="Roll-center P2P "
            f"{report.roll_center_peak_to_peak_in:.1f} in "
            f"(limit {TARGET_MAX_ROLL_CENTER_PEAK_TO_PEAK_IN:.1f} in, "
            f"off {report.roll_center_peak_to_peak_distance_in:.1f} in)",
        ),
    )
    print(
        status_line(
            passed=wheel_center_passed,
            text="Wheel-center X "
            f"{report.wheel_center_rest_x_in:.1f} in "
            f"(default target {TARGET_WHEEL_CENTER_REST_X_IN:.1f} in, "
            f"off {report.wheel_center_x_distance_in:.1f} in)",
        ),
    )
    print(
        status_line(
            passed=design_passed,
            text=f"SCORE {report.aggregate_score_pct:.1f}%",
        ),
    )


def plot_four_bar_and_kingpin_axis(
    geometry: Geometry,
    lower_angles_rad: Sequence[float],
    geometry_output_path: Path,
) -> None:
    """Plot the four-bar linkage and kingpin axis over an arm-angle sweep.

    Args:
        geometry: Fixed dimensions for the modeled suspension side.
        lower_angles_rad: Lower control-arm angles to display in radians.
        geometry_output_path: PNG file for the geometry view.

    """
    figure, axis = plt.subplots(figsize=(12.0, 9.0), layout="constrained")
    color_map = plt.colormaps["coolwarm"]
    lower_pickup_point = geometry.lower_pickup()
    upper_pickup_point = geometry.upper_pickup()
    lower_angle_limit_deg = max(
        abs(float(np.rad2deg(lower_angle_rad))) for lower_angle_rad in lower_angles_rad
    )

    for lower_angle_rad in lower_angles_rad:
        lower_trunnion_point = geometry.lower_trunnion(lower_angle_rad)
        upper_trunnion_point = geometry.upper_trunnion(lower_angle_rad)
        lower_kingpin_point, upper_kingpin_point = geometry.kingpin_axis(
            lower_angle_rad,
        )
        axle_point = geometry.axle(lower_angle_rad)
        axle_direction = geometry.axle_direction(lower_angle_rad)
        wheel_center_point = axle_point + Point(
            geometry.axle_length_in * axle_direction.x,
            geometry.axle_length_in * axle_direction.y,
            geometry.axle_length_in * axle_direction.z,
        )
        lower_angle_deg = float(np.rad2deg(lower_angle_rad))
        color = color_map(
            (lower_angle_deg + lower_angle_limit_deg) / (2.0 * lower_angle_limit_deg),
        )
        four_bar_points = (
            lower_pickup_point,
            upper_pickup_point,
            upper_trunnion_point,
            lower_trunnion_point,
            lower_pickup_point,
        )
        axis.plot(
            [point.x for point in four_bar_points],
            [point.z for point in four_bar_points],
            color=color,
            linewidth=1.5,
            label=f"{lower_angle_deg:+.0f} degrees",
        )
        axis.plot(
            [lower_kingpin_point.x, upper_kingpin_point.x],
            [lower_kingpin_point.z, upper_kingpin_point.z],
            color=color,
            linestyle="--",
            linewidth=2.0,
        )
        axis.scatter(
            [axle_point.x],
            [axle_point.z],
            color="black",
            marker="x",
            zorder=3,
        )
        axis.plot(
            [axle_point.x, wheel_center_point.x],
            [axle_point.z, wheel_center_point.z],
            color="black",
            linestyle=":",
            linewidth=1.5,
        )

    axis.scatter(
        [lower_pickup_point.x, upper_pickup_point.x],
        [lower_pickup_point.z, upper_pickup_point.z],
        color="black",
        label="Chassis pickups",
        zorder=4,
    )
    axis.plot([], [], color="black", linestyle="--", label="Kingpin axis")
    axis.scatter([], [], color="black", marker="x", label="Axle rotation point")
    axis.plot([], [], color="black", linestyle=":", label="Axle droop")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Lateral position (in)")
    axis.set_ylabel("Height (in)")
    axis.set_title("Front four-bar and kingpin axis: lower arm -5 to +5 degrees")
    axis.grid()
    axis.legend(loc="best", title="Lower arm")
    figure.savefig(geometry_output_path, dpi=200)
    plt.close(figure)


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
    ride_height_rest_in = -geometry.wheel_contact(
        lower_angle_rest_rad,
        toe_angle_rest_rad,
    ).z
    ride_height_changes_from_rest_in = [
        ride_height_rest_in - ride_height_in for ride_height_in in ride_heights_in
    ]
    wheel_camber_deg = [
        float(
            -np.rad2deg(
                np.arctan2(
                    geometry.wheel_axis(lower_angle_rad, toe_angle_rad).z,
                    geometry.wheel_axis(lower_angle_rad, toe_angle_rad).x,
                ),
            ),
        )
        for lower_angle_rad, toe_angle_rad in zip(
            lower_angles_rad,
            toe_angles_rad,
            strict=True,
        )
    ]
    roll_center_heights_above_ground_in = [
        roll_center_point.z - wheel_contact_point.z
        for roll_center_point, wheel_contact_point in zip(
            roll_center_points,
            wheel_contact_points,
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
    wheel_contact_rest_x_in = geometry.wheel_contact(
        lower_angle_rest_rad,
        toe_angle_rest_rad,
    ).x
    for lower_angle_rad, toe_angle_rad, state_name, state_color in state_angles:
        lower_trunnion_point = geometry.lower_trunnion(lower_angle_rad)
        upper_trunnion_point = geometry.upper_trunnion(lower_angle_rad)
        lower_kingpin_point, upper_kingpin_point = geometry.kingpin_axis(
            lower_angle_rad,
        )
        instant_center_point = geometry.instant_center(lower_angle_rad)
        axle_point = geometry.axle(lower_angle_rad)
        wheel_axis_vector = geometry.wheel_axis(
            lower_angle_rad,
            toe_angle_rad,
        )
        wheel_ellipse_xz_width_in = (
            2.0 * geometry.wheel_radius_in * abs(wheel_axis_vector.y)
        )
        wheel_ellipse_yz_width_in = (
            2.0 * geometry.wheel_radius_in * abs(wheel_axis_vector.x)
        )
        wheel_ellipse_xz_angle_deg = float(
            np.rad2deg(np.arctan2(wheel_axis_vector.z, wheel_axis_vector.x)),
        )
        wheel_ellipse_yz_angle_deg = float(
            np.rad2deg(np.arctan2(wheel_axis_vector.z, wheel_axis_vector.y)),
        )
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
        axis_geometry.plot(
            [lower_kingpin_point.x, upper_kingpin_point.x],
            [lower_kingpin_point.z, upper_kingpin_point.z],
            color="black",
            linewidth=2.0,
            label="Kingpin axis" if state_name == "Maximum bump" else None,
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
            color="black",
            linestyle="--",
            linewidth=1.5,
            zorder=2,
        )
        axis_geometry.add_patch(
            Ellipse(
                (wheel_center_point.x, wheel_center_point.z),
                width=wheel_ellipse_xz_width_in,
                height=2.0 * geometry.wheel_radius_in,
                angle=wheel_ellipse_xz_angle_deg,
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
        axis_geometry_contact.plot(
            [lower_kingpin_point.x, upper_kingpin_point.x],
            [
                lower_kingpin_point.z - wheel_contact_point.z,
                upper_kingpin_point.z - wheel_contact_point.z,
            ],
            color="black",
            linewidth=2.0,
            label="Kingpin axis" if state_name == "Maximum bump" else None,
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
            color="black",
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
                angle=wheel_ellipse_xz_angle_deg,
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
                angle=wheel_ellipse_yz_angle_deg,
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
                    wheel_center_point.z - wheel_contact_point.z,
                ),
                width=wheel_ellipse_yz_width_in,
                height=2.0 * geometry.wheel_radius_in,
                angle=wheel_ellipse_yz_angle_deg,
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
                roll_center_point.z - wheel_contact_point.z,
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
            ride_height_rest_in + wheel_contact_point.z,
            float(
                -np.rad2deg(
                    np.arctan2(
                        geometry.wheel_axis(lower_angle_rad, toe_angle_rad).z,
                        geometry.wheel_axis(lower_angle_rad, toe_angle_rad).x,
                    ),
                ),
            ),
            color=state_color,
            zorder=2,
        )
        axis_wheel_contact.scatter(
            ride_height_rest_in + wheel_contact_point.z,
            wheel_contact_point.x - wheel_contact_rest_x_in,
            color=state_color,
            zorder=2,
        )
        axis_roll_center.scatter(
            ride_height_rest_in + wheel_contact_point.z,
            roll_center_point.z - wheel_center_point.z + geometry.wheel_radius_in,
            color=state_color,
            zorder=2,
        )
        axis_future.scatter(
            ride_height_rest_in + wheel_contact_point.z,
            float(np.rad2deg(toe_angle_rad)),
            color=state_color,
            zorder=2,
        )

    xz_view_points = [
        Point(0.0, 0.0, 0.0),
        lower_pickup_point,
        upper_pickup_point,
        *lower_trunnion_points,
        *upper_trunnion_points,
        *wheel_center_points,
        *wheel_contact_points,
        *tie_rod_points,
        *steering_rod_link_points,
    ]
    view_margin_in = 2.0
    x_view_min_in = min(point.x for point in xz_view_points) - view_margin_in
    x_view_max_in = max(point.x for point in xz_view_points) + view_margin_in
    z_view_min_in = (
        min(
            [point.z for point in xz_view_points]
            + [point.z - geometry.wheel_radius_in for point in wheel_center_points],
        )
        - view_margin_in
    )
    z_view_max_in = (
        max(
            [point.z for point in xz_view_points]
            + [point.z + geometry.wheel_radius_in for point in wheel_center_points],
        )
        + view_margin_in
    )

    axis_geometry.set_aspect("equal", adjustable="box")
    axis_geometry.set_xlim(left=x_view_min_in, right=x_view_max_in)
    axis_geometry.set_ylim(bottom=z_view_min_in, top=z_view_max_in)
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
    axis_geometry_contact.set_xlim(left=x_view_min_in, right=x_view_max_in)
    axis_geometry_contact.set_ylim(
        bottom=-view_margin_in,
        top=2.0 * geometry.wheel_radius_in + view_margin_in,
    )
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
        ride_height_changes_from_rest_in,
        wheel_camber_deg,
        color="tab:blue",
        zorder=1,
    )
    axis_camber.set_xlabel("Ride-height change from rest (in)")
    axis_camber.set_ylabel("Camber (degrees)")
    axis_camber.set_title("Camber versus ride-height change")
    axis_camber.grid()
    axis_wheel_contact.plot(
        ride_height_changes_from_rest_in,
        [
            wheel_contact_point.x - wheel_contact_rest_x_in
            for wheel_contact_point in wheel_contact_points
        ],
        color="tab:purple",
        zorder=1,
    )
    axis_wheel_contact.set_xlabel("Ride-height change from rest (in)")
    axis_wheel_contact.set_ylabel("Contact-patch lateral displacement from rest (in)")
    axis_wheel_contact.set_title(
        "Wheel contact lateral position versus ride-height change",
    )
    axis_wheel_contact.grid()
    axis_roll_center.plot(
        ride_height_changes_from_rest_in,
        roll_center_heights_above_ground_in,
        color="tab:blue",
        zorder=1,
    )
    axis_roll_center.set_xlabel("Ride-height change from rest (in)")
    axis_roll_center.set_ylabel("Roll-center height above ground (in)")
    axis_roll_center.set_title(
        "Roll-center height above ground versus ride-height change",
    )
    axis_roll_center.grid()
    axis_future.plot(
        ride_height_changes_from_rest_in,
        np.rad2deg(toe_angles_rad),
        color="tab:blue",
        label="Straight ahead",
        zorder=1,
    )
    axis_future.plot(
        ride_height_changes_from_rest_in,
        np.rad2deg(toe_angles_right_turn_rad),
        color="tab:orange",
        label="Right turn",
        zorder=1,
    )
    axis_future.plot(
        ride_height_changes_from_rest_in,
        np.rad2deg(toe_angles_left_turn_rad),
        color="tab:green",
        label="Left turn",
        zorder=1,
    )
    axis_future.set_xlabel("Ride-height change from rest (in)")
    axis_future.set_ylabel("Toe angle (degrees)")
    axis_future.set_title("Bump steer: toe versus ride-height change")
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
    """Create the focused front-suspension four-bar plot.

    Returns:
        Zero after saving the plot.

    """
    geometry = Geometry(
        lower_in=11.21813684,
        upper_in=6.85853975,
        inner_in=9.3297941,
        inner_angle_rad=float(np.deg2rad(-31.55241894)),
        steering_rod_y_in=-4.74654607,
        steering_rod_z_in=5.41529298,
    )
    lower_angles_rad = np.deg2rad(np.linspace(-5.0, 5.0, 11))
    plot_four_bar_and_kingpin_axis(
        geometry,
        lower_angles_rad,
        Path("front_suspension_kingpin_axis.png"),
    )
    valid_lower_angles_rad = [
        float(lower_angle_rad)
        for lower_angle_rad in np.linspace(-np.pi / 4.0, np.pi / 4.0, 301)
        if geometry_is_valid(geometry, float(lower_angle_rad), 0.0)
    ]
    ride_height_rest_in = 8
    ride_height_max_bump_in = ride_height_rest_in - 2.0
    ride_height_max_drop_in = ride_height_rest_in + 2.0
    toe_angle_rest_rad = float(np.deg2rad(-0.08))
    ride_heights_in = [
        -geometry.wheel_contact(lower_angle_rad, toe_angle_rest_rad).z
        for lower_angle_rad in valid_lower_angles_rad
    ]
    ride_height_order = np.argsort(ride_heights_in)
    sorted_ride_heights_in = np.asarray(ride_heights_in)[ride_height_order]
    sorted_lower_angles_rad = np.asarray(valid_lower_angles_rad)[ride_height_order]
    lower_angle_rest_rad = float(
        np.interp(
            ride_height_rest_in,
            sorted_ride_heights_in,
            sorted_lower_angles_rad,
        ),
    )
    steering_rod_link_point_rest_x_in = geometry.steering_rod_link_point(
        lower_angle_rest_rad,
        toe_angle_rest_rad,
    ).x
    steering_rod_length_rest_in = 2.0 * steering_rod_link_point_rest_x_in
    print(f"Resting steering rod length: {steering_rod_length_rest_in:.1f} in")
    toe_angle_right_turn_rest_rad = toe_angle_rest_rad + float(np.deg2rad(5.0))
    toe_angle_left_turn_rest_rad = toe_angle_rest_rad - float(np.deg2rad(5.0))
    constrained_lower_angles_rad = []
    constrained_toe_angles_rad = []
    constrained_ride_heights_in = []
    for lower_angle_rad in valid_lower_angles_rad:
        try:
            toe_angle_rad = toe_angle_for_fixed_steering_rod_point(
                geometry,
                lower_angle_rad,
                steering_rod_link_point_rest_x_in,
                toe_angle_rest_rad,
            )
        except (FloatingPointError, ValueError):
            continue
        constrained_lower_angles_rad.append(lower_angle_rad)
        constrained_toe_angles_rad.append(toe_angle_rad)
        constrained_ride_heights_in.append(
            -geometry.wheel_contact(lower_angle_rad, toe_angle_rad).z,
        )
    constrained_ride_height_order = np.argsort(constrained_ride_heights_in)
    sorted_constrained_ride_heights_in = np.asarray(constrained_ride_heights_in)[
        constrained_ride_height_order
    ]
    sorted_constrained_lower_angles_rad = np.asarray(constrained_lower_angles_rad)[
        constrained_ride_height_order
    ]
    lower_angle_max_bump_rad = float(
        np.interp(
            ride_height_max_bump_in,
            sorted_constrained_ride_heights_in,
            sorted_constrained_lower_angles_rad,
        ),
    )
    lower_angle_max_drop_rad = float(
        np.interp(
            ride_height_max_drop_in,
            sorted_constrained_ride_heights_in,
            sorted_constrained_lower_angles_rad,
        ),
    )
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
        for lower_angle_rad, ride_height_in in zip(
            constrained_lower_angles_rad,
            constrained_ride_heights_in,
            strict=True,
        )
        if ride_height_max_bump_in <= ride_height_in <= ride_height_max_drop_in
    ]
    toe_angles_rad = [
        toe_angle_rad
        for toe_angle_rad, ride_height_in in zip(
            constrained_toe_angles_rad,
            constrained_ride_heights_in,
            strict=True,
        )
        if ride_height_max_bump_in <= ride_height_in <= ride_height_max_drop_in
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
    toe_angle_right_turn_max_bump_rad = toe_angle_for_fixed_steering_rod_point(
        geometry,
        lower_angle_max_bump_rad,
        steering_rod_link_point_right_turn_rest_x_in,
        toe_angle_right_turn_rest_rad,
    )
    toe_angle_right_turn_max_drop_rad = toe_angle_for_fixed_steering_rod_point(
        geometry,
        lower_angle_max_drop_rad,
        steering_rod_link_point_right_turn_rest_x_in,
        toe_angle_right_turn_rest_rad,
    )
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
    toe_angle_left_turn_max_bump_rad = toe_angle_for_fixed_steering_rod_point(
        geometry,
        lower_angle_max_bump_rad,
        steering_rod_link_point_left_turn_rest_x_in,
        toe_angle_left_turn_rest_rad,
    )
    toe_angle_left_turn_max_drop_rad = toe_angle_for_fixed_steering_rod_point(
        geometry,
        lower_angle_max_drop_rad,
        steering_rod_link_point_left_turn_rest_x_in,
        toe_angle_left_turn_rest_rad,
    )
    travel_in = [
        -2.0,
        *(
            ride_height_rest_in - ride_height_in
            for ride_height_in in constrained_ride_heights_in
            if ride_height_max_bump_in <= ride_height_in <= ride_height_max_drop_in
        ),
        2.0,
    ]
    metric_lower_angles_rad = [
        lower_angle_max_drop_rad,
        *lower_angles_rad,
        lower_angle_max_bump_rad,
    ]
    metric_toe_angles_rad = [
        toe_angle_max_drop_rad,
        *toe_angles_rad,
        toe_angle_max_bump_rad,
    ]
    camber_deg = [
        float(
            -np.rad2deg(
                np.arctan2(
                    geometry.wheel_axis(lower_angle_rad, toe_angle_rad).z,
                    geometry.wheel_axis(lower_angle_rad, toe_angle_rad).x,
                ),
            ),
        )
        for lower_angle_rad, toe_angle_rad in zip(
            metric_lower_angles_rad,
            metric_toe_angles_rad,
            strict=True,
        )
    ]
    roll_center_heights_in = [
        geometry.roll_center(lower_angle_rad, toe_angle_rad).z
        - geometry.wheel_contact(lower_angle_rad, toe_angle_rad).z
        for lower_angle_rad, toe_angle_rad in zip(
            metric_lower_angles_rad,
            metric_toe_angles_rad,
            strict=True,
        )
    ]
    suspension_target_report = evaluate_suspension_targets(
        travel_in,
        camber_deg,
        [float(np.rad2deg(toe_angle_rad)) for toe_angle_rad in metric_toe_angles_rad],
        [
            float(np.rad2deg(toe_angle_rad))
            for toe_angle_rad in (
                toe_angle_right_turn_max_drop_rad,
                *toe_angles_right_turn_rad,
                toe_angle_right_turn_max_bump_rad,
            )
        ],
        [
            float(np.rad2deg(toe_angle_rad))
            for toe_angle_rad in (
                toe_angle_left_turn_max_drop_rad,
                *toe_angles_left_turn_rad,
                toe_angle_left_turn_max_bump_rad,
            )
        ],
        roll_center_heights_in,
        geometry.wheel_center(lower_angle_rest_rad, toe_angle_rest_rad).x,
    )
    print_suspension_target_report(suspension_target_report)
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
