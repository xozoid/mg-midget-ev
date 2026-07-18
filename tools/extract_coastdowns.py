"""Extract repeated coast-down trials from a Physics Toolbox-style experiment ZIP.

Expected files:
    Location.csv
    Pressure.csv
    Temperature.csv

Outputs:
    coastdown_trials.csv   one row per GPS sample, with trial metadata
    coastdown_summary.csv  one row per detected coast-down
    coastdown_detection.png
    trial_XX.csv            individual trial files

Example:
    python extract_coastdowns.py "My Experiment.zip" -o coastdown_output

The defaults are tuned for roughly 10 runs accelerating to 50-60 mph and
coasting down to about 18 mph. Adjust --start-mph and --end-mph if needed.

"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import cast, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

if TYPE_CHECKING:
    from os import PathLike
    from typing import Protocol

    from numpy.typing import NDArray

    class _PlotAxes(Protocol):
        """Typed subset of matplotlib axes used by this script."""

        def plot(
            self,
            *args: object,
            linewidth: float,
            label: str,
        ) -> object: ...

        def text(
            self,
            x: object,
            y: object,
            s: str,
            *,
            ha: str,
            va: str,
        ) -> object: ...

        def set_xlabel(self, xlabel: str) -> object: ...

        def set_ylabel(self, ylabel: str) -> object: ...

        def set_title(self, label: str) -> object: ...

        def grid(self, *, visible: bool, alpha: float) -> None: ...

        def get_legend_handles_labels(self) -> tuple[list[object], list[str]]: ...

        def legend(
            self,
            handles: list[object],
            labels: list[str],
            *,
            loc: str,
        ) -> object: ...

    class _PlotFigure(Protocol):
        """Typed subset of matplotlib figures used by this script."""

        def tight_layout(self) -> None: ...

        def savefig(self, fname: str | PathLike[str], *, dpi: int) -> None: ...


MPH_PER_MPS = 2.2369362920544
R_DRY_AIR = 287.058  # J/(kg K)
RESIDUAL_ACCELERATION_MPS2 = 0.10
GPS_NOISE_INCREASE_MPH = 0.5


def trim_to_deceleration_start(
    smooth_speed_mph: NDArray[np.float64],
    sample_times_s: NDArray[np.float64],
    peak: int,
) -> int:
    """Trim a peak candidate to the first decelerating sample.

    Returns:
        The index where the trial should start.

    """
    start = peak
    accel = np.gradient(smooth_speed_mph, sample_times_s)
    while (
        start < len(smooth_speed_mph) - 1 and accel[start] > RESIDUAL_ACCELERATION_MPS2
    ):
        start += 1
    return start


def find_monotonic_coastdown(
    smooth_speed_mph: NDArray[np.float64],
    sample_times_s: NDArray[np.float64],
    peak: int,
    search_stop: int,
    end_mph: float,
) -> tuple[int, int, int] | None:
    """Find a decreasing coast-down segment, re-arming after speed increases.

    Returns:
        A tuple of ``(start_index, end_index, peak_index)`` when a usable
        candidate reaches the end speed, otherwise ``None``.

    """
    candidate_peak = peak
    start = trim_to_deceleration_start(smooth_speed_mph, sample_times_s, candidate_peak)
    i = start + 1

    while i <= search_stop:
        if smooth_speed_mph[i] <= end_mph:
            return start, i, candidate_peak

        if smooth_speed_mph[i] > smooth_speed_mph[i - 1] + GPS_NOISE_INCREASE_MPH:
            while (
                i < search_stop
                and smooth_speed_mph[i + 1]
                >= smooth_speed_mph[i] - GPS_NOISE_INCREASE_MPH
            ):
                i += 1
            candidate_peak = i
            start = trim_to_deceleration_start(
                smooth_speed_mph,
                sample_times_s,
                candidate_peak,
            )
            i = start + 1
            continue

        i += 1

    return None


def circular_mean_deg(values: NDArray[np.float64]) -> float:
    """Circular mean of headings in degrees.

    Returns:
        The circular mean heading in degrees, or NaN if no finite values exist.

    """
    radians = np.deg2rad(values[np.isfinite(values)])
    if len(radians) == 0:
        return float("nan")
    return float(
        np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
        % 360.0,
    )


def cardinal_direction(heading_deg: float) -> str:
    """Return a readable 8-point compass direction.

    Returns:
        An 8-point compass label, or ``unknown`` for non-finite headings.

    """
    if not np.isfinite(heading_deg):
        return "unknown"
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[int((heading_deg + 22.5) // 45) % 8]


def read_experiment(
    input_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read CSVs from either a ZIP file or an extracted directory.

    Returns:
        Location, pressure, and temperature data frames.

    Raises:
        FileNotFoundError: A required CSV is missing from a ZIP input.
        ValueError: The input is neither a ZIP file nor a directory.

    """
    if input_path.is_dir():
        root = input_path
        return (
            pd.read_csv(root / "Location.csv"),
            pd.read_csv(root / "Pressure.csv"),
            pd.read_csv(root / "Temperature.csv"),
        )

    if input_path.suffix.lower() != ".zip":
        message = "Input must be the experiment ZIP or its extracted directory."
        raise ValueError(message)

    with zipfile.ZipFile(input_path) as zf:
        required = ["Location.csv", "Pressure.csv", "Temperature.csv"]
        missing = [name for name in required if name not in zf.namelist()]
        if missing:
            message = f"Missing from ZIP: {', '.join(missing)}"
            raise FileNotFoundError(message)
        with zf.open("Location.csv") as f:
            location = pd.read_csv(f)
        with zf.open("Pressure.csv") as f:
            pressure = pd.read_csv(f)
        with zf.open("Temperature.csv") as f:
            temperature = pd.read_csv(f)
    return location, pressure, temperature


def interpolate_sensor(
    sample_times: np.ndarray,
    sensor_df: pd.DataFrame,
    value_column: str,
) -> NDArray[np.float64]:
    """Linearly interpolate a high-rate sensor onto GPS timestamps.

    Returns:
        Sensor values interpolated at each GPS sample time.

    """
    sensor = sensor_df[["Time (s)", value_column]].dropna().sort_values("Time (s)")
    # Duplicate timestamps can make interpolation ambiguous.
    sensor = sensor.groupby("Time (s)", as_index=False)[value_column].mean()
    interpolated = cast(
        "object",
        np.interp(
            sample_times,
            sensor["Time (s)"].to_numpy(),
            sensor[value_column].to_numpy(),
        ),
    )
    return cast(
        "NDArray[np.float64]",
        np.asarray(interpolated, dtype=float),
    )


def detect_trials(
    location: pd.DataFrame,
    start_mph: float,
    end_mph: float,
    min_peak_distance_s: float,
    min_prominence_mph: float,
    min_duration_s: float,
) -> list[tuple[int, int, int]]:
    """Detect coast-downs as high-speed peaks followed by a sustained descent.

    Returns:
        Tuples of ``(start_index, end_index, peak_index)``.

    Raises:
        ValueError: Location timestamps are not usable.

    """
    t = location["Time (s)"].to_numpy(float)
    speed_mph = location["Velocity (m/s)"].to_numpy(float) * MPH_PER_MPS

    dt = np.nanmedian(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        message = "Location timestamps are invalid."
        raise ValueError(message)

    # Smooth only for detection. Exported trial data retains raw GPS velocity.
    window = max(5, round(9.0 / dt))
    if window % 2 == 0:
        window += 1
    if window >= len(speed_mph):
        window = len(speed_mph) - (1 - len(speed_mph) % 2)
    smooth = cast(
        "NDArray[np.float64]",
        np.asarray(
            savgol_filter(speed_mph, window_length=window, polyorder=2),
            dtype=float,
        ),
    )

    peak_distance_samples = max(1, round(min_peak_distance_s / dt))
    peaks = cast(
        "NDArray[np.int_]",
        np.asarray(
            find_peaks(
                smooth,
                height=start_mph,
                distance=peak_distance_samples,
                prominence=min_prominence_mph,
            )[0],
            dtype=int,
        ),
    )

    trials: list[tuple[int, int, int]] = []
    for p_i, peak_raw in enumerate(peaks):
        peak = int(peak_raw)
        # Search only until the next acceleration peak (or end of recording).
        search_stop = peaks[p_i + 1] if p_i + 1 < len(peaks) else len(smooth) - 1
        if search_stop <= peak + 2:
            continue

        segment = find_monotonic_coastdown(
            smooth,
            t,
            peak,
            int(search_stop),
            end_mph,
        )
        if segment is None:
            continue

        start, end, peak = segment

        duration = t[end] - t[start]
        if duration < min_duration_s:
            continue
        if smooth[start] < start_mph - 3:
            continue
        if end <= start:
            continue

        trials.append((start, end, peak))

    return trials


def main() -> None:
    """Run the coast-down extraction command.

    Raises:
        RuntimeError: No coast-down trials can be detected.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Experiment ZIP or extracted directory",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("coastdown_output"))
    parser.add_argument(
        "--start-mph",
        type=float,
        default=48.0,
        help="Minimum peak/start speed used to detect a trial",
    )
    parser.add_argument(
        "--end-mph",
        type=float,
        default=18.0,
        help="Nominal speed at which a trial ends",
    )
    parser.add_argument(
        "--min-peak-distance-s",
        type=float,
        default=70.0,
        help="Minimum time between run peaks",
    )
    parser.add_argument(
        "--min-prominence-mph",
        type=float,
        default=18.0,
        help="Required peak prominence",
    )
    parser.add_argument("--min-duration-s", type=float, default=20.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    location, pressure, temperature = read_experiment(args.input)

    location = location.sort_values("Time (s)").reset_index(drop=True).copy()
    t = location["Time (s)"].to_numpy(float)
    location["speed_mph"] = location["Velocity (m/s)"] * MPH_PER_MPS

    location["pressure_hPa"] = interpolate_sensor(t, pressure, "Pressure (hPa)")
    location["temperature_C"] = interpolate_sensor(t, temperature, "Temperature (°C)")

    # Air density using measured static pressure and temperature. Phone pressure
    # may include small cabin/dynamic effects, so use trial averages later.
    pressure_pa = location["pressure_hPa"].to_numpy() * 100.0
    temperature_k = location["temperature_C"].to_numpy() + 273.15
    location["air_density_kg_m3"] = pressure_pa / (R_DRY_AIR * temperature_k)

    # Relative barometric altitude. This is mainly useful for seeing local slope;
    # the absolute GPS altitude is not required.
    p_ref = float(np.nanmedian(location["pressure_hPa"]))
    location["baro_altitude_relative_m"] = 44330.0 * (
        1.0 - (location["pressure_hPa"] / p_ref) ** 0.190294957
    )

    trials = detect_trials(
        location,
        start_mph=args.start_mph,
        end_mph=args.end_mph,
        min_peak_distance_s=args.min_peak_distance_s,
        min_prominence_mph=args.min_prominence_mph,
        min_duration_s=args.min_duration_s,
    )

    if not trials:
        message = (
            "No coast-down trials found. Try reducing --start-mph or "
            "--min-prominence-mph."
        )
        raise RuntimeError(
            message,
        )

    all_trials: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []

    for trial_number, (start, end, peak) in enumerate(trials, start=1):
        segment = location.iloc[start : end + 1].copy()
        segment["trial"] = trial_number
        segment["trial_time_s"] = segment["Time (s)"] - segment["Time (s)"].iloc[0]

        heading = circular_mean_deg(segment["Direction (°)"].to_numpy(float))
        direction = cardinal_direction(heading)

        # A simple quality measure: fraction of successive GPS samples that
        # decrease. Small increases are normal at 1 Hz due to GPS noise.
        dv = np.diff(segment["speed_mph"].to_numpy(float))
        monotonic_fraction = (
            float(np.mean(dv <= GPS_NOISE_INCREASE_MPH)) if len(dv) else float("nan")
        )

        summaries.append(
            {
                "trial": trial_number,
                "direction": direction,
                "mean_heading_deg": heading,
                "start_time_s": float(segment["Time (s)"].iloc[0]),
                "end_time_s": float(segment["Time (s)"].iloc[-1]),
                "duration_s": float(segment["trial_time_s"].iloc[-1]),
                "start_speed_mph": float(segment["speed_mph"].iloc[0]),
                "peak_speed_mph": float(location["speed_mph"].iloc[peak]),
                "end_speed_mph": float(segment["speed_mph"].iloc[-1]),
                "distance_m": float(
                    np.trapezoid(
                        segment["Velocity (m/s)"].to_numpy(float),
                        segment["Time (s)"].to_numpy(float),
                    ),
                ),
                "mean_pressure_hPa": float(segment["pressure_hPa"].mean()),
                "mean_temperature_C": float(segment["temperature_C"].mean()),
                "mean_air_density_kg_m3": float(segment["air_density_kg_m3"].mean()),
                "baro_altitude_change_m": float(
                    segment["baro_altitude_relative_m"].iloc[-1]
                    - segment["baro_altitude_relative_m"].iloc[0],
                ),
                "gps_height_change_m": float(
                    segment["Height (m)"].iloc[-1] - segment["Height (m)"].iloc[0],
                ),
                "mostly_decreasing_fraction": monotonic_fraction,
            },
        )

        ordered = [
            "trial",
            "trial_time_s",
            "Time (s)",
            "Latitude (°)",
            "Longitude (°)",
            "Height (m)",
            "baro_altitude_relative_m",
            "Velocity (m/s)",
            "speed_mph",
            "Direction (°)",
            "Horizontal Accuracy (m)",
            "Vertical Accuracy (m)",
            "pressure_hPa",
            "temperature_C",
            "air_density_kg_m3",
        ]
        segment[ordered].to_csv(
            args.output / f"trial_{trial_number:02d}.csv",
            index=False,
        )
        all_trials.append(segment[ordered])

    combined = pd.concat(all_trials, ignore_index=True)
    summary = pd.DataFrame(summaries)
    combined.to_csv(args.output / "coastdown_trials.csv", index=False)
    summary.to_csv(args.output / "coastdown_summary.csv", index=False)

    # Detection plot.
    raw_fig, raw_ax = plt.subplots(figsize=(13, 6))
    fig = cast("_PlotFigure", raw_fig)
    ax = cast("_PlotAxes", raw_ax)
    ax.plot(
        location["Time (s)"],
        location["speed_mph"],
        linewidth=1.2,
        label="GPS speed",
    )
    for trial_number, (start, end, peak) in enumerate(trials, start=1):
        seg = location.iloc[start : end + 1]
        ax.plot(
            seg["Time (s)"],
            seg["speed_mph"],
            linewidth=2.5,
            label=f"Trial {trial_number}",
        )
        ax.text(
            location["Time (s)"].iloc[peak],
            location["speed_mph"].iloc[peak] + 1.2,
            str(trial_number),
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Experiment time (s)")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"Detected coast-down trials: {len(trials)}")
    ax.grid(visible=True, alpha=0.3)
    # Avoid a giant legend; the numeric labels identify trials.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:1], labels[:1], loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output / "coastdown_detection.png", dpi=180)
    plt.close(raw_fig)

    print(f"Detected {len(trials)} coast-down trials.")
    print(summary.to_string(index=False))
    print(f"\nWrote results to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
