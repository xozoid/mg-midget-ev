#!/usr/bin/env python3
"""Fit effective rolling resistance and CdA from extracted coast-down trials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from numpy.typing import NDArray

G = 9.80665
MPH_PER_MPS = 2.2369362920544
LB_TO_KG = 0.45359237


def simulate(
    df: pd.DataFrame,
    sign: float,
    params: NDArray[np.float64],
    mass_kg: float,
) -> NDArray[np.float64]:
    """Simulate one coast-down speed trace.

    Returns:
        Predicted vehicle speed in meters per second at each trial sample.

    """
    crr, cda, grade = params
    t = df["trial_time_s"].to_numpy(dtype=float)
    rho = df["air_density_kg_m3"].to_numpy(dtype=float)
    v = np.empty_like(t)
    v[0] = float(df["Velocity (m/s)"].iloc[0])

    def acceleration(v_now: float, rho_now: float) -> float:
        return (
            (-G * crr) - (0.5 * rho_now * cda * v_now**2 / mass_kg) - (sign * G * grade)
        )

    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        r0, r1 = rho[i - 1], rho[i]
        rm = 0.5 * (r0 + r1)
        k1 = acceleration(v[i - 1], r0)
        k2 = acceleration(v[i - 1] + (0.5 * h * k1), rm)
        k3 = acceleration(v[i - 1] + (0.5 * h * k2), rm)
        k4 = acceleration(v[i - 1] + (h * k3), r1)
        v[i] = v[i - 1] + (h * (k1 + (2 * k2) + (2 * k3) + k4) / 6)
    return v


def main() -> None:
    """Run the glider model fit command."""
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_directory", type=Path)
    ap.add_argument("--mass-lb", type=float, default=2075.0)
    ap.add_argument(
        "--split",
        type=int,
        default=5,
        help="Number of trials in direction A; remaining trials are direction B",
    )
    args = ap.parse_args()

    files = sorted(args.trial_directory.glob("trial_*.csv"))
    trials = [pd.read_csv(f) for f in files]
    signs = np.array([1] * args.split + [-1] * (len(trials) - args.split), dtype=float)
    mass_kg = args.mass_lb * LB_TO_KG

    def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
        out: list[float] = []
        for df, sign in zip(trials, signs, strict=True):
            pred = simulate(df, sign, params, mass_kg)
            actual = df["Velocity (m/s)"].to_numpy(dtype=float)
            out.extend(pred[1:] - actual[1:])
        return np.asarray(out, dtype=float)

    result = least_squares(
        residuals,
        np.asarray([0.02, 0.60, 0.003], dtype=float),
        bounds=([0, 0, -0.02], [0.08, 2.0, 0.02]),
        loss="soft_l1",
        f_scale=0.15,
    )

    crr, cda, grade = result.x
    rmse = np.sqrt(np.mean(residuals(result.x) ** 2))
    print(f"Effective Crr: {crr:.5f}")
    print(f"CdA:           {cda:.4f} m^2")
    print(f"Grade term:    {grade * 100:.3f}%")
    print(f"RMSE:          {rmse * MPH_PER_MPS:.3f} mph")


if __name__ == "__main__":
    main()
