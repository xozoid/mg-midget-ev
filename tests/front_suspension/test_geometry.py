# ruff: file-ignore[undocumented-public-module, undocumented-public-method, pytest-unittest-assertion, pytest-unittest-raises-assertion]
from __future__ import annotations

import unittest
from dataclasses import replace

import matplotlib.pyplot as plt

from front_suspension.helpers.geometry import GeometryFixed, Solver
from front_suspension.helpers.utils import Point


class TestInstantCenter(unittest.TestCase):
    """Check the front-view instant center and its construction."""

    def test_instant_center_intersects_arm_lines(self) -> None:
        state = replace(
            Solver.at(GeometryFixed(), 0),
            lp=Point(1, 1, 1),
            lf=Point(2, 2, 2),
            up=Point(1, 1, 3),
            uf=Point(2, 2, 2),
        )

        center = state.instant_center

        self.assertAlmostEqual(center.x, 2)
        self.assertAlmostEqual(center.y, 2)
        self.assertAlmostEqual(center.z, 2)

    def test_parallel_arms_have_no_finite_instant_center(self) -> None:
        state = replace(
            Solver.at(GeometryFixed(), 0),
            lp=Point(1, 0, 1),
            lf=Point(2, 0, 2),
            up=Point(1, 0, 3),
            uf=Point(2, 0, 4),
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            _ = state.instant_center

    def test_plot_can_hide_roll_center_construction(self) -> None:
        state = Solver.at(GeometryFixed(), 0)
        fig, (shown, hidden) = plt.subplots(1, 2)
        self.addCleanup(plt.close, fig)
        state.plot(shown, "XZ")

        state.plot(hidden, "XZ", roll_center_construction=False)

        self.assertEqual(len(shown.lines) - len(hidden.lines), 5)
