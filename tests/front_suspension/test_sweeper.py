# ruff: file-ignore[undocumented-public-module, undocumented-public-method, pytest-unittest-assertion, pytest-unittest-raises-assertion, private-member-access]
from __future__ import annotations

import unittest
from dataclasses import replace

from mg.front_suspension.geometry import GeometryFixed, Solver
from mg.front_suspension.sweeper import SuspensionSweep
from mg.front_suspension.utils import Point


class TestSuspensionTargets(unittest.TestCase):
    """Check target evaluation using the suspension solver."""

    def test_target_tests_ignore_sample_order(self) -> None:
        sweep = SuspensionSweep(GeometryFixed(), bump_range=[-2.0, -1.0, 1.0, 2.5])
        reversed_sweep = replace(sweep, bump_range=list(reversed(sweep.bump_range)))

        result = reversed_sweep.target_tests

        self.assertEqual(result, sweep.target_tests)

    def test_target_tests_reject_incomplete_travel(self) -> None:
        sweep = SuspensionSweep(GeometryFixed(), bump_range=[-1.0, 0.0, 2.5])

        with self.assertRaisesRegex(ValueError, "cover"):
            _ = sweep.target_tests

    def test_roll_center_uses_contact_height(self) -> None:
        state = Solver.at(GeometryFixed(), 0)
        state = replace(
            state,
            lp=Point(1, 0, 1),
            lf=Point(2, 0, 2),
            up=Point(1, 0, 3),
            uf=Point(2, 0, 2),
            wheel_center=Point(3, 0, 1),
            _wheel_contact_z=0,
        )

        height_in = SuspensionSweep._roll_center_height_in(state)

        self.assertAlmostEqual(height_in, 6.0)
