# ruff: file-ignore[undocumented-public-module, undocumented-public-method, pytest-unittest-assertion, pytest-unittest-raises-assertion]
from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

from colorama import Fore, Style

from mg.front_suspension.geometry import GeometryFixed, Solver
from mg.front_suspension.sweeper import SuspensionSweep
from mg.front_suspension.utils import Point


class TestSuspensionTargets(unittest.TestCase):
    """Check target evaluation using the suspension solver."""

    def test_report_prints_measured_targets_with_status_colors(self) -> None:
        sweep = SuspensionSweep(GeometryFixed())
        output = StringIO()

        with redirect_stdout(output):
            sweep.report()

        report = output.getvalue()
        self.assertIn(f"{Fore.RED}[FAIL]{Style.RESET_ALL} Static camber ", report)
        self.assertIn(f"{Fore.GREEN}[PASS]{Style.RESET_ALL} Bump steer", report)
        self.assertIn(f"{Fore.RED}[FAIL]{Style.RESET_ALL} Static toe", report)
        self.assertIn(f"{Fore.BLUE}0.125{Style.RESET_ALL} deg", report)
        self.assertIn(f"target <= {Fore.MAGENTA}0.200{Style.RESET_ALL} deg", report)

    def test_static_toe_is_measured_at_rest(self) -> None:
        sweep = SuspensionSweep(GeometryFixed(), bump_range=[-2.0, 0.0, 2.5])

        actual = sweep.target_values.static_toe_deg

        self.assertAlmostEqual(actual, Solver.at(sweep.geometry, 0).toe.deg)

    def test_static_toe_is_a_hard_constraint(self) -> None:
        sweep = SuspensionSweep(GeometryFixed())
        result = replace(
            sweep.target_tests,
            static_camber=True,
            bump_camber=True,
            droop_camber=True,
            bump_steer=True,
            wheel_center=True,
        )

        passed = result.hard_constraints_passed

        self.assertFalse(result.static_toe)
        self.assertFalse(passed)

    def test_static_toe_limits_are_inclusive(self) -> None:
        sweep = SuspensionSweep(GeometryFixed())
        values = sweep.target_values

        for toe_deg, expected in (
            (-0.13, True),
            (-0.03, True),
            (-0.131, False),
            (-0.029, False),
        ):
            with self.subTest(toe_deg=toe_deg):
                sample = SuspensionSweep(GeometryFixed())
                sample.__dict__["target_values"] = replace(
                    values,
                    static_toe_deg=toe_deg,
                )
                self.assertEqual(sample.target_tests.static_toe, expected)

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

        height_in = state.roll_center.z - state.wheel_contact.z

        self.assertAlmostEqual(height_in, 6.0)

    def test_roll_center_world_height_tracks_contact(self) -> None:
        state = Solver.at(GeometryFixed(), 1.0)

        world = state.world

        self.assertAlmostEqual(world.roll_center.z, state.roll_center.z - 1.0)

    def test_roll_center_migration_is_a_soft_constraint(self) -> None:
        sweep = SuspensionSweep(GeometryFixed())

        result = sweep.target_tests

        self.assertGreater(sweep.target_values.roll_center_pp_in, 2.0)
        self.assertFalse(result.roll_center)
        self.assertTrue(
            replace(
                result,
                static_camber=True,
                static_toe=True,
                bump_camber=True,
                droop_camber=True,
                bump_steer=True,
                wheel_center=True,
            ).hard_constraints_passed,
        )

    def test_parallel_arms_have_a_finite_roll_center(self) -> None:
        state = replace(
            Solver.at(GeometryFixed(), 0),
            lp=Point(1, 0, 1),
            lf=Point(2, 0, 2),
            up=Point(1, 0, 3),
            uf=Point(2, 0, 4),
            wheel_center=Point(3, 0, 1),
            _wheel_contact_z=0,
        )

        center = state.roll_center

        self.assertAlmostEqual(center.z, -3.0)
