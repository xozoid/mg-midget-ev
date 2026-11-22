# ruff: file-ignore[undocumented-public-module, undocumented-public-method, pytest-unittest-assertion]
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from front_suspension.candidates import CANDIDATES
from front_suspension.sensitivity import (
    ADJUSTABLE_DIMENSIONS,
    evaluate,
    failed_target,
    Trial,
)


class TestSensitivity(unittest.TestCase):
    """Check alignment adjustment and suspension evaluation."""

    def test_only_adjustable_dimensions_are_varied(self) -> None:
        self.assertEqual(
            ADJUSTABLE_DIMENSIONS,
            ("up_x", "up_z", "arm_upper", "arm_lower"),
        )

    def test_numpy_boolean_failure_is_reported(self) -> None:
        row = {"bump_steer_pass": np.bool_(0)}

        failed = failed_target(row, "bump_steer")

        self.assertTrue(failed)

    def test_baseline_passes_with_adjustable_tie_rod(self) -> None:
        geometry = CANDIDATES[1]
        trial = Trial("baseline", geometry, (0.0,) * len(ADJUSTABLE_DIMENSIONS), 0.25)

        result = evaluate(trial)

        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(result["pass"])
        self.assertAlmostEqual(float(result["static_toe_deg"]), -0.08, places=3)

    def test_adjustment_can_restore_static_toe(self) -> None:
        geometry = replace(CANDIDATES[1], arm_tie_rod=CANDIDATES[1].arm_tie_rod + 0.1)
        trial = Trial("baseline", geometry, (0.0,) * len(ADJUSTABLE_DIMENSIONS), 0.25)

        result = evaluate(trial)

        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(result["pass"])
        self.assertAlmostEqual(float(result["tie_rod_adjustment_in"]), -0.1, places=3)

    def test_limited_adjustment_rejects_unreachable_toe(self) -> None:
        geometry = replace(CANDIDATES[1], arm_tie_rod=CANDIDATES[1].arm_tie_rod + 0.1)
        trial = Trial("baseline", geometry, (0.0,) * len(ADJUSTABLE_DIMENSIONS), 0.01)

        result = evaluate(trial)

        self.assertEqual(result["status"], "toe_out_of_range")
        self.assertFalse(result["pass"])
        self.assertAlmostEqual(float(result["tie_rod_adjustment_in"]), -0.01, places=6)

    def test_adjustment_limit_can_still_meet_toe_range(self) -> None:
        geometry = replace(
            CANDIDATES[1],
            arm_tie_rod=CANDIDATES[1].arm_tie_rod + 0.004,
        )
        trial = Trial("baseline", geometry, (0.0,) * len(ADJUSTABLE_DIMENSIONS), 0)

        result = evaluate(trial)

        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(result["pass"])
        self.assertEqual(result["tie_rod_adjustment_in"], 0)
