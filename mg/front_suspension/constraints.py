"""Allowed wheel travel and steering limits for the front suspension."""

from __future__ import annotations

from mg.front_suspension.utils import Angle

BUMP = (-2, 2.5)
STEERING = (-2, 2)

STATIC_CAMBER = (Angle.from_deg(-1), Angle.from_deg(-0.5))
STATIC_CAMBER_SLOPE = (Angle.from_deg(-0.7), Angle.from_deg(-0.3))
BUMP_CAMBER_MAX = Angle.from_deg(0)
DROOP_CAMBER_MAX = Angle.from_deg(0.5)

ROLL_CENTER_PP = 2.0
BUMP_STEER_PP = Angle.from_deg(0.2)
COILOVER_MOTION_RATIO = (0.6, 0.9)

_WHEEL_CENTER_X_NOMINAL = 23.7
_WHEEL_CENTER_X_TOL = 0.25
WHEEL_CENTER_X = (
    _WHEEL_CENTER_X_NOMINAL - _WHEEL_CENTER_X_TOL,
    _WHEEL_CENTER_X_NOMINAL + _WHEEL_CENTER_X_TOL,
)
