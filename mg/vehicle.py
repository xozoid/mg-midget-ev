"""Vehicle specification and simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mg.cost import ConversionCostModel
from mg.utils import (
    AIR_DENSITY,
    G,
    grade_force_n,
    HP_PER_KW,
    INCH_TO_M,
    mph_to_mps,
    mps_to_mph,
)

if TYPE_CHECKING:
    from typing import Literal

    from mg.battery import BatteryPack
    from mg.motor import Motor


# ------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------


class OperatingLimitError(ValueError):
    """Raised when a cruise operating point exceeds a component limit."""


# ------------------------------------------------------------
# Result types
# ------------------------------------------------------------


@dataclass
class CruiseResult:
    """Output of a single cruise operating-point calculation."""

    speed_mph: float
    grade_percent: float
    gear: int
    motor_rpm: float
    motor_efficiency: float
    mass_kg: float
    aero_n: float
    rolling_n: float
    grade_n: float
    total_force_n: float
    wheel_power_kw: float
    battery_power_kw: float
    pack_current_a: float
    discharge_c: float
    regen_c: float
    range_miles: float

    def print_summary(self, name: str) -> None:
        """Print a formatted cruise operating-point result.

        Args:
            name: Display label shown in the section header.

        """
        print(f"=== {name.upper()} ===")
        print(f"Speed:                        {self.speed_mph:.1f} mph")
        print(f"Grade:                        {self.grade_percent:.1f} %")
        print(
            f"Gear / motor speed:           {self.gear} / {self.motor_rpm:.0f} rpm",
        )
        print(f"Motor efficiency:             {self.motor_efficiency * 100.0:.1f} %")
        print(f"Wheel power:                  {self.wheel_power_kw:.2f} kW")
        print(f"Battery power:                {self.battery_power_kw:.2f} kW")
        print(f"Pack current:                 {self.pack_current_a:.1f} A")
        if self.pack_current_a >= 0:
            print(f"Discharge C-rate:             {self.discharge_c:.2f} C")
        else:
            print(f"Regen charge C-rate:          {self.regen_c:.2f} C")
        if math.isfinite(self.range_miles):
            print(f"Estimated range:              {self.range_miles:.1f} miles")
        else:
            print("Estimated range:              effectively unlimited on this descent")
        print()


@dataclass
class CruiseGearChoice:
    """Gear selected for a steady-state cruise operating point."""

    gear: int
    motor_rpm: float
    motor_efficiency: float
    available_force_n: float


@dataclass
class GradeabilityResult:
    """Output of a single gradeability calculation."""

    gear: int
    speed_mph: float
    rpm: float
    valid: bool
    max_grade_percent: float
    available_force_n: float


@dataclass
class AccelSegment:
    """One speed-band segment of a 0-60 acceleration estimate."""

    from_mph: float
    to_mph: float
    gear: int | None
    rpm_mid: float
    accel_mps2: float
    dt_s: float


@dataclass
class AccelResult:
    """Output of a 0-60 mph acceleration estimate."""

    zero_to_60_s: float
    segments: list[AccelSegment]

    def print_summary(self) -> None:
        """Print a formatted acceleration estimate."""
        print("=== 0-60 ESTIMATE ===")
        print(f"Estimated 0-60 mph:           {self.zero_to_60_s:.2f} s")
        for seg in self.segments[::5]:
            gear_str = str(seg.gear) if seg.gear is not None else "?"
            print(
                f"{seg.from_mph:>4.0f}-{seg.to_mph:<4.0f} mph | "
                f"gear {gear_str} | rpm {seg.rpm_mid:>6.0f} | "
                f"a {seg.accel_mps2:.2f} m/s^2",
            )
        print()


@dataclass
class BestForceResult:
    """Best available tractive force across all gears at a given speed."""

    force_n: float
    gear: int | None
    rpm: float


# ------------------------------------------------------------
# Vehicle
# ------------------------------------------------------------


@dataclass
class Vehicle:
    """Physical and drivetrain specification for the converted vehicle."""

    battery: BatteryPack
    motor: Motor

    name: str = "MG Midget EV"

    # Glider = bare rolling car before motor/batteries/EV hardware
    glider_mass_kg: float = 700.0
    """Mass of the bare rolling chassis before any EV hardware."""

    misc_ev_mass_kg: float = 45.0
    """Mass of miscellaneous EV hardware (controller, wiring, charger, etc.)."""

    cd: float = 0.48
    """Drag coefficient."""

    frontal_area_m2: float = 1.67
    """Frontal area in square metres."""

    rolling_resistance: float = 0.013
    """Coefficient of rolling resistance."""

    tire_diameter_in: float = 22.1
    """Loaded tire diameter in inches."""

    final_drive_ratio: float = 3.90
    """Final drive (differential) ratio."""

    transmission_efficiency: float = 0.97
    """Gearbox power transmission efficiency (0-1)."""

    differential_efficiency: float = 0.98
    """Differential power transmission efficiency (0-1)."""

    accessory_power_kw: float = 0.40
    """Continuous 12 V accessory load in kilowatts."""

    regen_recovery_efficiency: float = 0.70
    """Fraction of braking energy recovered through regeneration (0-1)."""

    traction_limit_g: float = 0.7
    """Maximum traction-limited acceleration in g."""

    gear_ratios: dict[Literal[1, 2, 3, 4], float] = field(
        default_factory=lambda: {
            1: 3.20,
            2: 2.00,
            3: 1.378,
            4: 1.00,
        },
    )
    """Gearbox ratio for each gear number."""

    # ----------------------------------------------------------
    # Basic kinematics and forces
    # ----------------------------------------------------------

    @property
    def total_mass_kg(self) -> float:
        """Total vehicle mass including battery pack in kilograms."""
        return (
            self.glider_mass_kg
            + self.motor.mass_kg
            + self.misc_ev_mass_kg
            + self.battery.total_pack_mass_kg
        )

    @property
    def wheel_radius_m(self) -> float:
        """Loaded wheel radius in metres."""
        return (self.tire_diameter_in * INCH_TO_M) / 2.0

    @property
    def wheel_circumference_m(self) -> float:
        """Loaded wheel circumference in metres."""
        return 2.0 * math.pi * self.wheel_radius_m

    @property
    def driveline_efficiency(self) -> float:
        """Combined transmission and differential efficiency (0-1)."""
        return self.transmission_efficiency * self.differential_efficiency

    def wheel_rpm_from_speed(self, speed_mph: float) -> float:
        """Wheel rotational speed for a given road speed.

        Args:
            speed_mph: Vehicle speed in miles per hour.

        Returns:
            Wheel speed in revolutions per minute.

        """
        return (mph_to_mps(speed_mph) / self.wheel_circumference_m) * 60.0

    def motor_rpm(self, speed_mph: float, gear: Literal[1, 2, 3, 4]) -> float:
        """Motor shaft speed for a given road speed and gear.

        Args:
            speed_mph: Vehicle speed in miles per hour.
            gear: Gear number (1-4).

        Returns:
            Motor speed in revolutions per minute.

        """
        return (
            self.wheel_rpm_from_speed(speed_mph)
            * self.gear_ratios[gear]
            * self.final_drive_ratio
        )

    def speed_mph_at_motor_rpm(self, rpm: float, gear: Literal[1, 2, 3, 4]) -> float:
        """Road speed corresponding to a given motor RPM and gear.

        Args:
            rpm: Motor speed in revolutions per minute.
            gear: Gear number (1-4).

        Returns:
            Vehicle speed in miles per hour.

        """
        wheel_rpm = rpm / (self.gear_ratios[gear] * self.final_drive_ratio)
        speed_mps = (wheel_rpm / 60.0) * self.wheel_circumference_m
        return mps_to_mph(speed_mps)

    def aero_force_n(self, speed_mps: float) -> float:
        """Aerodynamic drag force at a given speed.

        Args:
            speed_mps: Vehicle speed in metres per second.

        Returns:
            Aerodynamic drag force in Newtons.

        """
        return 0.5 * AIR_DENSITY * self.cd * self.frontal_area_m2 * speed_mps**2

    def rolling_force_n(self) -> float:
        """Return rolling resistance force at the current vehicle mass.

        Returns:
            Rolling resistance force in Newtons.

        """
        return self.total_mass_kg * G * self.rolling_resistance

    def road_load_forces(
        self,
        speed_mph: float,
        grade_percent: float,
    ) -> dict[str, float]:
        """All road-load force components at a given speed and grade.

        Args:
            speed_mph: Vehicle speed in miles per hour.
            grade_percent: Road grade as a percentage (positive = uphill).

        Returns:
            Dict with keys ``aero_n``, ``rolling_n``, ``grade_n``, ``total_n``
            in Newtons.

        """
        f_aero = self.aero_force_n(mph_to_mps(speed_mph))
        f_roll = self.rolling_force_n()
        f_grade = grade_force_n(self.total_mass_kg, grade_percent)
        return {
            "aero_n": f_aero,
            "rolling_n": f_roll,
            "grade_n": f_grade,
            "total_n": f_aero + f_roll + f_grade,
        }

    def wheel_torque_nm(self, gear: Literal[1, 2, 3, 4], rpm: float) -> float:
        """Torque delivered to the wheels through the driveline.

        Args:
            gear: Gear number (1-4).
            rpm: Motor speed in revolutions per minute.

        Returns:
            Wheel torque in Newton-metres.

        """
        return (
            self.motor.available_torque_nm(rpm)
            * self.gear_ratios[gear]
            * self.final_drive_ratio
            * self.driveline_efficiency
        )

    def drive_force_n(self, gear: Literal[1, 2, 3, 4], rpm: float) -> float:
        """Tractive force at the contact patch.

        Args:
            gear: Gear number (1-4).
            rpm: Motor speed in revolutions per minute.

        Returns:
            Drive force in Newtons.

        """
        return self.wheel_torque_nm(gear, rpm) / self.wheel_radius_m

    def continuous_drive_force_n(
        self,
        gear: Literal[1, 2, 3, 4],
        rpm: float,
    ) -> float:
        """Continuous tractive force at the contact patch.

        Args:
            gear: Gear number (1-4).
            rpm: Motor speed in revolutions per minute.

        Returns:
            Continuous drive force in Newtons.

        """
        wheel_torque_nm = (
            self.motor.continuous_torque_nm(rpm)
            * self.gear_ratios[gear]
            * self.final_drive_ratio
            * self.driveline_efficiency
        )
        return wheel_torque_nm / self.wheel_radius_m

    def traction_limited_force_n(self) -> float:
        """Maximum tractive force before wheel slip.

        Returns:
            Traction-limited drive force in Newtons.

        """
        return self.traction_limit_g * self.total_mass_kg * G

    # ----------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------

    def best_continuous_cruise_gear(
        self,
        speed_mph: float,
        required_force_n: float,
    ) -> CruiseGearChoice:
        """Return the most efficient valid gear for a cruise operating point.

        Args:
            speed_mph: Vehicle speed in miles per hour.
            required_force_n: Wheel force needed at the contact patch in
                Newtons.

        Returns:
            The valid gear with the highest continuous motor efficiency at the
            corresponding motor RPM.

        Raises:
            OperatingLimitError: If no gear can operate at the speed and supply
                the required positive wheel force.

        """
        traction_limit_n = self.traction_limited_force_n()
        best_choice: CruiseGearChoice | None = None

        for gear in self.gear_ratios:
            motor_rpm = self.motor_rpm(speed_mph, gear)
            if motor_rpm > self.motor.max_rpm:
                continue

            available_force_n = min(
                self.continuous_drive_force_n(gear, motor_rpm),
                traction_limit_n,
            )
            if required_force_n > 0.0 and available_force_n < required_force_n:
                continue

            choice = CruiseGearChoice(
                gear=gear,
                motor_rpm=motor_rpm,
                motor_efficiency=self.motor.continuous_efficiency_at_rpm(motor_rpm),
                available_force_n=available_force_n,
            )
            if (
                best_choice is None
                or choice.motor_efficiency > best_choice.motor_efficiency
            ):
                best_choice = choice

        if best_choice is None:
            msg = (
                f"No gear can sustain {required_force_n:.0f} N at "
                f"{speed_mph:.1f} mph continuously"
            )
            raise OperatingLimitError(msg)

        return best_choice

    def cruise_case(
        self,
        speed_mph: float,
        grade_percent: float,
        motor_efficiency: float | None = None,
    ) -> CruiseResult:
        """Steady-state cruise operating point.

        Computes wheel power, battery power, pack current, C-rates, and
        estimated range for the given speed and road grade. Negative wheel
        power (descending) is handled as regenerative braking.

        Args:
            speed_mph: Vehicle speed in miles per hour.
            grade_percent: Road grade as a percentage (positive = uphill).
            motor_efficiency: Motor efficiency override (0-1). Defaults to
                the continuous efficiency at the RPM of the most efficient
                valid gear.

        Returns:
            ``CruiseResult`` with all operating-point values.

        Raises:
            OperatingLimitError: If no gear can sustain the operating point,
                pack discharge current exceeds the pack continuous limit, motor
                peak current limit, or regen current exceeds the pack maximum
                charge current.

        """
        forces = self.road_load_forces(speed_mph, grade_percent)
        p_wheel_kw = forces["total_n"] * mph_to_mps(speed_mph) / 1000.0

        gear_choice = self.best_continuous_cruise_gear(
            speed_mph,
            forces["total_n"],
        )
        motor_efficiency_value = (
            motor_efficiency
            if motor_efficiency is not None
            else gear_choice.motor_efficiency
        )
        total_eff = self.driveline_efficiency * motor_efficiency_value

        if p_wheel_kw >= 0:
            p_batt_kw = (p_wheel_kw / max(total_eff, 1e-6)) + self.accessory_power_kw
        else:
            regen_kw = (
                (-p_wheel_kw) * self.regen_recovery_efficiency * motor_efficiency_value
            )
            p_batt_kw = self.accessory_power_kw - regen_kw

        pack_current_a = (p_batt_kw * 1000.0) / self.battery.nominal_voltage_v
        discharge_c = max(pack_current_a, 0.0) / self.battery.capacity_ah
        regen_c = max(-pack_current_a, 0.0) / self.battery.capacity_ah

        if p_batt_kw > 0:
            range_miles = (self.battery.usable_energy_kwh / p_batt_kw) * speed_mph
        else:
            range_miles = float("inf")

        if pack_current_a > self.battery.continuous_pack_current_a:
            msg = (
                f"Discharge current {pack_current_a:.0f} A exceeds pack "
                f"continuous limit {self.battery.continuous_pack_current_a:.0f} A"
            )
            raise OperatingLimitError(msg)
        if pack_current_a > self.motor.peak_pack_current_a:
            msg = (
                f"Requested current {pack_current_a:.0f} A exceeds motor "
                f"peak limit {self.motor.peak_pack_current_a:.0f} A"
            )
            raise OperatingLimitError(msg)
        if -pack_current_a > self.battery.max_regen_charge_current_a:
            msg = (
                f"Regen current {-pack_current_a:.0f} A exceeds pack max "
                f"charge current {self.battery.max_regen_charge_current_a:.0f} A"
            )
            raise OperatingLimitError(msg)

        return CruiseResult(
            speed_mph=speed_mph,
            grade_percent=grade_percent,
            gear=gear_choice.gear,
            motor_rpm=gear_choice.motor_rpm,
            motor_efficiency=motor_efficiency_value,
            mass_kg=self.total_mass_kg,
            aero_n=forces["aero_n"],
            rolling_n=forces["rolling_n"],
            grade_n=forces["grade_n"],
            total_force_n=forces["total_n"],
            wheel_power_kw=p_wheel_kw,
            battery_power_kw=p_batt_kw,
            pack_current_a=pack_current_a,
            discharge_c=discharge_c,
            regen_c=regen_c,
            range_miles=range_miles,
        )

    def gradeability_at_speed(
        self,
        speed_mph: float,
        gear: Literal[1, 2, 3, 4],
    ) -> GradeabilityResult:
        """Maximum climbable grade at a given speed and gear.

        Args:
            speed_mph: Vehicle speed in miles per hour.
            gear: Gear number (1-4).

        Returns:
            ``GradeabilityResult`` with valid flag and max grade percentage.
            When ``valid`` is ``False`` the motor exceeds ``max_rpm`` and
            ``max_grade_percent`` is ``nan``.

        """
        rpm = self.motor_rpm(speed_mph, gear)

        if rpm > self.motor.max_rpm:
            return GradeabilityResult(
                gear=gear,
                speed_mph=speed_mph,
                rpm=rpm,
                valid=False,
                max_grade_percent=float("nan"),
                available_force_n=0.0,
            )

        force_motor = self.drive_force_n(gear, rpm)
        force_limit = min(force_motor, self.traction_limited_force_n())

        f_left = (
            force_limit
            - self.aero_force_n(mph_to_mps(speed_mph))
            - self.rolling_force_n()
        )
        max_grade_percent = (f_left / (self.total_mass_kg * G)) * 100.0

        return GradeabilityResult(
            gear=gear,
            speed_mph=speed_mph,
            rpm=rpm,
            valid=True,
            max_grade_percent=max_grade_percent,
            available_force_n=force_limit,
        )

    def best_force_at_speed(self, speed_mph: float) -> BestForceResult:
        """Best tractive force available across all gears at a given speed.

        Selects the gear that maximises drive force without exceeding
        ``motor.max_rpm`` or the traction limit.

        Args:
            speed_mph: Vehicle speed in miles per hour.

        Returns:
            ``BestForceResult`` with the maximum force, the gear that
            achieves it, and the corresponding motor RPM. ``gear`` is
            ``None`` if no gear is valid at this speed.

        """
        traction_limit = self.traction_limited_force_n()
        best_force = 0.0
        best_gear: int | None = None
        best_rpm = 0.0

        for gear in self.gear_ratios:
            rpm = self.motor_rpm(speed_mph, gear)
            if rpm > self.motor.max_rpm:
                continue
            f = min(self.drive_force_n(gear, rpm), traction_limit)
            if f > best_force:
                best_force = f
                best_gear = gear
                best_rpm = rpm

        return BestForceResult(force_n=best_force, gear=best_gear, rpm=best_rpm)

    def estimate_0_to_n(
        self,
        target_mph: float = 60.0,
        step_mph: float = 1.0,
    ) -> AccelResult:
        """Estimate 0-to-N mph time using a constant-force stepping method.

        Divides the speed range into bands of ``step_mph`` width, computes
        the best available drive force at the midpoint of each band, and
        integrates dt = dv / a over all bands.

        Args:
            target_mph: Target speed in miles per hour. Defaults to 60.
            step_mph: Speed increment per integration step in mph.
                Smaller values give more accurate results.

        Returns:
            ``AccelResult`` with total elapsed time and per-band segments.

        """
        v = 0.5
        total_time_s = 0.0
        segments: list[AccelSegment] = []

        while v < target_mph:
            v2 = min(v + step_mph, target_mph)
            vmid = 0.5 * (v + v2)

            best = self.best_force_at_speed(vmid)
            road = self.road_load_forces(vmid, 0.0)
            net_force = max(best.force_n - road["total_n"], 1.0)
            accel = net_force / self.total_mass_kg
            dt = mph_to_mps(v2 - v) / accel
            total_time_s += dt

            segments.append(
                AccelSegment(
                    from_mph=v,
                    to_mph=v2,
                    gear=best.gear,
                    rpm_mid=best.rpm,
                    accel_mps2=accel,
                    dt_s=dt,
                ),
            )
            v = v2

        return AccelResult(zero_to_60_s=total_time_s, segments=segments)

    def print_summary(self) -> None:
        """Print a formatted summary of the vehicle and motor."""
        motor = self.motor
        total_mass = self.total_mass_kg
        print("=== VEHICLE / MOTOR SUMMARY ===")
        print(f"Vehicle:                      {self.name}")
        print(f"Glider mass:                  {self.glider_mass_kg:.1f} kg")
        print(f"Motor mass:                   {self.motor.mass_kg:.1f} kg")
        print(f"Battery pack mass:            {self.battery.total_pack_mass_kg:.1f} kg")
        print(f"Misc EV hardware mass:        {self.misc_ev_mass_kg:.1f} kg")
        print(f"Total vehicle mass:           {total_mass:.1f} kg")
        print(f"Total vehicle mass:           {total_mass * 2.20462:.0f} lb")
        print(f"Motor:                        {motor.name}")
        print(f"Motor price:                  ${motor.today_price_usd:,.0f}")
        print(
            f"Motor peak power:             "
            f"{motor.peak_power_kw:.1f} kW ({motor.peak_power_kw * HP_PER_KW:.0f} hp)",
        )
        print(f"Motor continuous power:       {motor.continuous_power_kw:.1f} kW")
        print(f"Motor peak torque:            {motor.peak_torque_lbft:.0f} lb-ft")
        print(f"Motor max rpm:                {motor.max_rpm:.0f}")
        print(f"Final drive ratio:            {self.final_drive_ratio:.2f}")
        print(f"Tire diameter:                {self.tire_diameter_in:.1f} in")
        print()

    def print_top_speed_table(self) -> None:
        """Print the road speed at motor redline for each gear."""
        print("=== SPEED AT MOTOR REDLINE BY GEAR ===")
        for gear in self.gear_ratios:
            mph = self.speed_mph_at_motor_rpm(self.motor.max_rpm, gear)
            print(f"  gear {gear}: {mph:6.1f} mph")
        print()

    def print_rpm_table(self, speeds: list[float]) -> None:
        """Print motor RPM in each gear at the given road speeds.

        Args:
            speeds: List of road speeds in miles per hour.

        """
        print("=== MOTOR RPM TABLE ===")
        for speed in speeds:
            print(f"\nSpeed {speed:.0f} mph")
            for gear in self.gear_ratios:
                rpm = self.motor_rpm(speed, gear)
                over = " OVER" if rpm > self.motor.max_rpm else ""
                print(f"  gear {gear}: {rpm:7.0f} rpm{over}")
        print()

    def print_gradeability_table(self, speed_mph: float) -> None:
        """Print the maximum climbable grade in each gear at a given speed.

        Args:
            speed_mph: Road speed in miles per hour.

        """
        print(f"=== MAX GRADE AT {speed_mph:.0f} MPH BY GEAR ===")
        for gear in self.gear_ratios:
            row = self.gradeability_at_speed(speed_mph, gear)
            if not row.valid:
                print(f"  gear {row.gear}: invalid (motor above max rpm)")
            else:
                print(
                    f"  gear {row.gear}: max grade ~ "
                    f"{row.max_grade_percent:.1f}% at {row.rpm:.0f} rpm",
                )
        print()

    def build_cost_model(self) -> ConversionCostModel:
        """Build a cost model seeded with this vehicle's motor and cell costs.

        Returns:
            ``ConversionCostModel`` with ``motor_kit_usd`` and
            ``battery_cells_usd`` populated from the motor and battery.
            All other line items use default estimates.

        """
        return ConversionCostModel(
            motor_kit_usd=self.motor.today_price_usd,
            battery_cells_usd=self.battery.cell_cost_usd,
        )
