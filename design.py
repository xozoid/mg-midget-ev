"""MG Midget EV - design study and pack comparison."""

from __future__ import annotations

from mg.battery import BatteryPack, EVE_LF100, EVE_LF230
from mg.motor import HYPER9_144V
from mg.vehicle import Vehicle

if __name__ == "__main__":
    packs = {
        "LF100; 45s2p": BatteryPack(EVE_LF100, 45, 2),
        "LF100; 36s2p": BatteryPack(EVE_LF100, 36, 2),
        "LF230; 45s1p": BatteryPack(EVE_LF230, 45),
        "LF230; 36s1p": BatteryPack(EVE_LF230, 36),
    }
    vehicle = Vehicle(
        packs["LF100; 45s2p"],
        motor=HYPER9_144V,
    )

    vehicle.battery.print_summary()
    vehicle.print_summary()

    cost = vehicle.build_cost_model()
    cost.print_summary()

    vehicle.print_top_speed_table()

    vehicle.cruise_case(45.0, 0.0).print_summary("Flat 45 mph")
    vehicle.cruise_case(25.0, 10.0).print_summary("10% uphill 25 mph")
    vehicle.cruise_case(45.0, 10.0).print_summary("10% uphill 45 mph")
    vehicle.cruise_case(20.0, 15.0).print_summary("15% uphill 20 mph")
    vehicle.cruise_case(35.0, -10.0).print_summary("10% downhill 35 mph")

    vehicle.print_gradeability_table(10.0)
    vehicle.print_gradeability_table(20.0)
    vehicle.print_gradeability_table(30.0)
    vehicle.print_gradeability_table(45.0)
