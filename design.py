"""MG Midget EV - design study and pack comparison."""

from __future__ import annotations

from mg.battery import BatteryPack, CALB_CA180FI, CATL_271AH, EVE_LF230, EVE_LF280K
from mg.motor import HYPER9_144V
from mg.vehicle import Vehicle

if __name__ == "__main__":
    battery_options: dict[str, BatteryPack] = {
        "LF280K_45s1p": BatteryPack(cell=EVE_LF280K, series_cells=45),
        "LF230_45s1p": BatteryPack(cell=EVE_LF230, series_cells=45),
        "CATL271_45s1p": BatteryPack(cell=CATL_271AH, series_cells=45),
        "CALB180_45s1p": BatteryPack(cell=CALB_CA180FI, series_cells=45),
    }

    for pack_name in ("LF230_45s1p",):
        pack = battery_options[pack_name]
        vehicle = Vehicle(battery=pack, motor=HYPER9_144V)

        print("=" * 78)
        print(f"PACK OPTION: {pack_name}")
        vehicle.battery.print_summary()
        vehicle.print_summary()

        cost = vehicle.build_cost_model()
        cost.print_summary()

        vehicle.print_top_speed_table()
        vehicle.print_rpm_table([10, 20, 30, 40, 50, 60, 70])

        vehicle.cruise_case(45.0, 0.0).print_summary("Flat 45 mph")
        vehicle.cruise_case(25.0, 10.0).print_summary("10% uphill 25 mph")
        vehicle.cruise_case(45.0, 10.0).print_summary("10% uphill 45 mph")
        vehicle.cruise_case(20.0, 15.0).print_summary("15% uphill 20 mph")
        vehicle.cruise_case(35.0, -10.0).print_summary("10% downhill 35 mph")

        vehicle.print_gradeability_table(10.0)
        vehicle.print_gradeability_table(20.0)
        vehicle.print_gradeability_table(30.0)
        vehicle.print_gradeability_table(45.0)

        vehicle.estimate_0_to_n().print_summary()

    # Quick summary table
    print("=" * 78)
    print("=== PACK COMPARISON SUMMARY ===")
    print(
        "Name".ljust(15),
        "Weight_lb".rjust(10),
        "Usable_kWh".rjust(12),
        "Cells_$".rjust(10),
        "Build_$".rjust(10),
        "45mph_12%_A".rjust(12),
        "45mph_12%_C".rjust(12),
    )

    for pack_name, pack in battery_options.items():
        vehicle = Vehicle(battery=pack, motor=HYPER9_144V)
        cost = vehicle.build_cost_model()
        case = vehicle.cruise_case(45.0, 12.0)
        total_lb = vehicle.total_mass_kg * 2.20462
        print(
            pack_name.ljust(15),
            f"{total_lb:10.0f}",
            f"{pack.usable_energy_kwh:12.1f}",
            f"{pack.cell_cost_usd:10.0f}",
            f"{cost.subtotal_parts_usd():10.0f}",
            f"{case.pack_current_a:12.0f}",
            f"{case.discharge_c:12.2f}",
        )
