"""Conversion cost model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConversionCostModel:
    """Itemised parts cost estimate for the EV conversion.

    Only ``motor_kit_usd`` and ``battery_cells_usd`` are required; the
    remaining fields default to rough sourced estimates and should be updated
    as quotes come in.
    """

    motor_kit_usd: float
    """Motor, controller, and associated kit price in USD."""

    battery_cells_usd: float
    """Total cell purchase price in USD."""

    adapter_plate_and_coupler_usd: float = 800.0
    """Custom adapter plate and shaft coupler."""

    charger_usd: float = 600.0
    """On-board charger unit."""

    bms_usd: float = 400.0
    """Battery management system."""

    dc_dc_converter_usd: float = 200.0
    """DC-DC converter (HV to 12 V)."""

    charge_inlet_and_port_usd: float = 150.0
    """Charge inlet socket and port hardware."""

    contactors_fuses_disconnect_usd: float = 300.0
    """Main contactors, fuses, and manual service disconnect."""

    cabling_lugs_busbars_usd: float = 250.0
    """High-voltage cabling, lugs, and busbars."""

    cooling_and_pumps_usd: float = 300.0
    """Coolant pump, hoses, and heat exchanger."""

    vacuum_pump_and_brake_assist_usd: float = 200.0
    """Vacuum pump and brake booster assist."""

    motor_mounts_and_battery_boxes_usd: float = 500.0
    """Fabricated motor mounts and battery enclosures."""

    instrumentation_and_misc_usd: float = 300.0
    """Instrumentation, wiring harness, and miscellaneous hardware."""

    def print_summary(self) -> None:
        """Print a formatted itemised cost summary."""
        print("=== CONVERSION COST SUMMARY ===")
        print(f"Motor kit:                    ${self.motor_kit_usd:,.0f}")
        print(f"Battery cells:                ${self.battery_cells_usd:,.0f}")
        print(
            f"Adapter + coupler:            ${self.adapter_plate_and_coupler_usd:,.0f}",
        )
        print(f"Charger:                      ${self.charger_usd:,.0f}")
        print(f"BMS:                          ${self.bms_usd:,.0f}")
        print(f"DC-DC converter:              ${self.dc_dc_converter_usd:,.0f}")
        print(f"Charge inlet + port:          ${self.charge_inlet_and_port_usd:,.0f}")
        print(
            f"Contactors/fuses/disconnect:  "
            f"${self.contactors_fuses_disconnect_usd:,.0f}",
        )
        print(f"Cabling/lugs/busbars:         ${self.cabling_lugs_busbars_usd:,.0f}")
        print(f"Cooling/pumps:                ${self.cooling_and_pumps_usd:,.0f}")
        print(
            f"Brake vacuum assist:          "
            f"${self.vacuum_pump_and_brake_assist_usd:,.0f}",
        )
        print(
            f"Mounts + battery boxes:       "
            f"${self.motor_mounts_and_battery_boxes_usd:,.0f}",
        )
        print(
            f"Instrumentation + misc:       ${self.instrumentation_and_misc_usd:,.0f}",
        )
        print(f"Estimated parts subtotal:     ${self.subtotal_parts_usd():,.0f}")
        print()

    def subtotal_parts_usd(self) -> float:
        """Return the sum of all line-item costs in USD.

        Returns:
            Total estimated parts cost in USD.

        """
        return (
            self.motor_kit_usd
            + self.battery_cells_usd
            + self.adapter_plate_and_coupler_usd
            + self.charger_usd
            + self.bms_usd
            + self.dc_dc_converter_usd
            + self.charge_inlet_and_port_usd
            + self.contactors_fuses_disconnect_usd
            + self.cabling_lugs_busbars_usd
            + self.cooling_and_pumps_usd
            + self.vacuum_pump_and_brake_assist_usd
            + self.motor_mounts_and_battery_boxes_usd
            + self.instrumentation_and_misc_usd
        )
