"""Battery cell and pack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CellSpec:
    """A single battery cell."""

    chemistry: str
    """Cell chemistry string, e.g. 'LiFePO4'."""

    manufacturer: str
    """Cell manufacturer name."""

    model: str
    """Manufacturer model or part number."""

    nominal_voltage_v: float
    """Nominal cell voltage in volts."""

    capacity_ah: float
    """Rated cell capacity in amp-hours."""

    mass_kg: float
    """Cell mass in kilograms."""

    continuous_c: float
    """Continuous discharge rate as a C-rate multiplier."""

    peak_c: float
    """Peak discharge rate as a C-rate multiplier."""

    max_charge_c: float
    """Maximum charge rate as a C-rate multiplier."""

    usable_soc_window: float
    """Fraction of nominal capacity available within the usable SOC window."""

    today_price_usd_per_cell: float
    """Current sourced or assumed price per cell in USD."""

    notes: str = ""
    """Free-text source or assumption notes."""


EVE_LF100 = CellSpec(
    chemistry="LiFePO4",
    manufacturer="EVE",
    model="LF100",
    nominal_voltage_v=3.2,
    capacity_ah=100.0,
    mass_kg=1.92,
    continuous_c=1.0,
    peak_c=2.0,
    max_charge_c=0.5,
    usable_soc_window=0.85,
    today_price_usd_per_cell=33,
    notes="Docan Energy",
)

EVE_LF173 = CellSpec(
    chemistry="LiFePO4",
    manufacturer="EVE",
    model="LF173",
    nominal_voltage_v=3.2,
    capacity_ah=173.0,
    mass_kg=3.19,
    continuous_c=1.0,
    peak_c=2.0,
    max_charge_c=0.5,
    usable_soc_window=0.85,
    today_price_usd_per_cell=120,
)


EVE_LF230 = CellSpec(
    chemistry="LiFePO4",
    manufacturer="EVE",
    model="LF230",
    nominal_voltage_v=3.2,
    capacity_ah=230.0,
    mass_kg=4.11,
    continuous_c=1.0,
    peak_c=2.0,
    max_charge_c=0.5,
    usable_soc_window=0.85,
    today_price_usd_per_cell=79.0,
    notes="18650 Battery Store price used",
)

EVE_LF280K = CellSpec(
    chemistry="LiFePO4",
    manufacturer="EVE",
    model="LF280K",
    nominal_voltage_v=3.2,
    capacity_ah=280.0,
    mass_kg=5.42,
    continuous_c=1.0,
    peak_c=2.0,
    max_charge_c=0.5,
    usable_soc_window=0.85,
    today_price_usd_per_cell=299.99 / 4.0,
    notes="IMR Batteries pack-of-4 price used",
)

CATL_271 = CellSpec(
    chemistry="LiFePO4",
    manufacturer="CATL",
    model="271Ah",
    nominal_voltage_v=3.2,
    capacity_ah=271.0,
    mass_kg=5.30,  # placeholder mass; confirm from actual supplier
    continuous_c=1.0,
    peak_c=3.0,
    max_charge_c=0.5,
    usable_soc_window=0.85,
    today_price_usd_per_cell=217.0,
    notes="Retail listing price used",
)

CALB_CA180FI = CellSpec(
    chemistry="LiFePO4",
    manufacturer="CALB",
    model="CA180FI",
    nominal_voltage_v=3.2,
    capacity_ah=180.0,
    mass_kg=5.60,
    continuous_c=1.0,
    peak_c=2.0,
    max_charge_c=0.5,
    usable_soc_window=0.85,
    today_price_usd_per_cell=50.0,
    notes="Battery Hookup 1-cycle surplus benchmark used",
)

TESLA_S = CellSpec(
    chemistry="Li-ion",
    manufacturer="Tesla",
    model="Model S",
    nominal_voltage_v=22.8,
    capacity_ah=232,
    mass_kg=25,
    continuous_c=1.0,
    peak_c=750 / 232,
    max_charge_c=0.5,
    usable_soc_window=0.85,
    today_price_usd_per_cell=1380,
)

CELLS: list[CellSpec] = [
    EVE_LF100,
    EVE_LF173,
    EVE_LF230,
    EVE_LF280K,
    CATL_271,
    CALB_CA180FI,
    # TESLA_S,
]


@dataclass
class BatteryPack:
    """An arrangement of Cells."""

    cell: CellSpec
    """Cell specification used for every cell in this pack."""

    series_cells: int
    """Number of cells wired in series."""

    parallel_strings: int = 1
    """Number of parallel strings."""

    pack_overhead_mass_kg: float = 35.0
    """Additional mass for enclosures, busbars, and wiring in kilograms."""

    @property
    def nominal_voltage_v(self) -> float:
        """Nominal pack voltage in volts."""
        return self.cell.nominal_voltage_v * self.series_cells

    @property
    def capacity_ah(self) -> float:
        """Pack capacity in amp-hours."""
        return self.cell.capacity_ah * self.parallel_strings

    @property
    def nominal_energy_kwh(self) -> float:
        """Nominal pack energy in kilowatt-hours."""
        return self.nominal_voltage_v * self.capacity_ah / 1000.0

    @property
    def usable_energy_kwh(self) -> float:
        """Usable pack energy in kilowatt-hours, accounting for SOC window."""
        return self.nominal_energy_kwh * self.cell.usable_soc_window

    @property
    def cell_count(self) -> int:
        """Total number of cells in the pack."""
        return self.series_cells * self.parallel_strings

    @property
    def cell_mass_kg(self) -> float:
        """Total mass of all cells in kilograms."""
        return self.cell_count * self.cell.mass_kg

    @property
    def total_pack_mass_kg(self) -> float:
        """Total pack mass including overhead in kilograms."""
        return self.cell_mass_kg + self.pack_overhead_mass_kg

    @property
    def continuous_pack_current_a(self) -> float:
        """Continuous discharge current in amps."""
        return self.capacity_ah * self.cell.continuous_c

    @property
    def peak_pack_current_a(self) -> float:
        """Peak discharge current in amps."""
        return self.capacity_ah * self.cell.peak_c

    @property
    def max_regen_charge_current_a(self) -> float:
        """Maximum regenerative charge current in amps."""
        return self.capacity_ah * self.cell.max_charge_c

    @property
    def cell_cost_usd(self) -> float:
        """Total cost of all cells in USD."""
        return self.cell_count * self.cell.today_price_usd_per_cell

    def print_summary(self) -> None:
        """Print a formatted summary of this battery pack."""
        print("=== BATTERY PACK SUMMARY ===")
        print(
            f"Cell:                         {self.cell.manufacturer} {self.cell.model}",
        )
        config = f"{self.series_cells}s{self.parallel_strings}p"
        print(f"Configuration:                {config}")
        print(f"Nominal voltage:              {self.nominal_voltage_v:.1f} V")
        print(f"Capacity:                     {self.capacity_ah:.1f} Ah")
        print(f"Nominal energy:               {self.nominal_energy_kwh:.2f} kWh")
        print(f"Usable energy:                {self.usable_energy_kwh:.2f} kWh")
        print(f"Cell count:                   {self.cell_count}")
        print(f"Cell mass total:              {self.cell_mass_kg:.1f} kg")
        print(f"Pack overhead mass:           {self.pack_overhead_mass_kg:.1f} kg")
        print(f"Total pack mass:              {self.total_pack_mass_kg:.1f} kg")
        print(f"Continuous discharge limit:   {self.continuous_pack_current_a:.0f} A")
        print(f"Peak discharge limit:         {self.peak_pack_current_a:.0f} A")
        print(f"Max regen current:            {self.max_regen_charge_current_a:.0f} A")
        print(f"Battery cell cost:            ${self.cell_cost_usd:,.0f}")
        print()
