# FILE: warsim/simulator/cmano_simulator.py (Complete and Corrected)

"""
    CmanoSimulator is a simulator of the essential characteristic of CMANO.
"""
from __future__ import annotations

from abc import ABC
from datetime import datetime, timedelta
from typing import Callable, List, Dict

# This file should NOT import the 'random' module.

# --- Local Project Imports ---
# (Assuming utils is at the project root)
from utils.geodesics import geodetic_direct, geodetic_distance_km, geodetic_bearing_deg

# --- Constants
knots_to_ms = 0.514444

# --- Classes
class Position:
    # ... (rest of the class is unchanged)
    def __init__(self, lat: float, lon: float, alt: float):
        self.lat = lat
        self.lon = lon
        self.alt = alt
    def copy(self) -> Position:
        return Position(self.lat, self.lon, self.alt)

class Event(ABC):
    # ... (rest of the class is unchanged)
    def __init__(self, name, origin: Unit):
        self.name = name
        self.origin = origin
    def __str__(self):
        return f"{self.origin.type}[{self.origin.id}].{self.name}"

class UnitDestroyedEvent(Event):
    # ... (rest of the class is unchanged)
    def __init__(self, origin: Unit, unit_killer: Unit, unit_destroyed: Unit):
        super().__init__("UnitDestroyedEvent", origin)
        self.unit_killer = unit_killer
        self.unit_destroyed = unit_destroyed
    def __str__(self):
        return super().__str__() + f"({self.unit_killer.type}{[self.unit_killer.id]} -> {self.unit_destroyed.type}{[self.unit_destroyed.id]})"

class Unit(ABC):
    # ... (rest of the class is unchanged)
    def __init__(self, type: str, position: Position, heading: float, speed_knots: float):
        if heading >= 360 or heading < 0: raise Exception(f"Unit.__init__: bad heading {heading}")
        self.type = type
        self.position = position
        self.heading = heading
        self.speed = speed_knots
        self.id = None
    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        if self.speed > 0:
            d = geodetic_direct(self.position.lat, self.position.lon, self.heading, self.speed * knots_to_ms * tick_secs)
            self.position.lat, self.position.lon = d[0], d[1]
        return []
    def to_string(self) -> str:
        return f"{self.type}[{self.id}]: p=({self.position.lat:.4f}, {self.position.lon:.4f}, {self.position.alt:.4f}) h=({self.heading:.4f}) s=({self.speed:.4f})"

class CmanoSimulator:
    # --- MODIFICATION: Accept a random number generator instead of creating one ---
    def __init__(self, random_generator, utc_time=datetime.now(), tick_secs=1, num_units=0, num_opp_units=0):
        self.active_units: Dict[int, Unit] = {}
        self.trace_record_units = {}
        self.utc_time = utc_time
        self.tick_secs = tick_secs
        
        # This is the safe, Gymnasium-provided numpy random generator.
        self.random_generator = random_generator
        
        self._next_unit_id = 1
        self.status_text = None
        self.num_units = num_units
        self.num_opp_units = num_opp_units

    # --- The rest of the class is unchanged ---
    def add_unit(self, unit: Unit) -> int:
        self.active_units[self._next_unit_id] = unit
        unit.id = self._next_unit_id
        self._next_unit_id += 1
        return self._next_unit_id - 1
    def remove_unit(self, unit_id: int):
        if self.unit_exists(unit_id): del self.active_units[unit_id]
    def get_unit(self, unit_id: int) -> Unit:
        return self.active_units[unit_id]
    def unit_exists(self, unit_id: int) -> bool:
        return unit_id in self.active_units
    def record_unit_trace(self, unit_id: int):
        if unit_id not in self.active_units: raise Exception(f"Unit.record_unit_trace(): unknown unit {unit_id}")
        if unit_id not in self.trace_record_units:
            self.trace_record_units[unit_id] = []
            self._store_unit_state(unit_id)
    def do_tick(self) -> List[Event]:
        events = []
        for unit in list(self.active_units.values()):
            event = unit.update(self.tick_secs, self)
            events.extend(event)
        self.utc_time += timedelta(seconds=self.tick_secs)
        for unit_id in self.trace_record_units.keys():
            self._store_unit_state(unit_id)
        return events
    def _store_unit_state(self, unit_id):
        if self.unit_exists(unit_id):
            unit = self.active_units[unit_id]
            self.trace_record_units[unit_id].append((self.utc_time, unit.position.copy(), unit.heading, unit.speed))

# --- General purpose utilities
def units_distance_km(unit_a: Unit, unit_b: Unit) -> float:
    return geodetic_distance_km(unit_a.position.lat, unit_a.position.lon, unit_b.position.lat, unit_b.position.lon)
def units_bearing(unit_from: Unit, unit_to: Unit) -> float:
    return geodetic_bearing_deg(unit_from.position.lat, unit_from.position.lon, unit_to.position.lat, unit_to.position.lon)