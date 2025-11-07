# FILE: warsim/simulator/ac2.py (Complete and Corrected)

"""
    A modified Rafale airplane unit (cannon only)
"""
from typing import List
import numpy as np

# --- MODIFICATION: Use relative imports for sibling modules ---
from .cmano_simulator import Unit, Position, Event, CmanoSimulator, units_bearing, UnitDestroyedEvent, units_distance_km
# This is correct.
from utils.angles import signed_heading_diff

class RafaleLong(Unit):
    # (The content of the class is unchanged from the previous correct version)
    max_deg_sec = 3.5; min_speed_knots = 0; max_speed_knots = 600; max_knots_sec = 28;
    cannon_range_km = 4.5; cannon_width_deg = 7; cannon_max_time_sec = 200;
    cannon_burst_time_sec = 3; cannon_hit_prob = 0.9; aircraft_type = 2

    def __init__(self, position: Position, heading: float, speed: float, group:str, friendly_check: bool = True):
        super().__init__("RafaleLong", position, heading, speed)
        self.new_heading = heading; self.new_speed = speed; self.max_speed = RafaleLong.max_speed_knots
        self.cannon_remain_secs = RafaleLong.cannon_max_time_sec; self.cannon_current_burst_secs = 0; self.cannon_max = RafaleLong.cannon_max_time_sec
        self.actual_missile = None; self.missile_remain = 0; self.rocket_max = 0
        self.friendly_check = friendly_check; self.group = group; self.ac_type = RafaleLong.aircraft_type

    def set_heading(self, new_heading: float): self.new_heading = new_heading
    def set_speed(self, new_speed: float): self.new_speed = new_speed
    def fire_cannon(self): self.cannon_current_burst_secs = min(self.cannon_remain_secs, RafaleLong.cannon_burst_time_sec)

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = RafaleLong.max_deg_sec * tick_secs
            self.heading = self.new_heading if abs(delta) <= max_deg else (self.heading + (max_deg if delta >= 0 else -max_deg)) % 360

        # Update speed
        if self.speed != self.new_speed:
            delta = self.new_speed - self.speed
            max_delta = RafaleLong.max_knots_sec * tick_secs
            self.speed = self.new_speed if abs(delta) <= max_delta else self.speed + (max_delta if delta >= 0 else -max_delta)

        # Update cannon
        events = []
        if self.cannon_current_burst_secs > 0:
            self.cannon_current_burst_secs = max(self.cannon_current_burst_secs - tick_secs, 0.0)
            self.cannon_remain_secs = max(self.cannon_remain_secs - tick_secs, 0.0)
            for unit in list(sim.active_units.values()):
                if unit.id != self.id and unit.id <= (sim.num_units + sim.num_opp_units):
                    is_opponent = (self.group == "agent" and unit.id > sim.num_units) or \
                                  (self.group == "opp" and unit.id <= sim.num_units)
                    if self.friendly_check or is_opponent:
                        if unit.type in ["RafaleLong", "Rafale"] and self._unit_in_cannon_range(unit):
                            if sim.random_generator.random() < (RafaleLong.cannon_hit_prob / (RafaleLong.cannon_burst_time_sec / tick_secs)):
                                sim.remove_unit(unit.id)
                                events.append(UnitDestroyedEvent(self, self, unit))
        
        events.extend(super().update(tick_secs, sim))
        return events

    def _unit_in_cannon_range(self, u: Unit) -> bool:
        if units_distance_km(self, u) < RafaleLong.cannon_range_km:
            delta = abs(signed_heading_diff(self.heading, units_bearing(self, u)))
            return delta <= RafaleLong.cannon_width_deg / 2.0
        return False