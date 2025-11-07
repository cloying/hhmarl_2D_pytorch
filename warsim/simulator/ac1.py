# FILE: warsim/simulator/ac1.py (Complete and Corrected)

"""
    A Rafale airplane unit with Rockets.
"""
from typing import List
import numpy as np

# --- MODIFICATION: Use relative imports for sibling modules ---
from .cmano_simulator import Unit, Position, Event, CmanoSimulator, units_bearing, UnitDestroyedEvent, units_distance_km
from .rocket_unit import Rocket
# This is correct because utils is at the project root.
from utils.angles import signed_heading_diff, sum_angles


class Rafale(Unit):
    # (The content of the class is unchanged from the previous correct version)
    max_deg_sec = 5;
    min_speed_knots = 0;
    max_speed_knots = 900;
    max_knots_sec = 35;
    cannon_range_km = 2.0;
    cannon_width_deg = 10;
    cannon_max_time_sec = 200;
    cannon_burst_time_sec = 5;
    cannon_hit_prob = 0.75;
    max_missiles = 5;
    missile_range_km = 111;
    missile_width_deg = 120;
    aircraft_type = 1

    def __init__(self, position: Position, heading: float, speed: float, group: str, friendly_check: bool = True):
        super().__init__("Rafale", position, heading, speed)
        self.new_heading = heading;
        self.new_speed = speed;
        self.max_speed = Rafale.max_speed_knots
        self.cannon_remain_secs = Rafale.cannon_max_time_sec;
        self.cannon_current_burst_secs = 0;
        self.cannon_max = Rafale.cannon_max_time_sec
        self.actual_missile = None;
        self.missile_remain = Rafale.max_missiles;
        self.rocket_max = Rafale.max_missiles
        self.friendly_check = friendly_check;
        self.group = group;
        self.ac_type = Rafale.aircraft_type

    def set_heading(self, new_heading: float):
        self.new_heading = new_heading

    def set_speed(self, new_speed: float):
        self.new_speed = new_speed

    def fire_cannon(self):
        self.cannon_current_burst_secs = min(self.cannon_remain_secs, Rafale.cannon_burst_time_sec)

    def fire_missile(self, ag_unit: Unit, opp_unit: Unit, sim: CmanoSimulator):
        if not self.actual_missile and self.missile_remain > 0:
            if units_distance_km(self, opp_unit) <= Rafale.missile_range_km and self._angle_in_radar_range(
                    units_bearing(self, opp_unit)):
                missile = Rocket(self.position.copy(), self.heading, sim.utc_time, opp_unit, ag_unit,
                                 self.friendly_check)
                sim.add_unit(missile)
                self.actual_missile = missile
                self.missile_remain = max(0, self.missile_remain - 1)

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = Rafale.max_deg_sec * tick_secs
            self.heading = self.new_heading if abs(delta) <= max_deg else (self.heading + (
                max_deg if delta >= 0 else -max_deg)) % 360

        # Update speed
        if self.speed != self.new_speed:
            delta = self.new_speed - self.speed
            max_delta = Rafale.max_knots_sec * tick_secs
            self.speed = self.new_speed if abs(delta) <= max_delta else self.speed + (
                max_delta if delta >= 0 else -max_delta)

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
                            if sim.random_generator.random() < (
                                    Rafale.cannon_hit_prob / (Rafale.cannon_burst_time_sec / tick_secs)):
                                sim.remove_unit(unit.id)
                                events.append(UnitDestroyedEvent(self, self, unit))

        # Update missile
        if self.actual_missile:
            if not sim.unit_exists(self.actual_missile.id):
                self.actual_missile = None
            else:
                heading_noise = sim.random_generator.uniform(0.95, 1.05)
                heading = np.clip(self.actual_missile.heading * heading_noise, 0, 359)
                self.actual_missile.set_heading(heading)

        events.extend(super().update(tick_secs, sim))
        return events

    def _unit_in_cannon_range(self, u: Unit) -> bool:
        if units_distance_km(self, u) < Rafale.cannon_range_km:
            delta = abs(signed_heading_diff(self.heading, units_bearing(self, u)))
            return delta <= Rafale.cannon_width_deg / 2.0
        return False

    def _angle_in_radar_range(self, angle: float) -> bool:
        delta = abs(signed_heading_diff(sum_angles(self.heading, Rafale.missile_width_deg / 2), angle))
        return delta <= Rafale.missile_width_deg / 2.0