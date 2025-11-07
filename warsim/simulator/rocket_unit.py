# FILE: warsim/simulator/rocket_unit.py (Complete and Corrected)

"""
    A PAC-3 Missile Unit
"""
from datetime import datetime
from typing import List
import numpy as np
from scipy.interpolate import interp1d

# --- MODIFICATION: Use relative imports for sibling modules ---
from .cmano_simulator import Unit, Position, Event, CmanoSimulator, units_distance_km, UnitDestroyedEvent
from utils.angles import signed_heading_diff # This remains absolute, which is correct.

class Rocket(Unit):
    max_deg_sec = 10
    speed_profile_time = np.array([0, 10, 20, 30])
    speed_profile_knots = np.array([500, 2000, 1400, 600])
    speed_profile = interp1d(speed_profile_time, speed_profile_knots, kind='quadratic', assume_sorted=True,
                             bounds_error=False, fill_value=(500, 600))

    def __init__(self, position: Position, heading: float, firing_time: datetime, target: Unit, source: Unit, friendly_check: bool = True):
        self.speed = Rocket.speed_profile(0)
        super().__init__("Rocket", position, heading, self.speed)
        self.new_heading = heading
        self.firing_time = firing_time
        self.target = target
        self.source = source
        self.friendly_check = friendly_check

    def set_heading(self, new_heading: float):
        if new_heading >= 360 or new_heading < 0:
            raise Exception(f"Rocket.set_heading Heading must be in [0, 360), got {new_heading}")
        self.new_heading = new_heading

    def update(self, tick_secs: float, sim: CmanoSimulator) -> List[Event]:
        # Check if the target has been hit
        if sim.unit_exists(self.target.id) and units_distance_km(self, self.target) < 1:
            sim.remove_unit(self.id)
            sim.remove_unit(self.target.id)
            return [UnitDestroyedEvent(self, self.source, self.target)]
        
        # Check for friendly fire
        if self.friendly_check:
            for i in range(1, sim.num_units + 1):
                if i != self.source.id and sim.unit_exists(i):
                    friendly_unit = sim.get_unit(i)
                    if units_distance_km(self, friendly_unit) < 1:
                        sim.remove_unit(self.id)
                        sim.remove_unit(friendly_unit.id)
                        return [UnitDestroyedEvent(self, self.source, friendly_unit)]

        # Check if end of life is reached
        life_time = (sim.utc_time - self.firing_time).seconds
        if life_time > Rocket.speed_profile_time[-1]: # Use the last time point
            sim.remove_unit(self.id)
            return []

        # Update heading
        if self.heading != self.new_heading:
            delta = signed_heading_diff(self.heading, self.new_heading)
            max_deg = Rocket.max_deg_sec * tick_secs
            self.heading = self.new_heading if abs(delta) <= max_deg else (self.heading + (max_deg if delta >= 0 else -max_deg)) % 360

        # Update speed
        self.speed = Rocket.speed_profile(life_time)

        return super().update(tick_secs, sim)