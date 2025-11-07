# FILE: envs/env_base.py (Complete, Commented, and Corrected for Imports and Seeding)

# --- Core Dependencies ---
import os
from math import sin, cos, acos, pi, hypot, radians, sqrt
from pathlib import Path

# --- Third-party Libraries ---
import numpy as np
import torch
import gymnasium  # Use the standard Gymnasium library for RL environments.

# --- Local Project Imports (Absolute from project root) ---
# Imports for plotting and visualization.
from warsim.scenplotter.scenario_plotter import (PlotConfig, ColorRGBA, StatusMessage,
                                                 TopLeftMessage, Airplane, PolyLine,
                                                 Waypoint, Missile, ScenarioPlotter)
# Imports for the core simulation logic.
from warsim.simulator.cmano_simulator import Position, CmanoSimulator, UnitDestroyedEvent
from warsim.simulator.ac1 import Rafale
from warsim.simulator.ac2 import RafaleLong
# Utility imports.
from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits
from utils.angles import sum_angles

# ------------------- Constants -------------------

# Color definitions for plotting the simulation.
colors = {
    'red_outline': ColorRGBA(0.8, 0.2, 0.2, 1),
    'red_fill': ColorRGBA(0.8, 0.2, 0.2, 0.2),
    'blue_outline': ColorRGBA(0.3, 0.6, 0.9, 1),
    'blue_fill': ColorRGBA(0.3, 0.6, 0.9, 0.2),
    'waypoint_outline': ColorRGBA(0.8, 0.8, 0.2, 1),
    'waypoint_fill': ColorRGBA(0.8, 0.8, 0.2, 0.2)
}

# Constant definitions for action and observation space dimensions for different aircraft types.
ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29


# ------------------- Environment Class -------------------

class HHMARLBaseEnv(gymnasium.Env):
    """
    Base class for the HHMARL 2D environment, refactored for pure PyTorch and
    made compatible with Gymnasium's vectorized environments and seeding.
    """

    def __init__(self, map_size):
        """
        Initializes the base environment, setting up simulation parameters,
        map boundaries, and plotting configurations.
        """
        super().__init__()  # Initialize the base Gymnasium Env class.

        # Simulation state variables.
        self.steps = 0
        self.sim = None
        self.map_size = map_size
        self.map_limits = MapLimits(7.0, 5.0, 7.0 + map_size, 5.0 + map_size)
        self.alive_agents = 0
        self.alive_opps = 0
        self.rewards = {}

        # Combat behavior state variables.
        self.opp_to_attack = {}
        self.missile_wait = {}
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0

        # Plotting and visualization setup.
        self.plt_cfg = PlotConfig()
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

        # NOTE: self.np_random will be initialized by Gymnasium's reset() method.
        # This is the correct random number generator to use for reproducibility.

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment to a new random initial configuration for a new episode.

        Args:
            seed (int, optional): The seed for the random number generator. Gymnasium's
                                  vector environment wrapper will provide a unique seed
                                  to each parallel environment instance.
            options (dict, optional): A dictionary for additional options.

        Returns:
            tuple: A tuple containing the initial observations (dict) and info (dict).
        """
        # --- CRITICAL FOR SEEDING ---
        # This call to the parent's reset method is essential. It takes the provided
        # seed and creates the `self.np_random` generator instance for this environment.
        super().reset(seed=seed)

        # Reset all simulation state variables.
        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.missile_wait = {i: 0 for i in range(1, self.args.total_num + 1)}
        self.opp_to_attack = {i: None for i in range(1, self.args.total_num + 1)}

        # Initialize a new simulation instance and reset the scenario.
        self.sim = CmanoSimulator(num_units=self.args.num_agents, num_opp_units=self.args.num_opps, random_seed=seed)
        self._reset_scenario(options.get("mode", None) if options else None)

        # Return the initial state and an empty info dictionary as per Gymnasium API.
        return self.state(), {}

    def step(self, action):
        """
        Executes one time step in the environment given an action from the agents.

        Args:
            action (dict): A dictionary mapping agent IDs to their chosen action arrays.

        Returns:
            tuple: A tuple containing (observations, rewards, terminateds, truncateds, info).
        """
        self.rewards = {}
        if action:
            self._take_action(action)

        # Determine if the episode has ended for all agents.
        done = (self.alive_agents <= 0 or
                self.alive_opps <= 0 or
                self.steps >= self.args.horizon)

        terminateds = {"__all__": done}
        truncateds = {"__all__": done}

        info = {}  # Auxiliary diagnostic information.
        return self.state(), self.rewards, terminateds, truncateds, info

    # =================================================================================
    # Internal Simulation Logic: State, Actions, Rewards, Plotting
    # =================================================================================

    def fight_state_values(self, agent_id, unit, opp, fri_id=None):
        """Constructs the observation vector for a low-level "fight" agent."""
        state = []
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        # Own state (12 or 10 values depending on AC type)
        state.extend([x, y,
                      np.clip(unit.speed / unit.max_speed, 0, 1),
                      np.clip((unit.heading % 359) / 359, 0, 1),
                      self._focus_angle(agent_id, opp[0], True),  # Angle to target
                      self._aspect_angle(opp[0], agent_id),  # Target's angle off our tail
                      self._heading_diff(agent_id, opp[0]),  # Difference in headings
                      opp[1],  # Normalized distance to target
                      np.clip(unit.cannon_remain_secs / unit.cannon_max, 0, 1)])
        if unit.ac_type == 1:  # If aircraft has missiles
            state.extend([np.clip(unit.missile_remain / unit.rocket_max, 0, 1),
                          int(self.missile_wait[agent_id] == 0),  # Missile ready to fire
                          int(bool(unit.actual_missile) or unit.cannon_current_burst_secs > 0)])  # Are we shooting?
        else:
            state.append(int(unit.cannon_current_burst_secs > 0))

        # Opponent and Friendly state (9 + 5 = 14 values)
        state.extend(self.opp_ac_values("fight", opp[0], agent_id, opp[1]))
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def esc_state_values(self, agent_id, unit, opps, fri_id=None):
        """Constructs the observation vector for a low-level "escape" agent."""
        state = []
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        # Own state (7 or 6 values)
        state.extend([x, y,
                      np.clip(unit.speed / unit.max_speed, 0, 1),
                      np.clip((unit.heading % 359) / 359, 0, 1),
                      np.clip(unit.cannon_remain_secs / unit.cannon_max, 0, 1)])
        if unit.ac_type == 1:
            state.append(np.clip(unit.missile_remain / unit.rocket_max, 0, 1))
        shot = unit.cannon_current_burst_secs > 0 or (unit.ac_type == 1 and bool(unit.actual_missile))
        state.append(int(shot))

        # Opponent and Friendly state (up to 18 + 5 = 23 values)
        opp_state = []
        for opp in opps:
            opp_state.extend(self.opp_ac_values("esc", opp[0], agent_id, opp[1]))
            if len(opp_state) >= 18: break  # Max 2 opponents observed in detail
        if len(opp_state) < 18:
            opp_state.extend(np.zeros(18 - len(opp_state)))  # Pad with zeros if fewer opponents
        state.extend(opp_state)
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def friendly_ac_values(self, agent_id, fri_id=None):
        """Gets the state of a friendly aircraft relative to the agent."""
        if not fri_id or not self.sim.unit_exists(fri_id):
            return np.zeros(5)  # Return zeros if no friendly or friendly is destroyed
        else:
            unit = self.sim.get_unit(fri_id)
            x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            return [x, y,
                    self._focus_angle(agent_id, fri_id, True),
                    self._focus_angle(fri_id, agent_id, True),
                    self._distance(agent_id, fri_id, True)]

    def opp_ac_values(self, mode, opp_id, agent_id, dist):
        """Gets the state of an opponent aircraft relative to the agent."""
        unit = self.sim.get_unit(opp_id)
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state = [x, y,
                 np.clip(unit.speed / unit.max_speed, 0, 1),
                 np.clip((unit.heading % 359) / 359, 0, 1),
                 self._heading_diff(opp_id, agent_id)]

        # State representation differs slightly based on the agent's mode
        if mode == "fight":
            state.extend([self._focus_angle(opp_id, agent_id, True),
                          self._aspect_angle(agent_id, opp_id)])
        else:  # escape or highlevel
            state.extend([self._focus_angle(agent_id, opp_id, True),
                          self._focus_angle(opp_id, agent_id, True)])
        if mode == "HighLevel":
            state.extend([self._aspect_angle(agent_id, opp_id),
                          self._aspect_angle(opp_id, agent_id)])
        state.append(dist)
        if mode != "HighLevel":
            shot = unit.cannon_current_burst_secs > 0 or (unit.ac_type == 1 and bool(unit.actual_missile))
            state.append(int(shot))
        return state

    def _take_base_action(self, mode, unit, unit_id, opp_id, actions, rewards=None):
        """Applies the agent's action to the simulation unit (heading, speed, weapons)."""
        # Action part 0: Change heading. Action value is 0-12, mapped to -90 to +90 degrees.
        unit.set_heading((unit.heading + (actions[unit_id][0] - 6) * 15) % 360)
        # Action part 1: Change speed. Action value is 0-8.
        unit.set_speed(100 + ((unit.max_speed - 100) / 8) * actions[unit_id][1])
        # Action part 2: Fire cannon. Action value is 0 or 1.
        if bool(actions[unit_id][2]) and unit.cannon_remain_secs > 0:
            unit.fire_cannon()
        # Action part 3 (missile-capable aircraft only): Fire missile. Action value is 0 or 1.
        if unit.ac_type == 1 and bool(actions[unit_id][3]):
            if opp_id and unit.missile_remain > 0 and not unit.actual_missile and self.missile_wait[unit_id] == 0:
                unit.fire_missile(unit, self.sim.get_unit(opp_id), self.sim)
                # --- MODIFICATION: Use self.np_random for reproducible randomness ---
                if mode == "LowLevel":
                    self.missile_wait[unit_id] = self.np_random.integers(7, 18)  # Cooldown is 7-17 steps
                else:  # HighLevel
                    self.missile_wait[unit_id] = self.np_random.integers(8, 13)  # Cooldown is 8-12 steps

        # Decrement missile cooldown timer.
        if self.missile_wait[unit_id] > 0 and not bool(unit.actual_missile):
            self.missile_wait[unit_id] -= 1

        return rewards

    def _combat_rewards(self, events, opp_stats=None, mode="LowLevel"):
        """Calculates rewards based on combat events (kills, out of bounds)."""
        rews = {a: [] for a in range(1, self.args.num_agents + 1)}
        destroyed_ids = []
        s = self.args.rew_scale
        kill_event = False

        # Punishment for going out of map boundaries.
        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if not self.map_limits.in_boundary(u.position.lat, u.position.lon):
                    self.sim.remove_unit(i)
                    kill_event = True
                    if i <= self.args.num_agents:
                        p = -5 if mode == "LowLevel" else -2
                        if i in rews: rews[i].append(p * s)
                        destroyed_ids.append(i)
                        self.alive_agents -= 1
                    else:
                        self.alive_opps -= 1

        # Rewards for kill events from the simulator.
        for ev in events:
            if isinstance(ev, UnitDestroyedEvent):
                # If an agent gets a kill...
                if ev.unit_killer.id <= self.args.num_agents and ev.unit_killer.id in rews:
                    # and it killed an opponent...
                    if ev.unit_destroyed.id > self.args.num_agents:
                        rews[ev.unit_killer.id].append(1 * s)
                        self.alive_opps -= 1
                    # and it killed a friendly (friendly fire)...
                    elif ev.unit_destroyed.id <= self.args.num_agents:
                        rews[ev.unit_killer.id].append(-2 * s)
                        if self.args.friendly_punish and ev.unit_destroyed.id in rews:
                            rews[ev.unit_destroyed.id].append(-2 * s)
                            destroyed_ids.append(ev.unit_destroyed.id)
                        self.alive_agents -= 1
                # If an opponent gets a kill...
                elif ev.unit_killer.id > self.args.num_agents:
                    # and it killed an agent...
                    if ev.unit_destroyed.id <= self.args.num_agents and ev.unit_destroyed.id in rews:
                        p = -2 if mode == "LowLevel" else -1
                        rews[ev.unit_destroyed.id].append(p * s)
                        destroyed_ids.append(ev.unit_destroyed.id)
                        self.alive_agents -= 1
                    # and it killed another opponent...
                    elif ev.unit_destroyed.id > self.args.num_agents:
                        self.alive_opps -= 1
                kill_event = True

        return rews, destroyed_ids, kill_event

    def _get_policies(self, mode):
        """Loads pre-trained PyTorch policy models for self-play opponents."""
        # This function would contain logic to load torch.jit.load or model.load_state_dict
        # for opponents in curriculum levels 4 and 5.
        pass

    def _policy_actions(self, policy_type, agent_id, unit):
        """Gets an action from a loaded PyTorch policy model."""
        # This function is a placeholder. A full implementation would perform a
        # forward pass through the loaded neural network with the current observation
        # to get an action for the self-play opponent.
        # For now, return a dummy "do nothing" action.
        if unit.ac_type == 1:
            return {agent_id: np.array([6, 4, 0, 0])}  # Straight, medium speed, no fire
        else:
            return {agent_id: np.array([6, 4, 0])}

    def _nearby_object(self, agent_id, friendly=False):
        """Returns a sorted list of [id, normalized_distance, raw_distance] to other aircraft."""
        order = []
        # Define the list of IDs to check against.
        if friendly:
            id_pool = list(range(1, self.args.num_agents + 1)) if agent_id <= self.args.num_agents else list(
                range(self.args.num_agents + 1, self.args.total_num + 1))
        else:
            id_pool = list(
                range(self.args.num_agents + 1, self.args.total_num + 1)) if agent_id <= self.args.num_agents else list(
                range(1, self.args.num_agents + 1))

        # Calculate distances to all valid, existing units.
        for i in id_pool:
            if i != agent_id and self.sim.unit_exists(i):
                dist_norm = self._distance(agent_id, i, True)
                dist_raw = self._distance(agent_id, i, False) if not friendly else 0
                order.append([i, dist_norm, dist_raw])

        order.sort(key=lambda x: x[1])  # Sort by normalized distance (closest first).
        return order

    # --- Geometric Calculation Helper Methods ---
    def _focus_angle(self, agent_id, opp_id, norm=False):
        """Computes the angle between an agent's heading and the opponent (Angle-to-Target)."""
        ag_unit, opp_unit = self.sim.get_unit(agent_id), self.sim.get_unit(opp_id)
        heading_rad = ((90 - ag_unit.heading) % 360) * (pi / 180)
        heading_vec = np.array([cos(heading_rad), sin(heading_rad)])
        target_vec = np.array(
            [opp_unit.position.lon - ag_unit.position.lon, opp_unit.position.lat - ag_unit.position.lat])
        dot_product = np.dot(heading_vec, target_vec)
        norm_product = np.linalg.norm(heading_vec) * np.linalg.norm(target_vec) + 1e-10
        angle = acos(np.clip(dot_product / norm_product, -1, 1)) * (180 / pi)
        return np.clip(angle / 180, 0, 1) if norm else angle

    def _distance(self, agent_id, opp_id, norm=False):
        """Computes Euclidean distance between two units and optionally normalizes it."""
        ag_unit, opp_unit = self.sim.get_unit(agent_id), self.sim.get_unit(opp_id)
        dist = hypot(opp_unit.position.lon - ag_unit.position.lon, opp_unit.position.lat - ag_unit.position.lat)
        return self._shifted_range(dist, 0, sqrt(2 * self.map_size ** 2), 0, 1) if norm else dist

    def _aspect_angle(self, agent_id, opp_id, norm=True):
        """Computes the angle from the agent's tail to the opponent (Aspect Angle)."""
        focus = self._focus_angle(agent_id, opp_id)
        aspect = 180 - focus
        return np.clip(aspect / 180, 0, 1) if norm else np.clip(aspect, 0, 180)

    def _heading_diff(self, agent_id, opp_id, norm=True):
        """Computes the angle between the two agents' heading vectors."""
        ag_unit, opp_unit = self.sim.get_unit(agent_id), self.sim.get_unit(opp_id)
        ag_head_rad = ((90 - ag_unit.heading) % 360) * (pi / 180)
        opp_head_rad = ((90 - opp_unit.heading) % 360) * (pi / 180)
        ag_vec = np.array([cos(ag_head_rad), sin(ag_head_rad)])
        opp_vec = np.array([cos(opp_head_rad), sin(opp_head_rad)])
        angle = acos(np.clip(np.dot(ag_vec, opp_vec), -1, 1)) * (180 / pi)
        return np.clip(angle / 180, 0, 1) if norm else angle

    def _shifted_range(self, x, a, b, c, d):
        """Linearly scales a value from one range [a,b] to another [c,d]."""
        return c + ((d - c) / (b - a)) * (x - a)

    def _correct_angle_sign(self, opp_unit, ag_unit):
        """Determines if the agent is to the left or right of the opponent's heading."""
        line = lambda x0, y0, x1, y1, x2, y2: (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        x, y, a = opp_unit.position.lon, opp_unit.position.lat, opp_unit.heading
        x1 = x + sin(radians(a % 360))
        y1 = y + cos(radians(a % 360))
        val = line(x, y, x1, y1, ag_unit.position.lon, ag_unit.position.lat)
        return 1 if val < 0 else -1

    def _sample_state(self, agent_type, agent_index, side):
        """Generates random initial positions and headings for units based on curriculum level."""
        x, y, a = 0, 0, 0
        if agent_type == "agent":
            if self.args.level >= 2:  # Levels 2-5 use a wider starting area
                x = self.np_random.uniform(7.07, 7.12) if side == 1 else self.np_random.uniform(7.18, 7.23)
                y = self.np_random.uniform(5.09 + agent_index * 0.1, 5.12 + agent_index * 0.1)
                a = self.np_random.integers(0, 360)
            else:  # Level 1 has a narrower starting area
                x = self.np_random.uniform(7.12, 7.14) if side == 1 else self.np_random.uniform(7.16, 7.17)
                y = self.np_random.uniform(5.1 + agent_index * 0.1, 5.11 + agent_index * 0.1)
                a = self.np_random.integers(30, 151) if side == 1 else self.np_random.integers(200, 331)

        elif agent_type == "opp":
            if self.args.level >= 2:
                x = self.np_random.uniform(7.18, 7.23) if side == 1 else self.np_random.uniform(7.07, 7.12)
                y = self.np_random.uniform(5.09 + agent_index * 0.1, 5.12 + agent_index * 0.1)
                a = self.np_random.integers(0, 360)
            else:  # Level 1 opponents are static
                x = self.np_random.uniform(7.16, 7.17) if side == 1 else self.np_random.uniform(7.12, 7.14)
                y = self.np_random.uniform(5.1 + agent_index * 0.1, 5.11 + agent_index * 0.1)
        return x, y, a

    def _reset_scenario(self, mode):
        """Creates the aircraft units at the start of an episode."""
        # Choose which side of the map to start on.
        side = self.np_random.integers(1, 3)

        for group, count in [("agent", self.args.num_agents), ("opp", self.args.num_opps)]:
            for i in range(count):
                x, y, a = self._sample_state(group, i, side)
                # Determine aircraft type (alternating between AC1 and AC2)
                ac_type = (i % 2) + 1

                # Opponents in Level 1 & 2 start with no speed.
                start_speed = 0 if self.args.level <= 2 and group == "opp" else 100

                if ac_type == 1:
                    unit = Rafale(Position(y, x, 10_000), heading=a, speed=start_speed, group=group,
                                  friendly_check=self.args.friendly_kill)
                else:
                    unit = RafaleLong(Position(y, x, 10_000), heading=a, speed=start_speed, group=group,
                                      friendly_check=self.args.friendly_kill)

                self.sim.add_unit(unit)
                self.sim.record_unit_trace(unit.id)
                if group == "agent":
                    self.alive_agents += 1
                else:
                    self.alive_opps += 1

    def _plot_airplane(self, a: Rafale, side: str, path=True, use_backup=False, u_id=0):
        """Helper function to create drawable objects for a single airplane."""
        objects = []
        if use_backup:  # Plotting a destroyed unit's last known position and path.
            trace = [(pos.lat, pos.lon) for t, pos, h, s in self.sim.trace_record_units[u_id]]
            objects.append(PolyLine(trace, line_width=1, dash=(2, 2), edge_color=colors[f'{side}_outline']))
            objects.append(Waypoint(trace[-1][0], trace[-1][1], edge_color=colors[f'{side}_outline'],
                                    fill_color=colors[f'{side}_fill'], info_text=f"d_{u_id}"))
        else:  # Plotting an active unit.
            objects = [Airplane(a.position.lat, a.position.lon, a.heading, edge_color=colors[f'{side}_outline'],
                                fill_color=colors[f'{side}_fill'], info_text=f"a_{a.id}")]
            if path:
                trace = [(pos.lat, pos.lon) for t, pos, h, s in self.sim.trace_record_units[a.id]]
                objects.append(PolyLine(trace, line_width=1, dash=(2, 2), edge_color=colors[f'{side}_outline']))
            if a.cannon_current_burst_secs > 0:  # Visualize cannon fire cone.
                d1 = geodetic_direct(a.position.lat, a.position.lon, sum_angles(a.heading, a.cannon_width_deg / 2.0),
                                     a.cannon_range_km * 1000)
                d2 = geodetic_direct(a.position.lat, a.position.lon, sum_angles(a.heading, -a.cannon_width_deg / 2.0),
                                     a.cannon_range_km * 1000)
                objects.append(PolyLine([(a.position.lat, a.position.lon), (d1[0], d1[1]), (d2[0], d2[1]),
                                         (a.position.lat, a.position.lon)], dash=(1, 1),
                                        edge_color=colors[f'{side}_outline']))
        return objects

    def plot(self, out_file: Path, paths=True):
        """Draws the current state of the entire scenario to a PNG file."""
        objects = [
            StatusMessage(self.sim.status_text or f"Step: {self.steps}"),
            TopLeftMessage(self.sim.utc_time.strftime("%H:%M:%S"))
        ]
        # Draw all agent and opponent aircraft (both alive and destroyed).
        for i in range(1, self.args.total_num + 1):
            col = 'blue' if i <= self.args.num_agents else 'red'
            if self.sim.unit_exists(i):
                objects.extend(self._plot_airplane(self.sim.get_unit(i), col, paths))
            elif i in self.sim.trace_record_units:  # Check if trace exists for destroyed unit
                objects.extend(self._plot_airplane(None, col, paths, True, i))

        # Draw all active missiles.
        for i in range(self.args.total_num + 1, self.sim._next_unit_id):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                col = "blue" if unit.source.id <= self.args.num_agents else "red"
                objects.append(
                    Missile(unit.position.lat, unit.position.lon, unit.heading, edge_color=colors[f'{col}_outline'],
                            fill_color=colors[f'{col}_fill'], info_text=f"m_{i}"))

        self.plotter.to_png(str(out_file), objects)