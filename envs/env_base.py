# FILE: envs/env_base.py (Cleaned, Final Version)

# --- Core Dependencies ---
import os
from math import sin, cos, acos, pi, hypot, radians, sqrt
from pathlib import Path

# --- Third-party Libraries ---
import numpy as np
import torch
import gymnasium

# --- Local Project Imports (Absolute from project root) ---
from warsim.scenplotter.scenario_plotter import (PlotConfig, ColorRGBA, StatusMessage,
                                                 TopLeftMessage, Airplane, PolyLine,
                                                 Waypoint, Missile, ScenarioPlotter)
from warsim.simulator.cmano_simulator import Position, CmanoSimulator, UnitDestroyedEvent
from warsim.simulator.ac1 import Rafale
from warsim.simulator.ac2 import RafaleLong
from utils.geodesics import geodetic_direct
from utils.map_limits import MapLimits
from utils.angles import sum_angles

# ------------------- Constants -------------------
colors = {
    'red_outline': ColorRGBA(0.8, 0.2, 0.2, 1), 'red_fill': ColorRGBA(0.8, 0.2, 0.2, 0.2),
    'blue_outline': ColorRGBA(0.3, 0.6, 0.9, 1), 'blue_fill': ColorRGBA(0.3, 0.6, 0.9, 0.2),
    'waypoint_outline': ColorRGBA(0.8, 0.8, 0.2, 1), 'waypoint_fill': ColorRGBA(0.8, 0.8, 0.2, 0.2)
}


# ------------------- Environment Class -------------------
class HHMARLBaseEnv(gymnasium.Env):
    """
    Base class for the HHMARL 2D environment, compatible with Gymnasium.
    """

    def __init__(self, map_size):
        super().__init__()
        self.steps = 0
        self.sim = None
        self.map_size = map_size
        self.map_limits = MapLimits(7.0, 5.0, 7.0 + map_size, 5.0 + map_size)
        self.alive_agents = 0
        self.alive_opps = 0
        self.rewards = {}
        self.opp_to_attack = {}
        self.missile_wait = {}
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.plt_cfg = PlotConfig()
        self.plt_cfg.units_scale = 20.0
        self.plotter = ScenarioPlotter(self.map_limits, dpi=200, config=self.plt_cfg)

    def reset(self, *, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)

        self.steps = 0
        self.alive_agents = 0
        self.alive_opps = 0
        self.hardcoded_opps_escaping = False
        self.opps_escaping_time = 0
        self.missile_wait = {i: 0 for i in range(1, self.args.total_num + 1)}
        self.opp_to_attack = {i: None for i in range(1, self.args.total_num + 1)}

        self.sim = CmanoSimulator(
            random_generator=self.np_random,
            num_units=self.args.num_agents,
            num_opp_units=self.args.num_opps
        )

        self._reset_scenario(options.get("mode", None) if options else None)
        return self.state(), {}

    def step(self, action):
        """Executes one time step in the environment."""
        self.rewards = {}
        if action:
            self._take_action(action)

        done = (self.alive_agents <= 0 or self.alive_opps <= 0 or self.steps >= self.args.horizon)
        terminateds = {"__all__": done}
        truncateds = {"__all__": done}
        info = {}
        return self.state(), self.rewards, terminateds, truncateds, info

    def fight_state_values(self, agent_id, unit, opp, fri_id=None):
        state = []
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.extend([x, y,
                      np.clip(unit.speed / unit.max_speed, 0, 1),
                      np.clip((unit.heading % 359) / 359, 0, 1),
                      self._focus_angle(agent_id, opp[0], True),
                      self._aspect_angle(opp[0], agent_id),
                      self._heading_diff(agent_id, opp[0]),
                      opp[1],
                      np.clip(unit.cannon_remain_secs / unit.cannon_max, 0, 1)])
        if unit.ac_type == 1:
            state.extend([np.clip(unit.missile_remain / unit.rocket_max, 0, 1),
                          int(self.missile_wait[agent_id] == 0),
                          int(bool(unit.actual_missile) or unit.cannon_current_burst_secs > 0)])
        else:
            state.append(int(unit.cannon_current_burst_secs > 0))
        state.extend(self.opp_ac_values("fight", opp[0], agent_id, opp[1]))
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def esc_state_values(self, agent_id, unit, opps, fri_id=None):
        state = []
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state.extend([x, y,
                      np.clip(unit.speed / unit.max_speed, 0, 1),
                      np.clip((unit.heading % 359) / 359, 0, 1),
                      np.clip(unit.cannon_remain_secs / unit.cannon_max, 0, 1)])
        if unit.ac_type == 1:
            state.append(np.clip(unit.missile_remain / unit.rocket_max, 0, 1))
        shot = unit.cannon_current_burst_secs > 0 or (unit.ac_type == 1 and bool(unit.actual_missile))
        state.append(int(shot))
        opp_state = []
        for opp in opps:
            opp_state.extend(self.opp_ac_values("esc", opp[0], agent_id, opp[1]))
            if len(opp_state) >= 18: break
        if len(opp_state) < 18:
            opp_state.extend(np.zeros(18 - len(opp_state), dtype=np.float32).tolist())
        state.extend(opp_state)
        state.extend(self.friendly_ac_values(agent_id, fri_id))
        return state

    def friendly_ac_values(self, agent_id, fri_id=None):
        if not fri_id or not self.sim.unit_exists(fri_id):
            return np.zeros(5, dtype=np.float32).tolist()
        else:
            unit = self.sim.get_unit(fri_id)
            x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
            return [x, y,
                    self._focus_angle(agent_id, fri_id, True),
                    self._focus_angle(fri_id, agent_id, True),
                    self._distance(agent_id, fri_id, True)]

    def opp_ac_values(self, mode, opp_id, agent_id, dist):
        unit = self.sim.get_unit(opp_id)
        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        state = [x, y,
                 np.clip(unit.speed / unit.max_speed, 0, 1),
                 np.clip((unit.heading % 359) / 359, 0, 1),
                 self._heading_diff(opp_id, agent_id)]
        if mode == "fight":
            state.extend([self._focus_angle(opp_id, agent_id, True), self._aspect_angle(agent_id, opp_id)])
        else:
            state.extend([self._focus_angle(agent_id, opp_id, True), self._focus_angle(opp_id, agent_id, True)])
        if mode == "HighLevel":
            state.extend([self._aspect_angle(agent_id, opp_id), self._aspect_angle(opp_id, agent_id)])
        state.append(dist)
        if mode != "HighLevel":
            shot = unit.cannon_current_burst_secs > 0 or (unit.ac_type == 1 and bool(unit.actual_missile))
            state.append(int(shot))
        return state

    def _take_base_action(self, mode, unit, unit_id, opp_id, actions, rewards=None):
        # The actions dictionary is now expected to be keyed by agent ID
        if unit_id not in actions:
            return rewards

        act = actions[unit_id]
        unit.set_heading((unit.heading + (act[0] - 6) * 15) % 360)
        unit.set_speed(100 + ((unit.max_speed - 100) / 8) * act[1])
        if bool(act[2]) and unit.cannon_remain_secs > 0:
            unit.fire_cannon()
        if unit.ac_type == 1 and bool(act[3]):
            if opp_id and unit.missile_remain > 0 and not unit.actual_missile and self.missile_wait[unit_id] == 0:
                if self.sim.unit_exists(opp_id):
                    unit.fire_missile(unit, self.sim.get_unit(opp_id), self.sim)
                    if mode == "LowLevel":
                        self.missile_wait[unit_id] = self.np_random.integers(7, 18)
                    else:
                        self.missile_wait[unit_id] = self.np_random.integers(8, 13)
        if self.missile_wait[unit_id] > 0 and not bool(unit.actual_missile):
            self.missile_wait[unit_id] -= 1
        return rewards

    def _combat_rewards(self, events, opp_stats=None, mode="LowLevel"):
        rews = {a: [] for a in range(1, self.args.num_agents + 1)}
        destroyed_ids = []
        s = self.args.rew_scale
        kill_event = False
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
        for ev in events:
            if isinstance(ev, UnitDestroyedEvent):
                if ev.unit_killer.id <= self.args.num_agents and ev.unit_killer.id in rews:
                    if ev.unit_destroyed.id > self.args.num_agents:
                        rews[ev.unit_killer.id].append(1 * s)
                        self.alive_opps -= 1
                    elif ev.unit_destroyed.id <= self.args.num_agents:
                        rews[ev.unit_killer.id].append(-2 * s)
                        if self.args.friendly_punish and ev.unit_destroyed.id in rews:
                            rews[ev.unit_destroyed.id].append(-2 * s)
                            destroyed_ids.append(ev.unit_destroyed.id)
                        self.alive_agents -= 1
                elif ev.unit_killer.id > self.args.num_agents:
                    if ev.unit_destroyed.id <= self.args.num_agents and ev.unit_destroyed.id in rews:
                        p = -2 if mode == "LowLevel" else -1
                        rews[ev.unit_destroyed.id].append(p * s)
                        destroyed_ids.append(ev.unit_destroyed.id)
                        self.alive_agents -= 1
                    elif ev.unit_destroyed.id > self.args.num_agents:
                        self.alive_opps -= 1
                kill_event = True
        return rews, destroyed_ids, kill_event

    def _get_policies(self, mode):
        pass

    def _policy_actions(self, policy_type, agent_id, unit):
        if unit.ac_type == 1:
            return {agent_id: np.array([6, 4, 0, 0])}
        else:
            return {agent_id: np.array([6, 4, 0])}

    def _nearby_object(self, agent_id, friendly=False):
        order = []
        if friendly:
            id_pool = list(range(1, self.args.num_agents + 1)) if agent_id <= self.args.num_agents else list(
                range(self.args.num_agents + 1, self.args.total_num + 1))
        else:
            id_pool = list(
                range(self.args.num_agents + 1, self.args.total_num + 1)) if agent_id <= self.args.num_agents else list(
                range(1, self.args.num_agents + 1))
        for i in id_pool:
            if i != agent_id and self.sim.unit_exists(i):
                dist_norm = self._distance(agent_id, i, True)
                dist_raw = self._distance(agent_id, i, False) if not friendly else 0
                order.append([i, dist_norm, dist_raw])
        order.sort(key=lambda x: x[1])
        return order

    def _focus_angle(self, agent_id, opp_id, norm=False):
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
        ag_unit, opp_unit = self.sim.get_unit(agent_id), self.sim.get_unit(opp_id)
        dist = hypot(opp_unit.position.lon - ag_unit.position.lon, opp_unit.position.lat - ag_unit.position.lat)
        return self._shifted_range(dist, 0, sqrt(2 * self.map_size ** 2), 0, 1) if norm else dist

    def _aspect_angle(self, agent_id, opp_id, norm=True):
        focus = self._focus_angle(agent_id, opp_id)
        aspect = 180 - focus
        return np.clip(aspect / 180, 0, 1) if norm else np.clip(aspect, 0, 180)

    def _heading_diff(self, agent_id, opp_id, norm=True):
        ag_unit, opp_unit = self.sim.get_unit(agent_id), self.sim.get_unit(opp_id)
        ag_head_rad = ((90 - ag_unit.heading) % 360) * (pi / 180)
        opp_head_rad = ((90 - opp_unit.heading) % 360) * (pi / 180)
        ag_vec = np.array([cos(ag_head_rad), sin(ag_head_rad)])
        opp_vec = np.array([cos(opp_head_rad), sin(opp_head_rad)])
        angle = acos(np.clip(np.dot(ag_vec, opp_vec), -1, 1)) * (180 / pi)
        return np.clip(angle / 180, 0, 1) if norm else angle

    def _shifted_range(self, x, a, b, c, d):
        return c + ((d - c) / (b - a)) * (x - a)

    def _correct_angle_sign(self, opp_unit, ag_unit):
        line = lambda x0, y0, x1, y1, x2, y2: (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        x, y, a = opp_unit.position.lon, opp_unit.position.lat, opp_unit.heading
        x1 = x + sin(radians(a % 360))
        y1 = y + cos(radians(a % 360))
        val = line(x, y, x1, y1, ag_unit.position.lon, ag_unit.position.lat)
        return 1 if val < 0 else -1

    def _sample_state(self, agent_type, agent_index, side):
        x, y, a = 0, 0, 0
        if agent_type == "agent":
            if self.args.level >= 2:
                x = self.np_random.uniform(7.07, 7.12) if side == 1 else self.np_random.uniform(7.18, 7.23)
                y = self.np_random.uniform(5.09 + agent_index * 0.1, 5.12 + agent_index * 0.1)
                a = self.np_random.integers(0, 360)
            else:
                x = self.np_random.uniform(7.12, 7.14) if side == 1 else self.np_random.uniform(7.16, 7.17)
                y = self.np_random.uniform(5.1 + agent_index * 0.1, 5.11 + agent_index * 0.1)
                a = self.np_random.integers(30, 151) if side == 1 else self.np_random.integers(200, 331)

        elif agent_type == "opp":
            if self.args.level >= 2:
                x = self.np_random.uniform(7.18, 7.23) if side == 1 else self.np_random.uniform(7.07, 7.12)
                y = self.np_random.uniform(5.09 + agent_index * 0.1, 5.12 + agent_index * 0.1)
                a = self.np_random.integers(0, 360)
            else:
                x = self.np_random.uniform(7.16, 7.17) if side == 1 else self.np_random.uniform(7.12, 7.14)
                y = self.np_random.uniform(5.1 + agent_index * 0.1, 5.11 + agent_index * 0.1)
        return x, y, a

    def _reset_scenario(self, mode):
        side = self.np_random.integers(1, 3)
        for group, count in [("agent", self.args.num_agents), ("opp", self.args.num_opps)]:
            for i in range(count):
                x, y, a = self._sample_state(group, i, side)
                ac_type = (i % 2) + 1
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
        objects = []
        if use_backup:
            trace = [(pos.lat, pos.lon) for t, pos, h, s in self.sim.trace_record_units[u_id]]
            objects.append(PolyLine(trace, line_width=1, dash=(2, 2), edge_color=colors[f'{side}_outline']))
            objects.append(Waypoint(trace[-1][0], trace[-1][1], edge_color=colors[f'{side}_outline'],
                                    fill_color=colors[f'{side}_fill'], info_text=f"d_{u_id}"))
        else:
            objects = [Airplane(a.position.lat, a.position.lon, a.heading, edge_color=colors[f'{side}_outline'],
                                fill_color=colors[f'{side}_fill'], info_text=f"a_{a.id}")]
            if path:
                trace = [(pos.lat, pos.lon) for t, pos, h, s in self.sim.trace_record_units[a.id]]
                objects.append(PolyLine(trace, line_width=1, dash=(2, 2), edge_color=colors[f'{side}_outline']))
            if a.cannon_current_burst_secs > 0:
                d1 = geodetic_direct(a.position.lat, a.position.lon, sum_angles(a.heading, a.cannon_width_deg / 2.0),
                                     a.cannon_range_km * 1000)
                d2 = geodetic_direct(a.position.lat, a.position.lon, sum_angles(a.heading, -a.cannon_width_deg / 2.0),
                                     a.cannon_range_km * 1000)
                objects.append(PolyLine([(a.position.lat, a.position.lon), (d1[0], d1[1]), (d2[0], d2[1]),
                                         (a.position.lat, a.position.lon)], dash=(1, 1),
                                        edge_color=colors[f'{side}_outline']))
        return objects

    def plot(self, out_file: Path, paths=True):
        objects = [
            StatusMessage(self.sim.status_text or f"Step: {self.steps}"),
            TopLeftMessage(self.sim.utc_time.strftime("%H:%M:%S"))
        ]
        for i in range(1, self.args.total_num + 1):
            col = 'blue' if i <= self.args.num_agents else 'red'
            if self.sim.unit_exists(i):
                objects.extend(self._plot_airplane(self.sim.get_unit(i), col, paths))
            elif i in self.sim.trace_record_units:
                objects.extend(self._plot_airplane(None, col, paths, True, i))
        for i in range(self.args.total_num + 1, self.sim._next_unit_id):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                col = "blue" if unit.source.id <= self.args.num_agents else "red"
                objects.append(
                    Missile(unit.position.lat, unit.position.lon, unit.heading, edge_color=colors[f'{col}_outline'],
                            fill_color=colors[f'{col}_fill'], info_text=f"m_{i}"))

        self.plotter.to_png(str(out_file), objects)