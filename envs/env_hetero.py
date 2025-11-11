# FILE: envs/env_hetero.py (Corrected with API Bridge)

"""
Low-Level Environment for HHMARL 2D Aircombat.
This environment is used for training the basic "fight" and "escape" policies
through a curriculum of increasing difficulty.
"""
import numpy as np
import gymnasium
import os

# --- Local Project Imports (This MUST be relative) ---
from .env_base import HHMARLBaseEnv

# --- Constants ---
OBS_AC1 = 26;
OBS_AC2 = 24;
OBS_ESC_AC1 = 30;
OBS_ESC_AC2 = 29


class LowLevelEnv(HHMARLBaseEnv):
    """Low-Level Environment for Aircombat Maneuvering."""

    def __init__(self, env_config):
        self.args = env_config.get("args", None)
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"
        self.total_steps_elapsed = 0

        obs_spaces, act_spaces, self.obs_dim_map = {}, {}, {}
        for i in range(1, self.args.num_agents + 1):
            is_ac1_type = (i - 1) % 2 == 0
            obs_dim = (OBS_AC1 if is_ac1_type else OBS_AC2) if self.agent_mode == "fight" else (
                OBS_ESC_AC1 if is_ac1_type else OBS_ESC_AC2)
            self.obs_dim_map[i] = obs_dim
            obs_spaces[i] = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            act_spaces[i] = gymnasium.spaces.MultiDiscrete([13, 9, 2, 2] if is_ac1_type else [13, 9, 2])

        self.observation_space = gymnasium.spaces.Dict(obs_spaces)
        self.action_space = gymnasium.spaces.Dict(act_spaces)
        self._agent_ids = set(range(1, self.args.num_agents + 1))
        super().__init__(self.args.map_size)

        if self.args.level >= 4: self._get_policies("LowLevel")
        if self.args.level == 5 and self.agent_mode == "escape": self.opp_mode = "fight"

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options={"mode": "LowLevel"})
        if self.args.level == 5 and self.args.agent_mode == "fight":
            k = self.np_random.integers(3, 6)
            self.opp_mode = "escape" if k == 5 else "fight"
        return obs, info

    def step(self, action):
        """
        --- FIX: API Bridge ---
        This method now returns a single aggregated reward to satisfy the Gymnasium
        vector wrappers, while placing the essential per-agent reward dictionary
        into the info dictionary for the MARL training loop.
        """
        obs, rewards_dict, terminateds_dict, truncateds_dict, info = super().step(action)

        # Aggregate reward for the wrapper
        aggregated_reward = float(sum(rewards_dict.values())) if rewards_dict else 0.0

        # Extract global done flags
        terminated = terminateds_dict.get("__all__", False)
        truncated = truncateds_dict.get("__all__", False)

        # Preserve the per-agent rewards in the info dict for the training loop
        info["agent_rewards"] = rewards_dict

        return obs, aggregated_reward, terminated, truncated, info

    def state(self):
        return self.lowlevel_state(self.agent_mode)

    def lowlevel_state(self, mode):
        state_dict = {}
        for ag_id in self.observation_space.keys():
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                unit, opps, friendlys = self.sim.get_unit(ag_id), self._nearby_object(ag_id), self._nearby_object(ag_id,
                                                                                                                  friendly=True)
                fri_id = friendlys[0][0] if friendlys else None
                if opps:
                    state = self.fight_state_values(ag_id, unit, opps[0],
                                                    fri_id) if mode == "fight" else self.esc_state_values(ag_id, unit,
                                                                                                          opps, fri_id)
                    self.opp_to_attack[ag_id] = opps[0][0]
                else:
                    state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            state_dict[ag_id] = state
        return state_dict

    def _take_action(self, action, info):
        self.steps += 1
        self.total_steps_elapsed += getattr(self.args, 'num_workers', 1)
        rewards = {i: 0 for i in self._agent_ids}
        opp_stats = {}

        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if i <= self.args.num_agents:
                    if self.opp_to_attack.get(i) and self.sim.unit_exists(self.opp_to_attack[i]):
                        opp_stats[i] = [self._focus_angle(i, self.opp_to_attack[i], True),
                                        self._distance(i, self.opp_to_attack[i], True)]
                    self._take_base_action("LowLevel", u, i, self.opp_to_attack.get(i), action, rewards)
                elif self.args.level >= 4:
                    opp_actions = self._policy_actions(policy_type=self.opp_mode, agent_id=i, unit=u)
                    self._take_base_action("LowLevel", u, i, self.opp_to_attack.get(i), opp_actions, rewards)
                else:
                    self._hardcoded_opp_logic(u, i)

        self.rewards = self._get_rewards(rewards, self.sim.do_tick(), opp_stats, info)

    def _hardcoded_opp_logic(self, unit, unit_id):
        if self.args.level == 3: self._hardcoded_opp_level3(unit, unit_id)

    def _get_rewards(self, rewards, events, opp_stats, info):
        rews, destroyed_ids, _ = self._combat_rewards(events, opp_stats, mode="LowLevel")

        shaping_reward_scale = 1.0
        if self.args.use_reward_schedule:
            progress = min(1.0, self.total_steps_elapsed / self.args.shaping_decay_timesteps)
            shaping_reward_scale = self.args.shaping_scale_initial * (
                        1 - progress) + self.args.shaping_scale_final * progress

        if "reward_components" not in info: info["reward_components"] = {}

        for agent_id in self._agent_ids:
            info["reward_components"][agent_id] = {"aim_reward": 0.0, "distance_reward": 0.0, "time_penalty": 0.0,
                                                   "kill_reward": 0.0, "escape_reward": 0.0}
            if self.agent_mode == "fight" and agent_id in opp_stats and self.sim.unit_exists(agent_id):
                focus_angle_norm, distance_norm = opp_stats[agent_id]
                distance_reward, aim_reward, time_penalty = (1.0 - distance_norm) * 0.05 * shaping_reward_scale, (
                            1.0 - focus_angle_norm) * 0.05 * shaping_reward_scale, -0.001
                rews.setdefault(agent_id, []).append(distance_reward + aim_reward + time_penalty)
                info["reward_components"][agent_id].update(
                    {"aim_reward": aim_reward, "distance_reward": distance_reward, "time_penalty": time_penalty})
            if self.agent_mode == "escape" and self.args.esc_dist_rew and self.sim.unit_exists(agent_id):
                opps, escape_rew = self._nearby_object(agent_id), 0
                for j, o in enumerate(opps, start=1):
                    if o[2] < 0.06:
                        escape_rew -= (0.02 / j) * shaping_reward_scale
                    elif o[2] > 0.13:
                        escape_rew += (0.02 / j) * shaping_reward_scale
                rews.setdefault(agent_id, []).append(escape_rew)
                info["reward_components"][agent_id]["escape_reward"] = escape_rew

        for i in self._agent_ids:
            if self.sim.unit_exists(i) or i in destroyed_ids:
                total_reward = sum(rews.get(i, []))
                if self.args.glob_frac > 0 and self.agent_mode == "fight":
                    for other_id in self._agent_ids:
                        if i != other_id: total_reward += self.args.glob_frac * sum(rews.get(other_id, []))
                rewards[i] = total_reward
        return rewards

    # (Other hardcoded opponent methods remain the same)
    def _hardcoded_opp_level3(self, unit, unit_id):
        if self.steps % 60 == 0 and not self.hardcoded_opps_escaping:
            self.hardcoded_opps_escaping = bool(self.np_random.integers(0, 2))
            if self.hardcoded_opps_escaping: self.opps_escaping_time = int(self.np_random.uniform(20, 30))
        if self.hardcoded_opps_escaping:
            _, heading, speed, fire, _, _ = self._escaping_opp(unit)
            self.opps_escaping_time -= 1
            if self.opps_escaping_time <= 0: self.hardcoded_opps_escaping = False
        else:
            opp, heading, speed, fire, fire_missile, _ = self._hardcoded_fight_opp(unit, unit_id)
            if fire_missile and opp and not unit.actual_missile and self.missile_wait[
                unit_id] == 0 and unit.ac_type == 1:
                unit.fire_missile(unit, self.sim.get_unit(opp), self.sim)
                self.missile_wait[unit_id] = 10
        unit.set_heading(heading);
        unit.set_speed(speed)
        if fire: unit.fire_cannon()

    def _escaping_opp(self, unit):
        y, x = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        heading = int(
            self.np_random.uniform(30, 60) if x < 0.5 else self.np_random.uniform(300, 330)) if y < 0.5 else int(
            self.np_random.uniform(120, 150) if x < 0.5 else self.np_random.uniform(210, 240))
        speed = int(self.np_random.uniform(300, 600))
        return None, heading, speed, bool(self.np_random.integers(0, 2)), False, 0

    def _hardcoded_fight_opp(self, opp_unit, opp_id):
        d_agt = self._nearby_object(opp_id)
        heading, fire, fire_missile, head_rel = opp_unit.heading, False, False, 0
        speed = int(self.np_random.uniform(100, 400))
        if d_agt:
            target_id, dist_norm = d_agt[0][0], d_agt[0][1]
            sign = self._correct_angle_sign(opp_unit, self.sim.get_unit(target_id))
            focus = self._focus_angle(opp_id, target_id)
            if dist_norm > 0.008 and focus > 4: heading = (heading + self.np_random.uniform(0.7,
                                                                                            1.3) * sign * focus) % 360
            if dist_norm > 0.05: speed = int(
                self.np_random.uniform(500, 800) if focus < 30 else self.np_random.uniform(100, 500))
            fire, fire_missile = (dist_norm < 0.03 and focus < 10), (dist_norm < 0.09 and focus < 5)
        speed = np.clip(speed, 0, opp_unit.max_speed)
        return d_agt[0][0] if d_agt else None, heading, speed, fire, fire_missile, head_rel