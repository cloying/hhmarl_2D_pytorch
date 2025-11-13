# FILE: envs/env_hetero.py (With Progress-Based Search Reward)

import numpy as np
import gymnasium
from .env_base import HHMARLBaseEnv

# --- Constants ---
OBS_AC1, OBS_AC2 = 26, 24
OBS_ESC_AC1, OBS_ESC_AC2 = 30, 29


class LowLevelEnv(HHMARLBaseEnv):
    def __init__(self, env_config):
        self.args = env_config.get("args")
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"
        self.episode_rewards = {i: 0.0 for i in range(1, self.args.num_agents + 1)}

        ### --- NEW: Attribute to store previous distance for progress reward --- ###
        self.previous_dist_to_center = {}

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

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options={"mode": "LowLevel"})
        self.episode_rewards = {i: 0.0 for i in self._agent_ids}

        ### --- NEW: Initialize previous distances at the start of an episode --- ###
        self.previous_dist_to_center = {}
        center_lat, center_lon = self.map_limits.absolute_position(0.5, 0.5)
        for agent_id in self._agent_ids:
            if self.sim.unit_exists(agent_id):
                unit = self.sim.get_unit(agent_id)
                self.previous_dist_to_center[agent_id] = np.hypot(unit.position.lat - center_lat,
                                                                  unit.position.lon - center_lon)

        return obs, info

    def step(self, action):
        self.steps += 1
        opp_stats = {}

        for i in self._agent_ids:
            if self.sim.unit_exists(i):
                if self.opp_to_attack.get(i) and self.sim.unit_exists(self.opp_to_attack[i]):
                    opp_stats[i] = [self._aspect_angle(self.opp_to_attack[i], i, True),
                                    self._distance(i, self.opp_to_attack[i], False)]
                self._take_base_action("LowLevel", self.sim.get_unit(i), i, self.opp_to_attack.get(i), action)

        for i in range(self.args.num_agents + 1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                self._hardcoded_opp_logic(self.sim.get_unit(i), i)

        events = self.sim.do_tick()
        rewards_dict = self._get_rewards(events, opp_stats)

        for agent_id, reward in rewards_dict.items():
            if agent_id in self.episode_rewards:
                self.episode_rewards[agent_id] += reward

        terminated = self.alive_agents <= 0 or self.alive_opps <= 0
        truncated = self.steps >= self.args.horizon
        done = terminated or truncated

        info = {"agent_rewards": rewards_dict}
        if done:
            info["episode"] = {"r": sum(self.episode_rewards.values()), "l": self.steps}

        agg_reward = float(sum(rewards_dict.values()))
        return self.state(), agg_reward, terminated, truncated, info

    def _get_shaping_rewards(self):
        shaping_rewards = {i: 0.0 for i in self._agent_ids}
        time_penalty = -0.0005

        for i in self._agent_ids:
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                shaping_rewards[i] += time_penalty

                ### --- MODIFICATION 1: Add a universal speed reward --- ###
                # This encourages the agent to always be active and explore.
                speed_reward = (unit.speed / unit.max_speed) * 0.001
                shaping_rewards[i] += speed_reward

                opps = self._nearby_object(i)

                ### --- MODIFICATION 2: Change search reward to be progress-based --- ###
                if not opps:
                    # No enemies nearby, reward for making progress towards the center.
                    center_lat, center_lon = self.map_limits.absolute_position(0.5, 0.5)
                    current_dist_to_center = np.hypot(unit.position.lat - center_lat, unit.position.lon - center_lon)

                    # Get the previous distance, defaulting to the current one if it's the first step.
                    prev_dist = self.previous_dist_to_center.get(i, current_dist_to_center)

                    # Reward is the change in distance (positive if distance decreased).
                    progress_reward = (prev_dist - current_dist_to_center) * 0.1  # Scaled to be a small nudge
                    shaping_rewards[i] += progress_reward

                    # Update the previous distance for the next step.
                    self.previous_dist_to_center[i] = current_dist_to_center
                    continue

                # --- Combat shaping rewards (only if enemies are present) ---
                closest_opp_id, closest_opp_dist_norm, _ = opps[0]
                if self.agent_mode == 'fight':
                    aspect_angle = self._aspect_angle(closest_opp_id, i, norm=True)
                    position_reward = (aspect_angle - 0.5) * 0.01
                    dist_reward = np.exp(-((closest_opp_dist_norm - 0.1) ** 2) / 0.05) * 0.01
                    shaping_rewards[i] += position_reward + dist_reward
                elif self.agent_mode == 'escape':
                    aspect_angle_of_opp = self._aspect_angle(i, closest_opp_id, norm=True)
                    threat_level = (1.0 - closest_opp_dist_norm) * aspect_angle_of_opp
                    escape_penalty = -threat_level * 0.02
                    shaping_rewards[i] += escape_penalty

        return shaping_rewards

    # --- All other methods (state, _get_rewards, _hardcoded_opp_logic, etc.) remain unchanged ---
    # (Code omitted for brevity, it's the same as the previous version)
    def state(self):
        state_dict = {}
        for ag_id in self.observation_space.keys():
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                unit = self.sim.get_unit(ag_id)
                opps = self._nearby_object(ag_id)
                friendlys = self._nearby_object(ag_id, friendly=True)
                fri_id = friendlys[0][0] if friendlys else None
                if opps:
                    if self.agent_mode == "fight":
                        state = self.fight_state_values(ag_id, unit, opps[0], fri_id)
                    else:
                        state = self.esc_state_values(ag_id, unit, opps, fri_id)
                    self.opp_to_attack[ag_id] = opps[0][0]
                else:
                    state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _get_rewards(self, events, opp_stats):
        combat_rews, _ = self._combat_rewards(events, opp_stats, mode="LowLevel")
        shaping_rews = self._get_shaping_rewards()
        final_rewards = {}
        for agent_id in self._agent_ids:
            total_reward = sum(combat_rews.get(agent_id, [])) + shaping_rews.get(agent_id, 0.0)
            if self.args.glob_frac > 0 and self.agent_mode == "fight":
                other_agent_id = 1 if agent_id == 2 else 2
                if other_agent_id in self._agent_ids:
                    total_reward += self.args.glob_frac * sum(combat_rews.get(other_agent_id, []))
            final_rewards[agent_id] = total_reward
        return final_rewards

    def _hardcoded_opp_logic(self, unit, unit_id):
        if self.args.level == 1:
            return
        elif self.args.level == 2:
            if self.steps % self.np_random.integers(35, 46) == 0:
                unit.set_heading(self.np_random.uniform(0, 360))
                unit.set_speed(self.np_random.uniform(100, unit.max_speed))
        elif self.args.level >= 3:
            d_agt = self._nearby_object(unit_id)
            if not d_agt or not self.sim.unit_exists(d_agt[0][0]): return
            target_agent = self.sim.get_unit(d_agt[0][0])
            bearing_to_agent = self._focus_angle(unit_id, target_agent.id, norm=False)
            sign = self._correct_angle_sign(unit, target_agent)
            turn = np.clip(bearing_to_agent * sign, -15, 15)
            unit.set_heading((unit.heading + turn) % 360)
            if d_agt[0][1] > 0.1:
                unit.set_speed(unit.max_speed * 0.8)
            else:
                unit.set_speed(unit.max_speed * 0.5)
            if d_agt[0][1] < 0.05 and bearing_to_agent < 10: unit.fire_cannon()