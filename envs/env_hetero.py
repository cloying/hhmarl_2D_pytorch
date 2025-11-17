# FILE: envs/env_hetero.py (Updated with Tunable Firing Reward)

import numpy as np
import gymnasium
import torch
from .env_base import HHMARLBaseEnv

# Constants
OBS_AC1, OBS_AC2 = 26, 24
OBS_ESC_AC1, OBS_ESC_AC2 = 30, 29
NODE_FEATURE_DIM = 10


class LowLevelEnv(HHMARLBaseEnv):
    def __init__(self, env_config):
        self.args = env_config.get("args")
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"
        self.episode_rewards = {i: 0.0 for i in range(1, self.args.num_agents + 1)}
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
        self.previous_dist_to_center = {}
        center_lat, center_lon = self.map_limits.absolute_position(0.5, 0.5)
        for agent_id in self._agent_ids:
            if self.sim.unit_exists(agent_id):
                unit = self.sim.get_unit(agent_id)
                self.previous_dist_to_center[agent_id] = np.hypot(unit.position.lat - center_lat,
                                                                  unit.position.lon - center_lon)
        info["graph_data"] = self._get_graph_state()
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
        combat_rewards, shaping_rewards = self._get_rewards(events, opp_stats, action)

        for agent_id in self._agent_ids:
            total_reward = combat_rewards.get(agent_id, 0.0) + shaping_rewards.get(agent_id, 0.0)
            if agent_id in self.episode_rewards:
                self.episode_rewards[agent_id] += total_reward

        terminated = self.alive_agents <= 0 or self.alive_opps <= 0
        truncated = self.steps >= self.args.horizon
        done = terminated or truncated

        info = {
            "agent_rewards": combat_rewards,
            "shaping_rewards": shaping_rewards,
            "graph_data": self._get_graph_state()
        }
        if done:
            info["episode"] = {"r": sum(self.episode_rewards.values()), "l": self.steps}

        agg_reward = float(sum(combat_rewards.values()) + sum(shaping_rewards.values()))
        return self.state(), agg_reward, terminated, truncated, info

    def _get_graph_state(self):
        active_units = []
        node_features = []
        all_unit_ids = list(range(1, self.args.total_num + 1))
        for unit_id in all_unit_ids:
            if self.sim.unit_exists(unit_id):
                active_units.append(unit_id)
                unit = self.sim.get_unit(unit_id)
                x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                team_id = [1.0, 0.0] if unit.group == "agent" else [0.0, 1.0]
                missile_ammo = 0.0;
                has_missile = 0.0
                if unit.ac_type == 1:
                    missile_ammo = np.clip(unit.missile_remain / unit.rocket_max, 0, 1) if unit.rocket_max > 0 else 0.0
                    has_missile = 1.0
                features = [x, y,
                            np.clip(unit.speed / unit.max_speed, 0, 1),
                            np.clip(unit.heading / 360.0, 0, 1),
                            np.clip(unit.cannon_remain_secs / unit.cannon_max, 0, 1),
                            missile_ammo,
                            1.0 if unit.cannon_current_burst_secs > 0 else 0.0,
                            has_missile] + team_id
                node_features.append(features)
        if not node_features:
            return {
                "x": torch.empty((0, NODE_FEATURE_DIM), dtype=torch.float32),
                "edge_index": torch.empty((2, 0), dtype=torch.long)
            }
        num_nodes = len(active_units)
        edge_index = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j: edge_index.append([i, j])
        return {
            "x": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        }

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

    def _get_rewards(self, events, opp_stats, actions):
        combat_rews_dict, _ = self._combat_rewards(events, opp_stats, mode="LowLevel",
                                                   kill_reward_bonus=self.args.kill_reward_bonus)
        shaping_rews = self._get_shaping_rewards(actions)
        final_combat_rewards = {}
        for agent_id in self._agent_ids:
            sparse_reward = sum(combat_rews_dict.get(agent_id, []))
            if self.args.glob_frac > 0 and self.agent_mode == "fight":
                other_agent_id = 1 if agent_id == 2 else 2
                if other_agent_id in self._agent_ids:
                    sparse_reward += self.args.glob_frac * sum(combat_rews_dict.get(other_agent_id, []))
            final_combat_rewards[agent_id] = sparse_reward
        return final_combat_rewards, shaping_rews

    def _get_shaping_rewards(self, actions):
        shaping_rewards = {i: 0.0 for i in self._agent_ids}
        time_penalty = -0.001  # A small penalty for every step to encourage efficiency

        for i in self._agent_ids:
            if self.sim.unit_exists(i):
                shaping_rewards[i] += time_penalty

                opps = self._nearby_object(i)
                if not opps:
                    continue

                closest_opp_id, closest_opp_dist_norm, _ = opps[0]

                if self.agent_mode == 'fight':
                    # REWARD 1: Simple reward for getting closer
                    # (You would need to store previous distance to calculate this)
                    # For now, let's use a simpler proxy:
                    progress_reward = (1.0 - closest_opp_dist_norm) * 0.005
                    shaping_rewards[i] += progress_reward

                    # REWARD 2: Reward for firing under good conditions
                    agent_action = actions.get(i)
                    if agent_action is not None and (
                            agent_action[2] == 1 or (len(agent_action) > 3 and agent_action[3] == 1)):
                        focus_angle_norm = self._focus_angle(i, closest_opp_id, norm=True)
                        # Condition: Pointing at enemy and reasonably close
                        if focus_angle_norm < 0.1 and closest_opp_dist_norm < 0.3:
                            shaping_rewards[i] += self.args.firing_reward
                        else:
                            shaping_rewards[i] += self.args.ammo_penalty

        return shaping_rewards

    def _hardcoded_opp_logic(self, unit, unit_id):
        if self.args.level == 1:
            return  # Static opponent

        elif self.args.level == 2:
            # Fire cannon periodically to apply pressure
            if self.steps % 10 == 0:
                unit.fire_cannon()

            # Make a sharp, random turn at intervals
            if self.steps % self.np_random.integers(35, 46) == 0:
                direction = 1 if self.np_random.random() < 0.5 else -1
                turn_angle = 90
                unit.set_heading((unit.heading + direction * turn_angle) % 360)
                unit.set_speed(self.np_random.uniform(100, unit.max_speed * 0.8))
            return

        elif self.args.level >= 3:
            # Your existing logic for Level 3+
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