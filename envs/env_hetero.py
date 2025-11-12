# FILE: envs/env_hetero.py (Corrected with Dense Rewards)

import numpy as np
import gymnasium
from .env_base import HHMARLBaseEnv

# --- Constants ---
OBS_AC1, OBS_AC2 = 26, 24
OBS_ESC_AC1, OBS_ESC_AC2 = 30, 29


class LowLevelEnv(HHMARLBaseEnv):
    """Low-Level Environment with corrected reward shaping."""

    def __init__(self, env_config):
        self.args = env_config.get("args")
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"
        self.total_steps_elapsed = 0  # Global step counter for reward scheduling

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

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options={"mode": "LowLevel"})
        if self.args.level == 5 and self.args.agent_mode == "fight":
            k = self.np_random.integers(3, 6)
            self.opp_mode = "escape" if k == 5 else "fight"
        return obs, info

    def step(self, action):
        obs, rewards_dict, term_dict, trunc_dict, info = super().step(action)
        agg_reward = float(sum(rewards_dict.values())) if rewards_dict else 0.0
        terminated = term_dict.get("__all__", False)
        truncated = trunc_dict.get("__all__", False)
        # --- FIX: Bridge for Vectorized Envs ---
        # The per-agent rewards are passed via the info dict
        info["agent_rewards"] = rewards_dict
        return obs, agg_reward, terminated, truncated, info

    def state(self):
        state_dict = {}
        for ag_id in self.observation_space.keys():
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                unit, opps, friendlys = self.sim.get_unit(ag_id), self._nearby_object(ag_id), self._nearby_object(ag_id,
                                                                                                                  friendly=True)
                fri_id = friendlys[0][0] if friendlys else None
                if opps:
                    state = self.fight_state_values(ag_id, unit, opps[0],
                                                    fri_id) if self.agent_mode == "fight" else self.esc_state_values(
                        ag_id, unit, opps, fri_id)
                    self.opp_to_attack[ag_id] = opps[0][0]
                else:
                    state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            state_dict[ag_id] = state
        return state_dict

    def _take_action(self, action, info):  # info is passed by reference
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
                    # Fictitious self-play for opponents
                    pass  # Placeholder for policy loading
                else:
                    self._hardcoded_opp_logic(u, i)

        self.rewards = self._get_rewards(rewards, self.sim.do_tick(), opp_stats, info)

    def _get_rewards(self, rewards, events, opp_stats, info):
        # --- FIX: Full reward shaping implementation ---
        rews, destroyed_ids, _ = self._combat_rewards(events, opp_stats, mode="LowLevel")

        # Calculate the current scale for shaping rewards based on training progress
        shaping_scale = 1.0
        if self.args.use_reward_schedule:
            progress = min(1.0, self.total_steps_elapsed / self.args.shaping_decay_timesteps)
            shaping_scale = self.args.shaping_scale_initial * (1 - progress) + self.args.shaping_scale_final * progress

        if "reward_components" not in info: info["reward_components"] = {}

        for agent_id in self._agent_ids:
            # Initialize reward components for logging
            info["reward_components"][agent_id] = {
                "aim_reward": 0.0, "distance_reward": 0.0, "time_penalty": 0.0,
                "kill_reward": 0.0, "escape_reward": 0.0
            }

            # Dense rewards for 'fight' mode
            if self.agent_mode == "fight" and agent_id in opp_stats and self.sim.unit_exists(agent_id):
                focus_angle_norm, distance_norm = opp_stats[agent_id]
                # Reward for being close
                distance_reward = (1.0 - distance_norm) * 0.05 * shaping_scale
                # Reward for aiming at the opponent
                aim_reward = (1.0 - focus_angle_norm) * 0.05 * shaping_scale
                # Small penalty for every step to encourage efficiency
                time_penalty = -0.001

                rews.setdefault(agent_id, []).extend([distance_reward, aim_reward, time_penalty])
                info["reward_components"][agent_id].update({
                    "aim_reward": aim_reward, "distance_reward": distance_reward, "time_penalty": time_penalty
                })

            # Dense rewards for 'escape' mode
            if self.agent_mode == "escape" and self.args.esc_dist_rew and self.sim.unit_exists(agent_id):
                opps, escape_rew = self._nearby_object(agent_id), 0
                for j, o in enumerate(opps, start=1):
                    dist_unnormalized = o[2]
                    if dist_unnormalized < 0.06:
                        escape_rew -= (0.02 / j) * shaping_scale  # Penalize being too close
                    elif dist_unnormalized > 0.13:
                        escape_rew += (0.02 / j) * shaping_scale  # Reward being far away
                rews.setdefault(agent_id, []).append(escape_rew)
                info["reward_components"][agent_id]["escape_reward"] = escape_rew

        # Final reward aggregation
        for i in self._agent_ids:
            if self.sim.unit_exists(i) or i in destroyed_ids:
                total_reward = sum(rews.get(i, []))
                rewards[i] = total_reward
        return rewards

    def _hardcoded_opp_logic(self, unit, unit_id):
        # Dummy logic for non-learning opponents in early curriculum levels
        if self.args.level == 1:  # Static
            pass
        elif self.args.level == 2:  # Random
            unit.set_heading((unit.heading + self.np_random.uniform(-15, 15)) % 360)
        elif self.args.level == 3:  # Scripted
            pass  # Add scripted logic if needed