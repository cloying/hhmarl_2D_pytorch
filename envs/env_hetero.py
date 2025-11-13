# FILE: envs/env_hetero.py (RADICALLY SIMPLIFIED FOR DEBUGGING)

import numpy as np
import gymnasium
import json
import os
from .env_base import HHMARLBaseEnv

# --- Constants ---
OBS_AC1, OBS_AC2 = 26, 24
OBS_ESC_AC1, OBS_ESC_AC2 = 30, 29


class LowLevelEnv(HHMARLBaseEnv):
    def __init__(self, env_config):
        self.args = env_config.get("args")
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"

        # --- Setup for debug logging ---
        self.run_dir = self.args.env_config.get("run_dir", ".")
        self.is_debug_worker = False

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
        return obs, info

    def step(self, action):
        obs, rewards_dict, term_dict, trunc_dict, info = super().step(action)
        agg_reward = float(sum(rewards_dict.values())) if rewards_dict else 0.0
        terminated = term_dict.get("__all__", False)
        truncated = trunc_dict.get("__all__", False)
        info["agent_rewards"] = rewards_dict
        return obs, agg_reward, terminated, truncated, info

    def state(self):
        # This method is correct and unchanged
        state_dict = {}
        for ag_id in self.observation_space.keys():
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                unit, opps, friendlys = self.sim.get_unit(ag_id), self._nearby_object(ag_id), self._nearby_object(ag_id,
                                                                                                                  friendly=True)
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

    def _take_action(self, action, info):
        # This method is correct and unchanged
        self.steps += 1
        rewards = {i: 0 for i in self._agent_ids}
        opp_stats_for_kill_reward = {}  # Unused in this version, but required by the base class
        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if i <= self.args.num_agents:
                    self._take_base_action("LowLevel", u, i, self.opp_to_attack.get(i), action, rewards)
                else:
                    self._hardcoded_opp_logic(u, i)
        # The key is that we pass the empty `rewards` dict into `_get_rewards`
        self.rewards = self._get_rewards(rewards, self.sim.do_tick(), opp_stats_for_kill_reward, info)

    ### --- THE SCORCHED EARTH REWARD FUNCTION --- ###
    def _get_rewards(self, rewards, events, opp_stats, info):
        """
        This function is radically simplified for debugging.
        It IGNORES all game events and gives a constant negative reward to
        any living agent on every single step.
        """
        # Get sparse event rewards just to check for destroyed agents
        _, destroyed_ids, _ = self._combat_rewards(events, opp_stats, mode="LowLevel")

        # For every agent that is supposed to be learning...
        for agent_id in self._agent_ids:
            # If the agent is alive...
            if self.sim.unit_exists(agent_id):
                # Give it an undeniable, non-zero reward.
                rewards[agent_id] = -0.01
            # If the agent was just destroyed, it might get a large negative reward from _combat_rewards
            elif agent_id in destroyed_ids:
                # In this case, we just use the reward from the event.
                # For this simple test, we can even override it to be sure.
                rewards[agent_id] = -1.0
            else:
                # If the agent is dead, its reward is zero.
                rewards[agent_id] = 0.0

        return rewards

    def _hardcoded_opp_logic(self, unit, unit_id):
        if self.args.level == 1:
            pass  # Opponent is static
        else:
            unit.set_heading((unit.heading + self.np_random.uniform(-15, 15)) % 360)