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
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29


class LowLevelEnv(HHMARLBaseEnv):
    """
    Low-Level Environment for Aircombat Maneuvering.
    """

    def __init__(self, env_config):
        """
        Initializes the low-level environment.
        """
        self.args = env_config.get("args", None)
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"

        obs_spaces = {}
        act_spaces = {}
        self.obs_dim_map = {}

        self.action_space = gymnasium.spaces.Dict(act_spaces)
        self._agent_ids = set(range(1, self.args.num_agents + 1))  # This line should already be here, just confirm it.
        super().__init__(self.args.map_size)

        for i in range(1, self.args.num_agents + 1):
            is_ac1_type = (i - 1) % 2 == 0
            if self.agent_mode == "fight":
                obs_dim = OBS_AC1 if is_ac1_type else OBS_AC2
            else:
                obs_dim = OBS_ESC_AC1 if is_ac1_type else OBS_ESC_AC2
            self.obs_dim_map[i] = obs_dim
            obs_spaces[i] = gymnasium.spaces.Box(
                low=np.zeros(obs_dim), high=np.ones(obs_dim), dtype=np.float32
            )
            if is_ac1_type:
                act_spaces[i] = gymnasium.spaces.MultiDiscrete([13, 9, 2, 2])
            else:
                act_spaces[i] = gymnasium.spaces.MultiDiscrete([13, 9, 2])

        self.observation_space = gymnasium.spaces.Dict(obs_spaces)
        self.action_space = gymnasium.spaces.Dict(act_spaces)
        self._agent_ids = set(range(1, self.args.num_agents + 1))
        super().__init__(self.args.map_size)
        self.total_steps_elapsed = 0

        if self.args.level >= 4:
            self._get_policies("LowLevel")
        if self.args.level == 5 and self.agent_mode == "escape":
            self.opp_mode = "fight"

    def reset(self, *, seed=None, options=None):
        """Resets the environment and handles curriculum logic for level 5."""
        obs, info = super().reset(seed=seed, options={"mode": "LowLevel"})
        if self.args.level == 5 and self.args.agent_mode == "fight":
            k = self.np_random.integers(3, 6)
            self.policy = self.policies[k]
            self.opp_mode = "escape" if k == 5 else "fight"
        return obs, info

    def step(self, action):
        """
        Wraps the base environment's step function. It adapts the multi-agent
        dictionary outputs (e.g., terminateds_dict) into the single boolean format
        expected by standard Gymnasium wrappers like RecordEpisodeStatistics,
        while preserving the detailed per-agent info in the `info` dictionary.
        """
        # 1. Get the dictionary-based outputs from the multi-agent base class
        obs, rewards_dict, terminateds_dict, truncateds_dict, info = super().step(action)

        # 2. Aggregate reward for the single-agent-style return signature
        aggregated_reward = float(sum(rewards_dict.values())) if rewards_dict else 0.0

        # 3. Extract the global done flags for the single-agent-style return signature
        terminated = terminateds_dict.get("__all__", False)
        truncated = truncateds_dict.get("__all__", False)

        # 4. Pass all detailed, per-agent information through the info dict
        info["agent_rewards"] = rewards_dict
        info["terminateds"] = terminateds_dict
        info["truncateds"] = truncateds_dict

        # 5. Return the data in the format the wrapper expects: (obs, float, bool, bool, dict)
        return obs, aggregated_reward, terminated, truncated, info

    def state(self):
        """Returns the current observation state for all agents."""
        return self.lowlevel_state(self.agent_mode)

    def lowlevel_state(self, mode, agent_id=None, **kwargs):
        """
        Constructs the observation dictionary for the low-level policies.
        """
        state_dict = {}
        ids_to_process = self.observation_space.keys()

        for ag_id in ids_to_process:
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                unit = self.sim.get_unit(ag_id)
                opps = self._nearby_object(ag_id)
                friendlys = self._nearby_object(ag_id, friendly=True)
                fri_id = friendlys[0][0] if friendlys else None
                if opps:
                    if mode == "fight":
                        state = self.fight_state_values(ag_id, unit, opps[0], fri_id)
                    else:
                        state = self.esc_state_values(ag_id, unit, opps, fri_id)
                    self.opp_to_attack[ag_id] = opps[0][0]
                else:
                    state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            else:
                state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)

        if agent_id is not None:
            return {agent_id: state_dict[agent_id]}
        return state_dict

    def _take_action(self, action, info):
        """Coordinates the actions of agents and opponents for one simulation step."""
        self.steps += 1
        self.total_steps_elapsed += self.args.num_workers  # Increment by number of parallel envs
        rewards = {}
        opp_stats = {}
        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                if i <= self.args.num_agents or self.args.level >= 4:
                    if i > self.args.num_agents:
                        actions_for_opp = self._policy_actions(policy_type=self.opp_mode, agent_id=i, unit=u)
                        rewards = self._take_base_action("LowLevel", u, i, self.opp_to_attack.get(i), actions_for_opp, rewards)
                    else:
                        rewards = self._take_base_action("LowLevel", u, i, self.opp_to_attack.get(i), action, rewards)
                        rewards[i] = 0
                        if self.opp_to_attack.get(i) and self.sim.unit_exists(self.opp_to_attack[i]):
                            opp_stats[i] = [self._focus_angle(i, self.opp_to_attack[i], True),
                                            self._distance(i, self.opp_to_attack[i], True)]
                else:
                    if self.args.level == 3: self._hardcoded_opp_level3(u, i)

        self.rewards = self._get_rewards(rewards, self.sim.do_tick(), opp_stats, info)

    def _get_rewards(self, rewards, events, opp_stats, info):
        """Calculates and aggregates rewards from combat events and reward shaping."""
        rews, destroyed_ids, _ = self._combat_rewards(events, opp_stats, mode="LowLevel")

        shaping_reward_scale = 1.0
        if self.args.use_reward_schedule:
            progress = min(1.0, self.total_steps_elapsed / self.args.shaping_decay_timesteps)
            shaping_reward_scale = self.args.shaping_scale_initial * (
                        1 - progress) + self.args.shaping_scale_final * progress

        if self.agent_mode == "fight":
            for agent_id, stats in opp_stats.items():
                if self.sim.unit_exists(agent_id) and agent_id in rews:
                    focus_angle_norm, distance_norm = stats[0], stats[1]

                    distance_reward = (1.0 - distance_norm) * 0.05 * shaping_reward_scale
                    aim_reward = (1.0 - focus_angle_norm) * 0.05 * shaping_reward_scale
                    time_penalty = -0.001
                    rews[agent_id].append(distance_reward + aim_reward + time_penalty)

                    # --- FIX: Ensure info dict is populated for logging ---
                    if "reward_components" not in info:
                        info["reward_components"] = {}

                    # Extract the sparse reward (kill/death) for separate logging
                    kill_reward = sum(r for r in rews.get(agent_id, []) if
                                      r in [self.args.rew_scale, -2 * self.args.rew_scale, -5 * self.args.rew_scale])

                    info["reward_components"][agent_id] = {
                        "aim_reward": aim_reward,
                        "distance_reward": distance_reward,
                        "time_penalty": time_penalty,
                        "kill_reward": kill_reward
                    }

        if self.agent_mode == "escape" and self.args.esc_dist_rew:
            for i in range(1, self.args.num_agents + 1):
                if self.sim.unit_exists(i) and i in rews:
                    opps = self._nearby_object(i)
                    for j, o in enumerate(opps, start=1):
                        if o[2] < 0.06:
                            rews[i].append((-0.02 / j) * shaping_reward_scale)
                        elif o[2] > 0.13:
                            rews[i].append((0.02 / j) * shaping_reward_scale)

        for i in range(1, self.args.num_agents + 1):
            if (self.sim.unit_exists(i) or i in destroyed_ids) and i in rewards:
                if self.args.glob_frac > 0 and self.agent_mode == "fight":
                    other_agent_id = 2 if i == 1 else 1
                    shared_reward = self.args.glob_frac * sum(rews.get(other_agent_id, []))
                    rewards[i] += sum(rews.get(i, [])) + shared_reward
                else:
                    rewards[i] += sum(rews.get(i, []))
        return rewards

    def _hardcoded_opp_level3(self, unit, unit_id):
        """Hardcoded behavior for opponents in curriculum level 3."""
        if self.steps % 60 == 0 and not self.hardcoded_opps_escaping:
            self.hardcoded_opps_escaping = bool(self.np_random.integers(0, 2))
            if self.hardcoded_opps_escaping:
                self.opps_escaping_time = int(self.np_random.uniform(20, 30))
        if self.hardcoded_opps_escaping:
            _, heading, speed, fire, fire_missile, _ = self._escaping_opp(unit)
            self.opps_escaping_time -= 1
            if self.opps_escaping_time <= 0: self.hardcoded_opps_escaping = False
        else:
            opp, heading, speed, fire, fire_missile, _ = self._hardcoded_fight_opp(unit, unit_id)
            if fire_missile and opp and not unit.actual_missile and self.missile_wait[
                unit_id] == 0 and unit.ac_type == 1:
                unit.fire_missile(unit, self.sim.get_unit(opp), self.sim)
                self.missile_wait[unit_id] = 10
        unit.set_heading(heading)
        unit.set_speed(speed)
        if fire: unit.fire_cannon()

    def _escaping_opp(self, unit):
        """Hardcoded logic for an opponent attempting to run to a corner of the map."""
        y, x = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
        if y < 0.5:
            heading = int(self.np_random.uniform(30, 60)) if x < 0.5 else int(self.np_random.uniform(300, 330))
        else:
            heading = int(self.np_random.uniform(120, 150)) if x < 0.5 else int(self.np_random.uniform(210, 240))
        speed = int(self.np_random.uniform(300, 600))
        return None, heading, speed, bool(self.np_random.integers(0, 2)), False, 0

    def _hardcoded_fight_opp(self, opp_unit, opp_id):
        """Hardcoded logic for an opponent in fight mode, turning towards the nearest agent."""
        d_agt = self._nearby_object(opp_id)
        heading, fire, fire_missile, head_rel = opp_unit.heading, False, False, 0
        speed = int(self.np_random.uniform(100, 400))
        if d_agt:
            target_id = d_agt[0][0]
            dist_norm = d_agt[0][1]
            sign = self._correct_angle_sign(opp_unit, self.sim.get_unit(target_id))
            turn_aggressiveness = self.np_random.uniform(0.7, 1.3)
            focus = self._focus_angle(opp_id, target_id)
            if dist_norm > 0.008 and focus > 4:
                head_rel = turn_aggressiveness * sign * focus
                heading = (heading + head_rel) % 360
            if dist_norm > 0.05:
                speed = int(self.np_random.uniform(500, 800)) if focus < 30 else int(self.np_random.uniform(100, 500))
            fire = dist_norm < 0.03 and focus < 10
            fire_missile = dist_norm < 0.09 and focus < 5
        speed = np.clip(speed, 0, opp_unit.max_speed)
        return d_agt[0][0] if d_agt else None, heading, speed, fire, fire_missile, head_rel