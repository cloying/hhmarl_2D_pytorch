"""
Low-Level Environment for HHMARL 2D Aircombat.

This environment is used for training the basic "fight" and "escape" policies
through a curriculum of increasing difficulty.
"""
import numpy as np
import gymnasium

# --- Local Project Imports (This MUST be relative) ---
# It correctly imports `HHMARLBaseEnv` from `env_base.py` which is in the same directory (`envs/`).
from .env_base import HHMARLBaseEnv

# --- Constants ---
# These define the raw observation dimensions for the two types of aircraft in different modes.
OBS_AC1 = 26  # Observation size for Aircraft Type 1 (missile-capable) in fight mode.
OBS_AC2 = 24  # Observation size for Aircraft Type 2 (cannon-only) in fight mode.
OBS_ESC_AC1 = 30  # Observation size for AC1 in escape mode.
OBS_ESC_AC2 = 29  # Observation size for AC2 in escape mode.


class LowLevelEnv(HHMARLBaseEnv):
    """
    Low-Level Environment for Aircombat Maneuvering.

    This class dynamically constructs its observation and action spaces
    based on the number of agents specified in the configuration, making it
    flexible for different scenarios (e.g., 1v1, 2v2).
    """

    def __init__(self, env_config):
        """
        Initializes the low-level environment.

        Args:
            env_config (dict): A dictionary containing the configuration arguments,
                               including the number of agents, agent mode, etc.
        """
        # Store the configuration arguments passed from the training script.
        self.args = env_config.get("args", None)
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"

        # --- MODIFICATION: Dynamically build spaces instead of hardcoding ---

        # Dictionaries to hold the dynamically created spaces and dimension info.
        obs_spaces = {}
        act_spaces = {}
        self.obs_dim_map = {}

        # The aircraft types alternate: Agent 1 is AC1, Agent 2 is AC2, Agent 3 is AC1, etc.
        # We loop from 1 to the configured number of agents to build the spaces.
        for i in range(1, self.args.num_agents + 1):
            # Agent IDs 1, 3, 5... are Type 1 (Rafale w/ missiles).
            # Agent IDs 2, 4, 6... are Type 2 (RafaleLong cannon-only).
            is_ac1_type = (i - 1) % 2 == 0

            # Determine observation dimension based on agent mode and aircraft type.
            if self.agent_mode == "fight":
                obs_dim = OBS_AC1 if is_ac1_type else OBS_AC2
            else:  # "escape"
                obs_dim = OBS_ESC_AC1 if is_ac1_type else OBS_ESC_AC2

            # Store the observation dimension for this agent for later use.
            self.obs_dim_map[i] = obs_dim

            # Create the Gymnasium observation space for this agent.
            obs_spaces[i] = gymnasium.spaces.Box(
                low=np.zeros(obs_dim), high=np.ones(obs_dim), dtype=np.float32
            )

            # Create the Gymnasium action space for this agent.
            if is_ac1_type:
                # AC1 has 4 action parts (heading, speed, cannon, missile).
                act_spaces[i] = gymnasium.spaces.MultiDiscrete([13, 9, 2, 2])
            else:
                # AC2 has 3 action parts (heading, speed, cannon).
                act_spaces[i] = gymnasium.spaces.MultiDiscrete([13, 9, 2])

        # Assign the dynamically created dictionaries to the class attributes.
        self.observation_space = gymnasium.spaces.Dict(obs_spaces)
        self.action_space = gymnasium.spaces.Dict(act_spaces)

        # --- END MODIFICATION ---

        # The set of agent IDs is now consistent with the created spaces.
        self._agent_ids = set(range(1, self.args.num_agents + 1))

        # Initialize the base environment class AFTER the spaces are defined.
        super().__init__(self.args.map_size)

        # Load pre-trained policies for self-play opponents if required by the curriculum level.
        if self.args.level >= 4:
            self._get_policies("LowLevel")
        if self.args.level == 5 and self.agent_mode == "escape":
            self.opp_mode = "fight"

    def reset(self, *, seed=None, options=None):
        """Resets the environment and handles curriculum logic for level 5."""
        # The base reset will handle seeding and creating the scenario.
        obs, info = super().reset(seed=seed, options={"mode": "LowLevel"})

        # For level 5 fight mode, randomly select an opponent policy from the pre-trained pool.
        if self.args.level == 5 and self.args.agent_mode == "fight":
            # --- MODIFICATION: Use self.np_random for reproducible curriculum ---
            # Randomly choose between opponent policies from level 3, 4, or 5 (escape).
            k = self.np_random.integers(3, 6)  # Exclusive upper bound, so it's 3, 4, or 5.
            self.policy = self.policies[k]
            self.opp_mode = "escape" if k == 5 else "fight"

        return obs, info

    def state(self):
        """Returns the current observation state for all agents."""
        return self.lowlevel_state(self.agent_mode)

    def lowlevel_state(self, mode, agent_id=None, **kwargs):
        """Constructs the observation dictionary for the low-level policies."""

        def fri_ac_id(ag_id):
            """Helper to find the ID of the other friendly agent in a 2-agent team."""
            if ag_id <= self.args.num_agents:
                return 1 if ag_id == 2 else 2
            else:
                # This logic is for opponents, assuming a similar pairing.
                opp_base = self.args.num_agents
                return (opp_base + 1) if ag_id == (opp_base + 2) else (opp_base + 2)

        state_dict = {}
        # If a specific agent_id is requested, loop only for that one (used by self-play).
        start, end = (agent_id, agent_id + 1) if agent_id else (1, self.args.num_agents + 1)

        for ag_id in range(start, end):
            self.opp_to_attack[ag_id] = None
            if self.sim.unit_exists(ag_id):
                opps = self._nearby_object(ag_id)
                if opps:  # If there are opponents nearby
                    unit = self.sim.get_unit(ag_id)
                    # Get the state vector based on whether the agent is fighting or escaping.
                    if mode == "fight":
                        state = self.fight_state_values(ag_id, unit, opps[0], fri_ac_id(ag_id))
                    else:  # "escape"
                        state = self.esc_state_values(ag_id, unit, opps, fri_ac_id(ag_id))

                    self.opp_to_attack[ag_id] = opps[0][0]  # Target the closest opponent
                else:  # No opponents nearby, return a zero vector.
                    state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)
            else:  # Agent is destroyed, return a zero vector.
                state = np.zeros(self.obs_dim_map[ag_id], dtype=np.float32)

            state_dict[ag_id] = np.array(state, dtype=np.float32)
        return state_dict

    def _take_action(self, action):
        """Coordinates the actions of agents and opponents for one simulation step."""
        self.steps += 1
        rewards = {}
        opp_stats = {}  # To store data for reward shaping

        # Process actions for all units in the simulation.
        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                u = self.sim.get_unit(i)
                # If the unit is a trainable agent OR an opponent controlled by a learned policy...
                if i <= self.args.num_agents or self.args.level >= 4:
                    if i > self.args.num_agents:  # Opponent using self-play policy
                        actions = self._policy_actions(policy_type=self.opp_mode, agent_id=i, unit=u)
                    else:  # Trainable agent
                        actions = action
                        rewards[i] = 0
                        # Store distance/angle to target for reward shaping
                        if self.opp_to_attack.get(i) and self.sim.unit_exists(self.opp_to_attack[i]):
                            opp_stats[i] = [self._focus_angle(i, self.opp_to_attack[i], True),
                                            self._distance(i, self.opp_to_attack[i], True)]

                    rewards = self._take_base_action("LowLevel", u, i, self.opp_to_attack.get(i), actions, rewards)

                # Opponent using hardcoded script (for curriculum levels 1-3)
                else:
                    if self.args.level == 3: self._hardcoded_opp_level3(u, i)
                    # Note: Levels 1 and 2 opponents are static/random and their logic
                    # is handled directly inside the simulator's unit update, not here.

        # Advance the simulation by one tick and calculate rewards.
        self.rewards = self._get_rewards(rewards, self.sim.do_tick(), opp_stats)

    def _get_rewards(self, rewards, events, opp_stats):
        """Calculates and aggregates rewards from combat events and reward shaping."""
        # Get base rewards from kills/deaths.
        rews, destroyed_ids, _ = self._combat_rewards(events, opp_stats, mode="LowLevel")

        # --- Reward Shaping for "fight" mode ---
        if self.agent_mode == "fight":
            for agent_id, stats in opp_stats.items():
                if self.sim.unit_exists(agent_id):
                    focus_angle_norm, distance_norm = stats[0], stats[1]
                    # Reward for getting closer
                    distance_reward = (1.0 - distance_norm) * 0.01
                    # Reward for aiming correctly
                    aim_reward = (1.0 - focus_angle_norm) * 0.01
                    # Small penalty per step to encourage efficiency
                    time_penalty = -0.001
                    rews[agent_id].append(distance_reward + aim_reward + time_penalty)

        # Reward shaping for "escape" mode.
        if self.agent_mode == "escape" and self.args.esc_dist_rew:
            for i in range(1, self.args.num_agents + 1):
                if self.sim.unit_exists(i) and i in rews:
                    opps = self._nearby_object(i)
                    for j, o in enumerate(opps, start=1):
                        # Penalize being too close, reward being far away.
                        if o[2] < 0.06:
                            rews[i].append(-0.02 / j)
                        elif o[2] > 0.13:
                            rews[i].append(0.02 / j)

        # Aggregate all rewards for each agent, applying reward sharing if enabled.
        for i in range(1, self.args.num_agents + 1):
            if (self.sim.unit_exists(i) or i in destroyed_ids) and i in rewards:
                if self.args.glob_frac > 0 and self.agent_mode == "fight":
                    other_agent_id = 2 if i == 1 else 1  # Assumes 2 agents
                    shared_reward = self.args.glob_frac * sum(rews.get(other_agent_id, []))
                    rewards[i] += sum(rews.get(i, [])) + shared_reward
                else:
                    rewards[i] += sum(rews.get(i, []))
        return rewards

    # --- Hardcoded Opponent Logic for Curriculum ---
    def _hardcoded_opp_level3(self, unit, unit_id):
        """Hardcoded behavior for opponents in curriculum level 3."""
        # Decide whether to switch to escaping behavior.
        if self.steps % 60 == 0 and not self.hardcoded_opps_escaping:
            self.hardcoded_opps_escaping = bool(self.np_random.integers(0, 2))
            if self.hardcoded_opps_escaping:
                self.opps_escaping_time = int(self.np_random.uniform(20, 30))

        # Execute behavior.
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

            # Turn to engage
            if dist_norm > 0.008 and focus > 4:
                head_rel = turn_aggressiveness * sign * focus
                heading = (heading + head_rel) % 360

            # Adjust speed based on range and angle
            if dist_norm > 0.05:
                speed = int(self.np_random.uniform(500, 800)) if focus < 30 else int(self.np_random.uniform(100, 500))

            # Decide to fire weapons
            fire = dist_norm < 0.03 and focus < 10
            fire_missile = dist_norm < 0.09 and focus < 5

        speed = np.clip(speed, 0, opp_unit.max_speed)
        return d_agt[0][0] if d_agt else None, heading, speed, fire, fire_missile, head_rel