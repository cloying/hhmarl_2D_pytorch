"""
High-Level Environment for HHMARL 2D Aircombat.

This environment is designed for training the high-level "commander" agent.
The commander's actions consist of choosing which opponent to target or deciding to
escape. These high-level commands are then executed for a series of sub-steps
by pre-trained low-level policies (fight/escape).

This file is already compatible with a pure PyTorch implementation because it
inherits from the refactored, gymnasium-compliant HHMARLBaseEnv.
"""
# --- Dependencies ---
import os
import random
import numpy as np
from fractions import Fraction
from gymnasium import spaces
from env_base import HHMARLBaseEnv

# --- Constants ---
# Define dimensions for observations and actions in the high-level environment
ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

N_OPP_HL = 2  # Number of opponents the commander can sense
OBS_OPP_HL = 10
OPP_SIZE = N_OPP_HL * OBS_OPP_HL
OBS_FRI_HL = 5
FRI_SIZE = 2 * OBS_FRI_HL
OBS_HL = 14 + N_OPP_HL * OBS_OPP_HL # Total observation dimension for the commander

# --- High-Level Environment Class ---

class HighLevelEnv(HHMARLBaseEnv):
    """
    High-Level Environment for Aircombat Maneuvering.
    """
    def __init__(self, env_config):
        # --- Configuration ---
        self.args = env_config.get("args", None)
        self.n_sub_steps = 15  # Number of low-level steps per high-level action
        self.min_sub_steps = 10

        # --- Define Observation and Action Spaces (using Gymnasium) ---
        # The commander agent observes a high-level state of the battle.
        self.observation_space = spaces.Box(
            low=np.zeros(OBS_HL), high=np.ones(OBS_HL), dtype=np.float32
        )
        # The commander's action is discrete: 0 for escape, 1 to N for attacking an opponent.
        self.action_space = spaces.Discrete(N_OPP_HL + 1)

        self._agent_ids = set(range(1, self.args.num_agents + 1))
        self.commander_actions = None

        # Initialize the parent class and load the pre-trained low-level policies
        super().__init__(self.args.map_size)
        self._get_policies("HighLevel")

    def reset(self, *, seed=None, options=None):
        """Resets the environment for a new high-level episode."""
        super().reset(seed=seed, options={"mode": "HighLevel"})
        self.commander_actions = None
        # Returns initial observations and an empty info dict, per gymnasium API
        return self.state(), {}

    def state(self):
        """
        Constructs and returns the high-level observation for the commander.
        The observation includes the agent's own state, and information about
        nearby friendly and enemy aircraft.
        """
        state_dict = {}
        for ag_id in range(1, self.args.total_num + 1):
            self.opp_to_attack[ag_id] = []
            if ag_id <= self.args.num_agents:
                if self.sim.unit_exists(ag_id):
                    state = []
                    opps = self._nearby_object(ag_id)
                    if opps:
                        unit = self.sim.get_unit(ag_id)
                        x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                        state.extend([x, y, np.clip(unit.speed / unit.max_speed, 0, 1), np.clip((unit.heading % 359) / 359, 0, 1)])

                        opp_state = []
                        for opp in opps:
                            opp_state.extend(self.opp_ac_values("HighLevel", opp[0], ag_id, opp[1]))
                            self.opp_to_attack[ag_id].append(opp)
                            if len(opp_state) >= OPP_SIZE: break

                        if len(opp_state) < OPP_SIZE:
                            opp_state.extend(np.zeros(OPP_SIZE - len(opp_state)))

                        fri_state = []
                        fri = self._nearby_object(ag_id, True)
                        for f in fri:
                            fri_state.extend(self.friendly_ac_values(ag_id, f[0]))
                            if len(fri_state) >= FRI_SIZE: break

                        if len(fri_state) < FRI_SIZE:
                            fri_state.extend(np.zeros(FRI_SIZE - len(fri_state)))

                        state.extend(opp_state)
                        state.extend(fri_state)
                        assert len(state) == OBS_HL, f"Hierarchy state len {len(state)} is not as required ({OBS_HL}) for agent {ag_id}"
                    else:
                        state = np.zeros(OBS_HL, dtype=np.float32)
                else:
                    state = np.zeros(OBS_HL, dtype=np.float32)

                state_dict[ag_id] = np.array(state, dtype=np.float32)
            else:
                 if self.sim.unit_exists(ag_id):
                    self.opp_to_attack[ag_id] = self._nearby_object(ag_id)

        return state_dict

    def lowlevel_state(self, mode, agent_id, unit):
        """
        Constructs the observation required by the low-level policies.
        """
        fri_id = self._nearby_object(agent_id, friendly=True)
        fri_id = fri_id[0][0] if fri_id else None

        if mode == "fight":
            state = self.fight_state_values(agent_id, unit, self.opp_to_attack[agent_id][self.commander_actions[agent_id] - 1], fri_id)
        else: # "escape"
            state = self.esc_state_values(agent_id, unit, self.opp_to_attack[agent_id], fri_id)

        return {agent_id: np.array(state, dtype=np.float32)}

    def _take_action(self, commander_actions):
        """
        Executes a high-level action by running low-level policies for several sub-steps.
        """
        s = 0
        kill_event = False
        situation_event = False
        rewards = {}
        self.commander_actions = commander_actions

        # Initial reward/punishment for the commander's choice
        rewards = self._action_assess(rewards)

        # Low-level simulation loop
        while s < self.n_sub_steps and not kill_event and not situation_event:
            for i in range(1, self.args.total_num + 1):
                if self.sim.unit_exists(i):
                    u = self.sim.get_unit(i)
                    # Determine policy type (escape or fight) from commander's action
                    policy_type = "escape" if self.commander_actions[i] == 0 else "fight"
                    actions = self._policy_actions(policy_type, agent_id=i, unit=u)

                    # Execute the low-level action
                    target_opp = self.opp_to_attack[i][self.commander_actions[i] - 1][0]
                    self._take_base_action("HighLevel", u, i, target_opp, actions)

            # Advance the simulation and get rewards
            rewards, kill_event = self._get_rewards(rewards, self.sim.do_tick())

            if s > self.min_sub_steps:
                situation_event = self._surrounding_event()

            s += 1

        self.steps += s
        self.rewards = rewards
        return

    def _action_assess(self, rewards):
        """Assess the commander's action and provide an initial reward."""
        for i in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                if i <= self.args.num_agents:
                    rewards[i] = 0
                    if self.commander_actions[i] > 0:
                        try:
                            opp_id = self.opp_to_attack[i][self.commander_actions[i] - 1][0]
                            # Reward for choosing to fight in a favorable situation
                            if self.args.hier_action_assess and opp_id:
                                dist = self._distance(i, opp_id)
                                focus_ag_opp = self._focus_angle(i, opp_id)
                                focus_opp_ag = self._focus_angle(opp_id, i)
                                if dist < 0.1 and focus_ag_opp < 15 and focus_opp_ag > 40:
                                    rewards[i] = 0.1
                                else:
                                    rewards[i] = 0
                        except IndexError:
                            # Punish for choosing a non-existent opponent
                            rewards[i] = -0.1
                            self.commander_actions[i] = 1 # Default to attacking the closest
                else: # For opponents, determine their action
                    # Logic to decide if opponent fights or escapes based on a probability
                    p_of = Fraction(self.args.hier_opp_fight_ratio, 100).limit_denominator().as_integer_ratio()
                    fight = bool(random.choices([0, 1], weights=[p_of[1] - p_of[0], p_of[0]], k=1)[0])
                    if fight and self.opp_to_attack[i]:
                        self.commander_actions[i] = random.randint(1, len(self.opp_to_attack[i]))
                    else:
                        self.commander_actions[i] = 0 # Escape

        return rewards

    def _surrounding_event(self):
        """Check for a significant change in the tactical situation."""
        def eval_event(ag_id, opp_id):
            if self._distance(ag_id, opp_id) < 0.1:
                if self._focus_angle(ag_id, opp_id) < 15 or self._focus_angle(opp_id, ag_id) < 15:
                    return True
            return False

        for i in range(1, self.args.num_agents + 1):
            for j in range(self.args.num_agents + 1, self.args.total_num + 1):
                if self.sim.unit_exists(i) and self.sim.unit_exists(j):
                    if eval_event(i, j):
                        return True
        return False

    def _get_rewards(self, rewards, events):
        """Calculates rewards for the high-level step."""
        rews, destroyed_ids, kill_event = self._combat_rewards(events, mode="HighLevel")

        for i in range(1, self.args.num_agents + 1):
            if self.sim.unit_exists(i) or i in destroyed_ids:
                if self.args.glob_frac > 0:
                    # Incorporate shared rewards from friendly agents
                    ids = list(range(1, self.args.num_agents + 1))
                    ids.remove(i)
                    rewards[i] += sum(rews[i]) + self.args.glob_frac * sum(sum(rews[j]) for j in ids)
                else:
                    rewards[i] += sum(rews[i])

        return rewards, kill_event

    def _sample_state(self, agent, i, r):
        # Overrides the base method for high-level scenario setup
        if agent == "agent":
            if r == 1:
                x = random.uniform(7.07, 7.22)
                y = random.uniform(5.07 + i * (0.4 / self.args.num_agents), 5.12 + i * (0.4 / self.args.num_agents))
            else: # r == 2
                x = random.uniform(7.28, 7.43)
                y = random.uniform(5.07 + i * (0.4 / self.args.num_agents), 5.12 + i * (0.4 / self.args.num_agents))
        else: # "opp"
            if r == 1:
                x = random.uniform(7.28, 7.43)
                y = random.uniform(5.07 + i * (0.4 / self.args.num_opps), 5.12 + i * (0.4 / self.args.num_opps))
            else: # r == 2
                x = random.uniform(7.07, 7.22)
                y = random.uniform(5.07 + i * (0.4 / self.args.num_opps), 5.12 + i * (0.4 / self.args.num_opps))

        a = random.randint(0, 359)
        return x, y, a