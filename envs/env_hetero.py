# FILE: envs/env_hetero.py (Corrected with dtype=torch.float32)

import numpy as np
import gymnasium
import torch
import os
from .env_base import HHMARLBaseEnv
from models.ac_models_hetero import RecurrentActor, NODE_FEATURE_DIM

# Constants
OBS_AC1, OBS_AC2 = 26, 24
OBS_ESC_AC1, OBS_ESC_AC2 = 30, 29


class LowLevelEnv(HHMARLBaseEnv):
    def __init__(self, env_config):
        self.args = env_config.get("args")
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"
        self.episode_rewards = {i: 0.0 for i in range(1, self.args.num_agents + 1)}
        self.global_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.gpu > 0 else "cpu")

        obs_spaces, act_spaces, self.obs_dim_map = {}, {}, {}
        for i in range(1, self.args.num_agents + 1):
            is_ac1_type = (i % 2) != 0
            obs_dim = (OBS_AC1 if is_ac1_type else OBS_AC2)
            self.obs_dim_map[i] = obs_dim
            obs_spaces[i] = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            act_spaces[i] = gymnasium.spaces.MultiDiscrete([13, 9, 2, 2] if is_ac1_type else [13, 9, 2])

        self.observation_space = gymnasium.spaces.Dict(obs_spaces)
        self.action_space = gymnasium.spaces.Dict(act_spaces)
        self._agent_ids = set(range(1, self.args.num_agents + 1))
        super().__init__(self.args.map_size)

        self.opponent_actors = {}
        self.opponent_hidden_states = {}
        if self.args.level == 4 and self.args.opponent_policy_path:
            print(f"--- LEVEL 4: Loading opponent policies from {self.args.opponent_policy_path} ---")
            try:
                opp_actor1 = RecurrentActor(obs_dim_own=OBS_AC1, actor_logits_dim=13 + 9 + 2 + 2).to(self.device)
                opp_actor1.load_state_dict(
                    torch.load(os.path.join(self.args.opponent_policy_path, "L3_Continuous_AC1_fight.pth")))
                opp_actor1.eval()
                self.opponent_actors['ac1_policy'] = opp_actor1
                opp_actor2 = RecurrentActor(obs_dim_own=OBS_AC2, actor_logits_dim=13 + 9 + 2).to(self.device)
                opp_actor2.load_state_dict(
                    torch.load(os.path.join(self.args.opponent_policy_path, "L3_Continuous_AC2_fight.pth")))
                opp_actor2.eval()
                self.opponent_actors['ac2_policy'] = opp_actor2
            except FileNotFoundError as e:
                print(
                    f"ERROR: Could not load opponent policies for Level 4 self-play. Make sure L1-3 training is complete.")
                raise e

    def set_global_step(self, step):
        self.global_step = step

    def reset(self, *, seed=None, options=None):
        self.opponent_hidden_states = {i: torch.zeros(1, 1, model.hidden_size, device=self.device)
                                       for i, model in self.opponent_actors.items()}
        obs, info = super().reset(seed=seed, options={"mode": "LowLevel"})
        self.episode_rewards = {i: 0.0 for i in self._agent_ids}
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
        opponent_actions = {}
        for i in range(self.args.num_agents + 1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                if self.args.level == 4 and self.opponent_actors:
                    opponent_actions.update(self._get_opponent_policy_action(unit, i))
                else:
                    self._hardcoded_opp_logic(unit, i)
        for opp_id, opp_action in opponent_actions.items():
            if self.sim.unit_exists(opp_id):
                opp_target = self._nearby_object(opp_id)[0][0] if self._nearby_object(opp_id) else None
                self._take_base_action("LowLevel", self.sim.get_unit(opp_id), opp_id, opp_target, {opp_id: opp_action})
        events = self.sim.do_tick()
        combat_rewards, shaping_rewards = self._get_rewards(events, opp_stats, action)
        for agent_id in self._agent_ids:
            total_reward = combat_rewards.get(agent_id, 0.0) + shaping_rewards.get(agent_id, 0.0)
            if agent_id in self.episode_rewards: self.episode_rewards[agent_id] += total_reward
        terminated = self.alive_agents <= 0 or self.alive_opps <= 0
        truncated = self.steps >= self.args.horizon
        done = terminated or truncated
        info = {"agent_rewards": combat_rewards, "shaping_rewards": shaping_rewards,
                "graph_data": self._get_graph_state()}
        if done: info["episode"] = {"r": sum(self.episode_rewards.values()), "l": self.steps}
        agg_reward = float(sum(combat_rewards.values()) + sum(shaping_rewards.values()))
        return self.state(), agg_reward, terminated, truncated, info

    def _get_opponent_policy_action(self, unit, unit_id):
        actions = {}
        is_ac1_type = (unit_id - self.args.num_agents - 1) % 2 == 0
        policy_id = 'ac1_policy' if is_ac1_type else 'ac2_policy'
        actor = self.opponent_actors[policy_id]
        opp_obs_dict = self.state(for_opponent_id=unit_id)
        if unit_id not in opp_obs_dict: return {}
        obs_np = opp_obs_dict[unit_id]
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        hidden_state = self.opponent_hidden_states.get(policy_id,
                                                       torch.zeros(1, 1, actor.hidden_size, device=self.device))
        with torch.no_grad():
            logits, new_hidden_state = actor(obs_tensor, hidden_state)
        self.opponent_hidden_states[policy_id] = new_hidden_state
        action_space_vec = [13, 9, 2, 2] if is_ac1_type else [13, 9, 2]
        action_parts = [torch.argmax(part, dim=-1) for part in logits.split(action_space_vec, dim=-1)]
        actions[unit_id] = torch.stack(action_parts, dim=-1).cpu().numpy()[0]
        return actions

    def _hardcoded_opp_logic(self, unit, unit_id):
        if self.global_step < 800_000:
            unit.set_speed(0);
            return
        elif self.global_step < 2_500_000:
            progress = (self.global_step - 800_000) / (1_700_000)
            d_agt = self._nearby_object(unit_id)
            if not d_agt or not self.sim.unit_exists(d_agt[0][0]):
                if self.steps % 20 == 0:
                    unit.set_speed(self.np_random.uniform(0, unit.max_speed * 0.5 * progress))
                    unit.set_heading((unit.heading + self.np_random.uniform(-90 * progress, 90 * progress)) % 360)
                return
            target_agent = self.sim.get_unit(d_agt[0][0])
            bearing = self._focus_angle(unit_id, target_agent.id, norm=False);
            sign = self._correct_angle_sign(unit, target_agent)
            turn = np.clip(bearing * sign, -15, 15) * progress
            unit.set_heading((unit.heading + turn) % 360)
            speed_multiplier = 0.8 if d_agt[0][1] > 0.1 else 0.5
            unit.set_speed(unit.max_speed * speed_multiplier * progress)
            if d_agt[0][1] < 0.05 and bearing < 10 and self.np_random.random() < (0.5 * progress): unit.fire_cannon()
        else:
            d_agt = self._nearby_object(unit_id)
            if not d_agt or not self.sim.unit_exists(d_agt[0][0]): return
            target_agent = self.sim.get_unit(d_agt[0][0])
            bearing = self._focus_angle(unit_id, target_agent.id, norm=False);
            sign = self._correct_angle_sign(unit, target_agent)
            turn = np.clip(bearing * sign, -15, 15)
            unit.set_heading((unit.heading + turn) % 360)
            if d_agt[0][1] > 0.1:
                unit.set_speed(unit.max_speed * 0.8)
            else:
                unit.set_speed(unit.max_speed * 0.5)
            if d_agt[0][1] < 0.05 and bearing < 10: unit.fire_cannon()

    def _get_graph_state(self):
        active_units, node_features = [], []
        for unit_id in range(1, self.args.total_num + 1):
            if self.sim.unit_exists(unit_id):
                active_units.append(unit_id)
                unit = self.sim.get_unit(unit_id)
                x, y = self.map_limits.relative_position(unit.position.lat, unit.position.lon)
                team_id = [1.0, 0.0] if unit.group == "agent" else [0.0, 1.0]
                missile_ammo = np.clip(unit.missile_remain / unit.rocket_max, 0,
                                       1) if unit.ac_type == 1 and unit.rocket_max > 0 else 0.0
                features = [x, y, np.clip(unit.speed / unit.max_speed, 0, 1), np.clip(unit.heading / 360.0, 0, 1),
                            np.clip(unit.cannon_remain_secs / unit.cannon_max, 0, 1), missile_ammo,
                            1.0 if unit.cannon_current_burst_secs > 0 else 0.0,
                            1.0 if unit.ac_type == 1 else 0.0] + team_id
                node_features.append(features)
        if not node_features:
            return {"x": torch.empty((0, NODE_FEATURE_DIM), dtype=torch.float32),
                    "edge_index": torch.empty((2, 0), dtype=torch.long)}

        num_nodes = len(active_units)
        edge_index = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]

        ### --- FIX: Explicitly set dtype to float32 --- ###
        return {
            "x": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        }
        ### ---------------------------------------------- ###

    def state(self, for_opponent_id=None):
        state_dict = {}
        agent_pool = [for_opponent_id] if for_opponent_id else self.observation_space.keys()
        for ag_id in agent_pool:
            if self.sim.unit_exists(ag_id):
                unit = self.sim.get_unit(ag_id)
                opps, friendlys = self._nearby_object(ag_id), self._nearby_object(ag_id, friendly=True)
                fri_id = friendlys[0][0] if friendlys else None
                if opps:
                    state = self.fight_state_values(ag_id, unit, opps[0], fri_id)
                    self.opp_to_attack[ag_id] = opps[0][0]
                else:
                    obs_dim = self.obs_dim_map.get(ag_id, OBS_AC1 if unit.ac_type == 1 else OBS_AC2)
                    state = np.zeros(obs_dim, dtype=np.float32)
            else:
                obs_dim = self.obs_dim_map.get(ag_id, OBS_AC1)
                state = np.zeros(obs_dim, dtype=np.float32)
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
                if other_agent_id in self._agent_ids: sparse_reward += self.args.glob_frac * sum(
                    combat_rews_dict.get(other_agent_id, []))
            final_combat_rewards[agent_id] = sparse_reward
        return final_combat_rewards, shaping_rews

    def _get_shaping_rewards(self, actions):
        shaping_rewards = {i: 0.0 for i in self._agent_ids}
        time_penalty = -0.001

        for i in self._agent_ids:
            if not self.sim.unit_exists(i):
                continue

            # Basic time penalty to encourage efficiency
            shaping_rewards[i] += time_penalty

            opps = self._nearby_object(i)
            if not opps or not self.sim.unit_exists(opps[0][0]):
                continue

            closest_opp_id, closest_opp_dist_norm, _ = opps[0]
            agent_unit = self.sim.get_unit(i)
            opponent_unit = self.sim.get_unit(closest_opp_id)

            # --- Calculate Key Tactical Angles (using non-normalized degrees for clarity) ---
            agent_focus_angle_deg = self._focus_angle(i, closest_opp_id, norm=False)
            opponent_aspect_angle_deg = self._aspect_angle(closest_opp_id, i, norm=False)

            # --- Proposal 1: "Tail Chase" Bonus ---
            # Condition: Agent is pointing at the opponent (focus < 20 deg) AND
            #            is in the opponent's rear hemisphere (aspect < 30 deg).
            is_in_tail_chase = agent_focus_angle_deg < 20 and opponent_aspect_angle_deg < 30
            if is_in_tail_chase:
                shaping_rewards[i] += self.args.tail_chase_bonus

            # --- Proposal 3: "Energy Advantage" Bonus ---
            if agent_unit.speed > opponent_unit.speed:
                shaping_rewards[i] += self.args.energy_advantage_bonus

            # --- Proposal 2: "High-Probability Kill" Reward ---
            agent_action = actions.get(i)
            if agent_action is not None and (agent_action[2] == 1 or (len(agent_action) > 3 and agent_action[3] == 1)):
                is_firing = True
            else:
                is_firing = False

            if is_firing:
                # Condition for a "good shot": Firing while in a tail chase at close range.
                is_good_shot = is_in_tail_chase and closest_opp_dist_norm < 0.2
                if is_good_shot:
                    shaping_rewards[i] += self.args.high_prob_kill_reward
                else:
                    # Penalize wasteful shots
                    shaping_rewards[i] += self.args.ammo_penalty

        return shaping_rewards