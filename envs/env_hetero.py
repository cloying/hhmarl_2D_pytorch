# FILE: envs/env_hetero.py (Full Replacement - Final with Positional Rewards)

import numpy as np
import gymnasium
import os
import torch

from .env_base import HHMARLBaseEnv
from models.ac_models_hetero import FightActorCritic, EscapeActorCritic

# --- Constants ---
OBS_AC1, OBS_AC2 = 26, 24
OBS_ESC_AC1, OBS_ESC_AC2 = 30, 29


def find_latest_checkpoint(path: str) -> str or None:
    if not path or not os.path.exists(path): return None
    checkpoint_dir = os.path.join(path, "checkpoints")
    if not os.path.exists(checkpoint_dir): return None
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
    if not files: return None
    try:
        latest_file = max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
        return os.path.join(checkpoint_dir, latest_file)
    except (ValueError, IndexError):
        return None


class LowLevelEnv(HHMARLBaseEnv):
    def __init__(self, env_config):
        self.args = env_config.get("args")
        self.agent_mode = self.args.agent_mode
        self.opp_mode = "fight"
        self.episode_rewards = {i: 0.0 for i in range(1, self.args.num_agents + 1)}
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.gpu > 0 else "cpu")
        self.current_shaping_scale = self.args.shaping_scale_initial

        obs_spaces, act_spaces, self.obs_dim_map = {}, {}, {}
        for i in range(1, self.args.total_num + 1):
            is_ac1_type = (i - 1) % 2 == 0
            obs_dim_fight, obs_dim_esc = (OBS_AC1, OBS_ESC_AC1) if is_ac1_type else (OBS_AC2, OBS_ESC_AC2)
            obs_dim = obs_dim_fight if self.agent_mode == "fight" else obs_dim_esc
            self.obs_dim_map[i] = {'fight': obs_dim_fight, 'escape': obs_dim_esc}
            if i <= self.args.num_agents:
                obs_spaces[i] = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                act_spaces[i] = gymnasium.spaces.MultiDiscrete([13, 9, 2, 2] if is_ac1_type else [13, 9, 2])
        self.observation_space = gymnasium.spaces.Dict(obs_spaces)
        self.action_space = gymnasium.spaces.Dict(act_spaces)
        self._agent_ids = set(range(1, self.args.num_agents + 1))

        super().__init__(self.args.map_size)
        self.opponent_models = self._load_opponent_policies()
        self.active_opp_policy_key = None

    def _load_opponent_policies(self):
        if self.args.level < 4: return {}
        loaded_models = {}
        if self.args.level >= 4:
            l3_fight_models = self._get_policy_from_level(3, "fight")
            if l3_fight_models: loaded_models.update(l3_fight_models)
        if self.args.level >= 5:
            l4_fight_models = self._get_policy_from_level(4, "fight")
            if l4_fight_models: loaded_models.update(l4_fight_models)
            l3_escape_models = self._get_policy_from_level(3, "escape")
            if l3_escape_models: loaded_models.update(l3_escape_models)
        return loaded_models

    def _get_policy_from_level(self, level, mode):
        restore_dir = f'L{level}_{mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
        restore_path = os.path.join(os.path.dirname(__file__), '..', 'results', restore_dir)
        latest_checkpoint_path = find_latest_checkpoint(restore_path)
        if not latest_checkpoint_path: return {}
        print(f"ENV INFO: Loading opponent policy '{mode}' from level {level}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=self.device)
        models, ModelClass = {}, FightActorCritic if mode == 'fight' else EscapeActorCritic
        for ac_type_id in [1, 2]:
            policy_id = f'ac{ac_type_id}_policy'
            is_ac1_type = (ac_type_id == 1)
            act_space_dummy = gymnasium.spaces.MultiDiscrete([13, 9, 2, 2] if is_ac1_type else [13, 9, 2])
            model_kwargs = {
                'obs_dim_own': self.obs_dim_map[ac_type_id][mode],
                'obs_dim_other': self.obs_dim_map[2 if is_ac1_type else 1][mode],
                'act_parts_own': len(act_space_dummy.nvec),
                'act_parts_other': len(
                    gymnasium.spaces.MultiDiscrete([13, 9, 2, 2] if not is_ac1_type else [13, 9, 2]).nvec),
                'actor_logits_dim': int(np.sum(act_space_dummy.nvec))
            }
            if ModelClass == FightActorCritic:
                model_kwargs['own_state_split_size'] = 12 if is_ac1_type else 10
            else:
                model_kwargs['own_state_split'] = (7, 18) if is_ac1_type else (6, 18)
            model = ModelClass(**model_kwargs).to(self.device)
            model.load_state_dict(checkpoint[f'model_{policy_id}_state_dict'])
            model.eval()
            models[f'L{level}_{mode}_ac{ac_type_id}'] = model
        return models

    def reset(self, *, seed=None, options=None):
        if self.args.level == 5 and self.args.agent_mode == "fight":
            choice = self.np_random.choice(["l3_fight", "l4_fight", "l3_escape"])
            self.opp_mode = "escape" if "escape" in choice else "fight"
            self.active_opp_policy_key = choice.split('_')[0].upper() + "_" + choice.split('_')[1]
        elif self.args.level == 4:
            self.opp_mode = "fight";
            self.active_opp_policy_key = "L3_fight"

        obs, info = super().reset(seed=seed, options={"mode": "LowLevel"})
        self.episode_rewards = {i: 0.0 for i in self._agent_ids}
        return obs, info

    def step(self, action):
        self.steps += 1

        ### --- MODIFIED: Collect opp_stats BEFORE taking action --- ###
        opp_stats = {}
        for i in self._agent_ids:
            if self.sim.unit_exists(i):
                # We need the agent's current target to calculate the positional advantage
                current_target_id = self.opp_to_attack.get(i)
                if current_target_id and self.sim.unit_exists(current_target_id):
                    # `_focus_angle(opponent, me)` -> angle of opp's nose w.r.t line-of-sight to me.
                    # High value is good (they are not pointing at me).
                    advantage_angle = self._focus_angle(current_target_id, i, True)
                    distance = self._distance(i, current_target_id, False)
                    opp_stats[i] = [advantage_angle, distance]

                # Now take the action
                self._take_base_action("LowLevel", self.sim.get_unit(i), i, current_target_id, action)
        ### --- END MODIFICATION --- ###

        opponent_actions = {}
        if self.args.level >= 4:
            with torch.no_grad():
                for i in range(self.args.num_agents + 1, self.args.total_num + 1):
                    if self.sim.unit_exists(i): opponent_actions.update(
                        self._policy_actions(self.opp_mode, i, self.sim.get_unit(i)))

        for i in range(self.args.num_agents + 1, self.args.total_num + 1):
            if self.sim.unit_exists(i):
                unit = self.sim.get_unit(i)
                if self.args.level >= 4:
                    self._take_base_action("LowLevel", unit, i, self.opp_to_attack.get(i), opponent_actions)
                else:
                    self._hardcoded_opp_logic(unit, i)

        events = self.sim.do_tick()

        ### --- MODIFIED: Pass opp_stats to the reward function --- ###
        rewards_dict = self._get_rewards(events, opp_stats)

        for agent_id, reward in rewards_dict.items():
            if agent_id in self.episode_rewards: self.episode_rewards[agent_id] += reward
        terminated = self.alive_agents <= 0 or self.alive_opps <= 0
        truncated = self.steps >= self.args.horizon
        info = {"agent_rewards": rewards_dict}
        if terminated or truncated:
            info["episode"] = {"r": sum(self.episode_rewards.values()), "l": self.steps}
        agg_reward = float(sum(rewards_dict.values()))
        return self.state(), agg_reward, terminated, truncated, info

    def _get_rewards(self, events, opp_stats):
        sparse_rews, _ = self._combat_rewards(events, opp_stats, mode="LowLevel")
        shaping_rews = self._get_shaping_rewards()
        final_rewards = {}
        for agent_id in self._agent_ids:
            scaled_shaping_reward = shaping_rews.get(agent_id, 0.0) * self.current_shaping_scale
            total_reward = sum(sparse_rews.get(agent_id, [])) + scaled_shaping_reward
            if self.args.glob_frac > 0 and self.agent_mode == "fight":
                other_agent_id = 1 if agent_id == 2 else 2
                if other_agent_id in self._agent_ids:
                    total_reward += self.args.glob_frac * sum(sparse_rews.get(other_agent_id, []))
            final_rewards[agent_id] = total_reward
        return final_rewards

    def _get_shaping_rewards(self):
        shaping_rewards = {i: 0.0 for i in self._agent_ids}
        time_penalty = -0.001
        for i in self._agent_ids:
            if self.sim.unit_exists(i):
                shaping_rewards[i] += time_penalty
                opps = self._nearby_object(i)
                if not opps: continue
                closest_opp_id, closest_opp_dist_norm, _ = opps[0]
                if self.agent_mode == 'fight':
                    aspect_reward = (self._aspect_angle(i, closest_opp_id, norm=True) - 0.5) * 0.02
                    dist_reward = np.exp(-((closest_opp_dist_norm - 0.1) ** 2) / 0.05) * 0.01
                    shaping_rewards[i] += aspect_reward + dist_reward
                elif self.agent_mode == 'escape':
                    threat_angle_penalty = -self._aspect_angle(closest_opp_id, i, norm=True) * 0.01
                    threat_dist_penalty = (1.0 - closest_opp_dist_norm) * -0.02
                    shaping_rewards[i] += threat_angle_penalty + threat_dist_penalty
        return shaping_rewards

    # ... (Other methods like _policy_actions, state, etc. are unchanged and omitted for brevity)
    def _policy_actions(self, policy_type, agent_id, unit):
        actions = {}
        opp_idx_in_team = (agent_id - (self.args.num_agents + 1));
        is_ac1_type = opp_idx_in_team % 2 == 0
        ac_type_id = 1 if is_ac1_type else 2
        model_key = f"{self.active_opp_policy_key}_ac{ac_type_id}"
        model = self.opponent_models.get(model_key)
        if not model: return {}
        state_dict = self._get_single_state(policy_type, agent_id, unit)
        if state_dict[agent_id] is None: return {}
        obs_tensor = torch.from_numpy(state_dict[agent_id]).unsqueeze(0).to(self.device)
        logits = model.get_action_logits(obs_tensor)
        act_space_dummy = gymnasium.spaces.MultiDiscrete([13, 9, 2, 2] if is_ac1_type else [13, 9, 2])
        split_sizes = list(act_space_dummy.nvec)
        dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
        action_parts = [torch.argmax(dist.probs, dim=-1) for dist in dists]
        actions[agent_id] = torch.stack(action_parts, dim=-1).cpu().numpy()[0]
        return actions

    def _get_single_state(self, mode, agent_id, unit):
        state_dict = {};
        self.opp_to_attack[agent_id] = None
        opps = self._nearby_object(agent_id);
        friendlys = self._nearby_object(agent_id, friendly=True)
        fri_id = friendlys[0][0] if friendlys else None
        if opps:
            if mode == "fight":
                state = self.fight_state_values(agent_id, unit, opps[0], fri_id)
            else:
                state = self.esc_state_values(agent_id, unit, opps, fri_id)
            self.opp_to_attack[agent_id] = opps[0][0]
        else:
            state = np.zeros(self.obs_dim_map[agent_id][mode], dtype=np.float32)
        state_dict[agent_id] = np.array(state, dtype=np.float32)
        return state_dict

    def state(self):
        state_dict = {}
        for ag_id in self.observation_space.keys():
            if self.sim.unit_exists(ag_id):
                state_dict.update(self._get_single_state(self.agent_mode, ag_id, self.sim.get_unit(ag_id)))
            else:
                state_dict[ag_id] = np.zeros(self.observation_space[ag_id].shape, dtype=np.float32)
        return state_dict

    def _hardcoded_opp_logic(self, unit, unit_id):
        if self.args.level <= 1: return
        if self.args.level == 2:
            if self.steps % self.np_random.integers(35, 46) == 0:
                unit.set_heading(self.np_random.uniform(0, 360))
                unit.set_speed(self.np_random.uniform(100, unit.max_speed))
        elif self.args.level >= 3:
            d_agt = self._nearby_object(unit_id)
            if not d_agt or not self.sim.unit_exists(d_agt[0][0]): return
            target = self.sim.get_unit(d_agt[0][0])
            bearing = self._focus_angle(unit_id, target.id, norm=False)
            sign = self._correct_angle_sign(unit, target)
            turn = np.clip(bearing * sign, -15, 15)
            unit.set_heading((unit.heading + turn) % 360)
            unit.set_speed(unit.max_speed * 0.8 if d_agt[0][1] > 0.1 else unit.max_speed * 0.5)
            if d_agt[0][1] < 0.05 and bearing < 10: unit.fire_cannon()
