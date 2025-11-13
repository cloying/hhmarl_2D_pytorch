# FILE: train_hetero.py (Use this version)

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import tqdm
from config import Config
from envs.env_hetero import LowLevelEnv
from models.ac_models_hetero import EscapeActorCritic, FightActorCritic

if __name__ == "__main__":
    args = Config(0).get_arguments
    total_timesteps, learning_rate, num_steps = args.total_timesteps, 1e-4, 2048
    num_update_epochs, num_mini_batches, gamma, gae_lambda = 10, 32, 0.99, 0.95
    clip_coef, ent_coef, vf_coef, max_grad_norm = 0.2, 0.01, 0.5, 0.5
    num_workers = args.num_workers
    batch_size = int(num_steps * num_workers)
    mini_batch_size = int(batch_size // num_mini_batches)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")
    run_name = f"L{args.level}_{args.agent_mode}_{args.num_agents}v{args.num_opps}_{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(run_dir)
    print(f"--- Training on {device} | Run: {run_name} ---")


    def make_env(worker_id=0):
        args.env_config["run_dir"] = run_dir
        env = LowLevelEnv(args.env_config)
        if worker_id == 0: env.is_debug_worker = True
        return env


    env_fns = [lambda i=i: make_env(i) for i in range(num_workers)]
    vec_env = gym.vector.SyncVectorEnv(env_fns)

    agent_ids = sorted(list(vec_env.single_observation_space.keys()))
    models, optimizers, policy_map = {}, {}, {}
    for agent_id in agent_ids:
        is_ac1_type, policy_id = (agent_id - 1) % 2 == 0, 'ac1_policy' if (agent_id - 1) % 2 == 0 else 'ac2_policy'
        policy_map[agent_id] = policy_id
        if policy_id not in models:
            obs_space, act_space = vec_env.single_observation_space, vec_env.single_action_space
            own_id, other_id = (1, 2) if is_ac1_type else (2, 1)
            ModelClass = FightActorCritic if args.agent_mode == 'fight' else EscapeActorCritic
            model_kwargs = {'obs_dim_own': obs_space[own_id].shape[0], 'obs_dim_other': obs_space[other_id].shape[0],
                            'act_parts_own': len(act_space[own_id].nvec),
                            'act_parts_other': len(act_space[other_id].nvec),
                            'actor_logits_dim': int(np.sum(act_space[own_id].nvec))}
            if ModelClass == FightActorCritic:
                model_kwargs['own_state_split_size'] = 12 if is_ac1_type else 10
            else:
                model_kwargs['own_state_split'] = (7, 18) if is_ac1_type else (6, 18)
            model = ModelClass(**model_kwargs).to(device)
            models[policy_id], optimizers[policy_id] = model, optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)


    class RolloutBuffer:
        def __init__(self):
            self.obs = {
                i: np.zeros((num_steps, num_workers, *vec_env.single_observation_space[i].shape), dtype=np.float32) for
                i in agent_ids}
            self.actions = {
                i: np.zeros((num_steps, num_workers, len(vec_env.single_action_space[i].nvec)), dtype=np.int64) for i in
                agent_ids}
            self.logprobs, self.rewards = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}, {
                i: np.zeros((num_steps, num_workers)) for i in agent_ids}
            self.dones, self.values = np.zeros((num_steps, num_workers)), {i: np.zeros((num_steps, num_workers)) for i
                                                                           in agent_ids}
            self.step = 0

        def add(self, obs, actions, logprobs, rewards, dones, values):
            for agent_id in agent_ids:
                if agent_id in obs:
                    self.obs[agent_id][self.step], self.actions[agent_id][self.step], self.logprobs[agent_id][
                        self.step] = obs[agent_id], actions[agent_id], logprobs[agent_id]
                    self.rewards[agent_id][self.step], self.values[agent_id][self.step] = rewards.get(agent_id,
                                                                                                      np.zeros(
                                                                                                          num_workers)), \
                    values[agent_id]
            self.dones[self.step] = dones
            self.step = (self.step + 1) % num_steps


    buffer, num_updates = RolloutBuffer(), total_timesteps // batch_size
    next_obs, _ = vec_env.reset()
    next_done, episodic_returns, global_step = np.zeros(num_workers), deque(maxlen=50), 0
    pbar = tqdm.trange(1, num_updates + 1)
    for update in pbar:
        for step in range(num_steps):
            global_step += num_workers
            with torch.no_grad():
                actions, logprobs, values = {}, {}, {}
                obs_tensors = {i: torch.from_numpy(next_obs[i]).to(device) for i in agent_ids}
                for agent_id, obs_tensor in obs_tensors.items():
                    logits = models[policy_map[agent_id]].get_action_logits(obs_tensor)
                    dists = [torch.distributions.Categorical(logits=l) for l in
                             logits.split(list(vec_env.single_action_space[agent_id].nvec), dim=-1)]
                    action_parts = [dist.sample() for dist in dists]
                    actions[agent_id], logprobs[agent_id] = torch.stack(action_parts, dim=-1).cpu().numpy(), torch.sum(
                        torch.stack([d.log_prob(a) for d, a in zip(dists, action_parts)]), dim=0).cpu().numpy()
                for policy_id, model in models.items():
                    main_agent = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent = next(aid for aid in agent_ids if aid != main_agent)
                    critic_obs_dict = {'obs_1_own': obs_tensors[main_agent],
                                       'act_1_own': torch.from_numpy(actions[main_agent]).to(device,
                                                                                             dtype=torch.float32),
                                       'obs_2': obs_tensors[other_agent],
                                       'act_2': torch.from_numpy(actions[other_agent]).to(device, dtype=torch.float32)}
                    value = model.get_value(critic_obs_dict).cpu().numpy()
                    for aid, pid in policy_map.items():
                        if pid == policy_id: values[aid] = value
            new_next_obs, _, terminateds, truncateds, infos = vec_env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            rewards_per_agent = {agent_id: np.zeros(num_workers) for agent_id in agent_ids}
            if "agent_rewards" in infos:
                for i, worker_rewards_dict in enumerate(infos["agent_rewards"]):
                    if worker_rewards_dict:
                        for agent_id, reward in worker_rewards_dict.items():
                            if agent_id in rewards_per_agent: rewards_per_agent[agent_id][i] = reward
            buffer.add(next_obs, actions, logprobs, rewards_per_agent, dones, values)
            next_obs, next_done = new_next_obs, dones
            if "_final_info" in infos and infos["_final_info"].any():
                for info in infos["final_info"]:
                    if info and "episode" in info: episodic_returns.append(info["episode"]["r"].sum())
        with torch.no_grad():
            next_values = {}
            next_obs_tensors = {i: torch.from_numpy(next_obs[i]).to(device) for i in agent_ids}
            for policy_id, model in models.items():
                main_agent = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                other_agent = next(aid for aid in agent_ids if aid != main_agent)
                dummy_actions = {i: torch.zeros(num_workers, len(vec_env.single_action_space[i].nvec), device=device,
                                                dtype=torch.float32) for i in agent_ids}
                critic_obs_dict = {'obs_1_own': next_obs_tensors[main_agent], 'act_1_own': dummy_actions[main_agent],
                                   'obs_2': next_obs_tensors[other_agent], 'act_2': dummy_actions[other_agent]}
                next_val = model.get_value(critic_obs_dict).cpu().numpy()
                for aid, pid in policy_map.items():
                    if pid == policy_id: next_values[aid] = next_val
        advantages, returns = {}, {}
        for agent_id in agent_ids:
            adv, last_gae_lam = np.zeros_like(buffer.rewards[agent_id]), 0
            for t in reversed(range(num_steps)):
                next_non_terminal = 1.0 - (next_done if t == num_steps - 1 else buffer.dones[t + 1])
                next_val = next_values[agent_id] if t == num_steps - 1 else buffer.values[agent_id][t + 1]
                delta = buffer.rewards[agent_id][t] + gamma * next_val * next_non_terminal - buffer.values[agent_id][t]
                adv[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[agent_id], returns[agent_id] = adv, adv + buffer.values[agent_id]
        b_obs = {i: buffer.obs[i].reshape((-1, *vec_env.single_observation_space[i].shape)) for i in agent_ids}
        b_actions, b_logprobs = {i: buffer.actions[i].reshape((-1, len(vec_env.single_action_space[i].nvec))) for i in
                                 agent_ids}, {i: buffer.logprobs[i].reshape(-1) for i in agent_ids}
        b_advantages, b_returns = {i: advantages[i].reshape(-1) for i in agent_ids}, {i: returns[i].reshape(-1) for i in
                                                                                      agent_ids}
        for epoch in range(num_update_epochs):
            inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = inds[start:end]
                for policy_id, model in models.items():
                    main_agent = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent = next(aid for aid in agent_ids if aid != main_agent)
                    critic_obs_dict = {'obs_1_own': torch.from_numpy(b_obs[main_agent][mb_inds]).to(device),
                                       'act_1_own': torch.from_numpy(b_actions[main_agent][mb_inds]).to(device,
                                                                                                        dtype=torch.float32),
                                       'obs_2': torch.from_numpy(b_obs[other_agent][mb_inds]).to(device),
                                       'act_2': torch.from_numpy(b_actions[other_agent][mb_inds]).to(device,
                                                                                                     dtype=torch.float32)}
                    logits, new_values = model(critic_obs_dict['obs_1_own'], critic_obs_dict)
                    dists = [torch.distributions.Categorical(logits=l) for l in
                             logits.split(list(vec_env.single_action_space[main_agent].nvec), dim=-1)]
                    new_logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, torch.from_numpy(
                        b_actions[main_agent][mb_inds]).to(device).T)]), dim=0)
                    entropy, logratio = torch.sum(torch.stack([d.entropy() for d in dists]),
                                                  dim=0), new_logprobs - torch.from_numpy(
                        b_logprobs[main_agent][mb_inds]).to(device)
                    ratio, mb_adv = logratio.exp(), torch.from_numpy(b_advantages[main_agent][mb_inds]).to(device)
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                    v_loss = 0.5 * (
                                (new_values - torch.from_numpy(b_returns[main_agent][mb_inds]).to(device)) ** 2).mean()
                    loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
                    optimizers[policy_id].zero_grad();
                    loss.backward();
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm);
                    optimizers[policy_id].step()

        avg_return = np.mean(episodic_returns) if len(episodic_returns) > 0 else 0.0
        pbar.set_description(f"Update {update}, Avg Return: {avg_return:.5f}, Global Step: {global_step}")
        writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)

    vec_env.close();
    writer.close()