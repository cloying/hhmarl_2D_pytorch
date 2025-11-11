# FILE: train_hetero.py (FINAL - Corrected for Device Placement, Style, and Diagnostics)

# --- Core Dependencies ---
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
from collections import defaultdict, deque

# --- Utility and Logging Dependencies ---
from torch.utils.tensorboard import SummaryWriter
import tqdm
from pathlib import Path

# --- Local Project Imports ---
from config import Config
from envs.env_hetero import LowLevelEnv
from models.ac_models_hetero import EscapeActorCritic, FightActorCritic

class VectorizedRolloutBuffer:
    def __init__(self, num_steps, num_workers, agent_ids, obs_spaces, act_spaces, device):
        self.num_steps = num_steps
        self.num_workers = num_workers
        self.agent_ids = agent_ids
        self.device = device
        self.obs = {i: torch.zeros((num_steps, num_workers, *obs_spaces[i].shape), device=device) for i in agent_ids}
        self.actions = {i: torch.zeros((num_steps, num_workers, len(act_spaces[i].nvec)), device=device) for i in agent_ids}
        self.logprobs = {i: torch.zeros((num_steps, num_workers), device=device) for i in agent_ids}
        self.rewards = {i: torch.zeros((num_steps, num_workers), device=device) for i in agent_ids}
        self.dones = {i: torch.zeros((num_steps, num_workers), device=device) for i in agent_ids}
        self.values = {i: torch.zeros((num_steps, num_workers), device=device) for i in agent_ids}
        self.step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values):
        for agent_id in self.agent_ids:
            if agent_id in obs:
                self.obs[agent_id][self.step] = torch.tensor(obs[agent_id], dtype=torch.float32, device=self.device)
                self.actions[agent_id][self.step] = actions[agent_id]
                self.logprobs[agent_id][self.step] = logprobs[agent_id]
                self.rewards[agent_id][self.step] = torch.tensor(rewards.get(agent_id, np.zeros(self.num_workers)), dtype=torch.float32, device=self.device)
                self.dones[agent_id][self.step] = torch.tensor(dones, dtype=torch.float32, device=self.device)
                self.values[agent_id][self.step] = values[agent_id].squeeze()
        self.step = (self.step + 1) % self.num_steps

    def compute_advantages_and_returns(self, next_values, next_dones, gamma, gae_lambda):
        self.advantages, self.returns = {}, {}
        for agent_id in self.agent_ids:
            last_gae_lam = 0
            advantages = torch.zeros_like(self.rewards[agent_id], device=self.device)
            for t in reversed(range(self.num_steps)):
                next_non_terminal = 1.0 - (next_dones if t == self.num_steps - 1 else self.dones[agent_id][t + 1])
                next_value = next_values.get(agent_id, torch.zeros(self.num_workers, device=self.device)) if t == self.num_steps - 1 else self.values[agent_id][t + 1]
                delta = self.rewards[agent_id][t] + gamma * next_value * next_non_terminal - self.values[agent_id][t]
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[agent_id] = advantages
            self.returns[agent_id] = advantages + self.values[agent_id]

    def get_mini_batches(self, batch_size):
        num_samples = self.num_steps * self.num_workers
        indices = np.random.permutation(num_samples)
        flat_data = defaultdict(dict)
        for agent_id in self.agent_ids:
            flat_data[agent_id]['obs'] = self.obs[agent_id].view(num_samples, -1)
            flat_data[agent_id]['actions'] = self.actions[agent_id].view(num_samples, -1)
            flat_data[agent_id]['logprobs'] = self.logprobs[agent_id].view(-1)
            flat_data[agent_id]['advantages'] = self.advantages[agent_id].view(-1)
            flat_data[agent_id]['returns'] = self.returns[agent_id].view(-1)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            mini_batch = defaultdict(dict)
            for agent_id in self.agent_ids:
                for key in flat_data[agent_id]:
                    mini_batch[agent_id][key] = flat_data[agent_id][key][batch_indices]
            yield mini_batch

if __name__ == "__main__":
    args = Config(0).get_arguments
    total_timesteps = 10_000_000
    learning_rate = 1e-4
    num_steps = 2048
    batch_size = args.batch_size
    mini_batch_size = args.mini_batch_size
    num_update_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    num_workers = args.num_workers

    # Seeding
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")
    run_name = f"L{args.level}_{args.agent_mode}_{args.num_agents}v{args.num_opps}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"--- Training on {device} with {num_workers} workers ---")

    env_fns = [lambda: LowLevelEnv(args.env_config) for _ in range(num_workers)]
    vec_env = gym.vector.SyncVectorEnv(env_fns) # Backward-compatible: no context

    agent_ids = sorted(list(vec_env.single_observation_space.keys()))
    models, optimizers, policy_map = {}, {}, {}

    for agent_id in agent_ids:
        is_ac1_type = (agent_id - 1) % 2 == 0
        policy_id = 'ac1_policy' if is_ac1_type else 'ac2_policy'
        policy_map[agent_id] = policy_id
        if policy_id not in models:
            obs_space, act_space = vec_env.single_observation_space, vec_env.single_action_space
            own_id, other_id = (1, 2) if policy_id == 'ac1_policy' else (2, 1)
            ModelClass = FightActorCritic if args.agent_mode == 'fight' else EscapeActorCritic
            model_kwargs = {
                'obs_dim_own': obs_space[own_id].shape[0],
                'obs_dim_other': obs_space[other_id].shape[0],
                'act_parts_own': len(act_space[own_id].nvec),
                'act_parts_other': len(act_space[other_id].nvec),
                'actor_logits_dim': int(np.sum(act_space[own_id].nvec))
            }
            if ModelClass == FightActorCritic:
                model_kwargs['own_state_split_size'] = 12 if is_ac1_type else 10
            else:
                model_kwargs['own_state_split'] = (7, 18) if is_ac1_type else (6, 18)
            
            model = ModelClass(**model_kwargs).to(device)
            models[policy_id] = model
            optimizers[policy_id] = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    buffer = VectorizedRolloutBuffer(num_steps, num_workers, agent_ids, vec_env.single_observation_space, vec_env.single_action_space, device)
    num_updates = total_timesteps // (num_steps * num_workers)
    next_obs, _ = vec_env.reset(seed=seed)

    episodic_returns = deque(maxlen=30)
    worker_ep_returns = np.zeros(num_workers)

    print("--- STARTING TRAINING ---")
    pbar = tqdm.trange(1, num_updates + 1)
    for update in pbar:
        v_loss_epoch, pg_loss_epoch, entropy_epoch = defaultdict(float), defaultdict(float), defaultdict(float)
        
        for step in range(num_steps):
            with torch.no_grad():
                actions, logprobs, values = {}, {}, {}
                obs_tensors = {i: torch.tensor(next_obs[i], dtype=torch.float32).to(device) for i in agent_ids}
                for agent_id, obs_tensor in obs_tensors.items():
                    policy_id = policy_map[agent_id]
                    logits = models[policy_id].get_action_logits(obs_tensor)
                    split_sizes = list(vec_env.single_action_space[agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    action_parts = [dist.sample() for dist in dists]
                    actions[agent_id] = torch.stack(action_parts, dim=-1)
                    logprobs[agent_id] = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, action_parts)]), dim=0)
                
                for policy_id, model in models.items():
                    main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent_id = next(aid for aid in agent_ids if aid != main_agent_id)
                    critic_obs_dict = {'obs_1_own': obs_tensors[main_agent_id], 'act_1_own': actions[main_agent_id], 'obs_2': obs_tensors[other_agent_id], 'act_2': actions[other_agent_id]}
                    value = model.get_value(critic_obs_dict)
                    for aid, pid in policy_map.items():
                        if pid == policy_id:
                            values[aid] = value
            
            env_actions = {i: a.cpu().numpy() for i, a in actions.items()}
            next_obs, agg_rewards, terminateds, truncateds, infos = vec_env.step(env_actions)
            dones = np.logical_or(terminateds, truncateds)
            
            rewards_per_agent = {agent_id: np.zeros(num_workers) for agent_id in agent_ids}
            for i, done in enumerate(dones):
                info_source = infos["final_info"][i] if done else infos.get(i)
                if info_source and "agent_rewards" in info_source:
                    agent_rewards_data = info_source["agent_rewards"]
                    if isinstance(agent_rewards_data, dict):
                        for agent_id, reward in agent_rewards_data.items():
                            if agent_id in rewards_per_agent:
                                rewards_per_agent[agent_id][i] = reward
                
                worker_ep_returns[i] += agg_rewards[i]
                if done:
                    final_info = infos["final_info"][i]
                    if final_info and "episode" in final_info:
                        episodic_returns.append(final_info["episode"]["r"][0])
                    worker_ep_returns[i] = 0

            buffer.add(next_obs, actions, logprobs, rewards_per_agent, dones, values)

        with torch.no_grad():
            next_values = {}
            for policy_id, model in models.items():
                main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                other_agent_id = next(aid for aid in agent_ids if aid != main_agent_id)
                next_obs_tensors = {i: torch.tensor(next_obs[i], dtype=torch.float32).to(device) for i in agent_ids}
                critic_obs_dict = {'obs_1_own': next_obs_tensors[main_agent_id], 'act_1_own': torch.zeros_like(actions[main_agent_id]), 'obs_2': next_obs_tensors[other_agent_id], 'act_2': torch.zeros_like(actions[other_agent_id])}
                next_val = model.get_value(critic_obs_dict).squeeze()
                for aid, pid in policy_map.items():
                    if pid == policy_id:
                        next_values[aid] = next_val
        buffer.compute_advantages_and_returns(next_values, torch.tensor(dones, dtype=torch.float32, device=device), gamma, gae_lambda)

        for epoch in range(num_update_epochs):
            for mini_batch in buffer.get_mini_batches(mini_batch_size):
                for policy_id, model in models.items():
                    main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent_id = next(aid for aid in agent_ids if aid != main_agent_id)
                    
                    with torch.no_grad():
                        old_approx_kl = (-mini_batch[main_agent_id]['logprobs']).mean()

                    critic_obs_dict = {'obs_1_own': mini_batch[main_agent_id]['obs'], 'act_1_own': mini_batch[main_agent_id]['actions'], 'obs_2': mini_batch[other_agent_id]['obs'], 'act_2': mini_batch[other_agent_id]['actions']}
                    logits, new_values = model(mini_batch[main_agent_id]['obs'], critic_obs_dict)
                    
                    split_sizes = list(vec_env.single_action_space[main_agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    new_logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, mini_batch[main_agent_id]['actions'].T)]), dim=0)
                    entropy = torch.sum(torch.stack([d.entropy() for d in dists]), dim=0)
                    
                    logratio = new_logprobs - mini_batch[main_agent_id]['logprobs']
                    ratio = logratio.exp()
                    advantages = mini_batch[main_agent_id]['advantages']
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    pg_loss = torch.max(-advantages * ratio, -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                    v_loss = 0.5 * ((new_values.squeeze() - mini_batch[main_agent_id]['returns']) ** 2).mean()
                    loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
                    
                    optimizers[policy_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizers[policy_id].step()

                    v_loss_epoch[policy_id] += v_loss.item()
                    pg_loss_epoch[policy_id] += pg_loss.item()
                    entropy_epoch[policy_id] += entropy.mean().item()
        
        num_minibatches = (num_steps * num_workers) // mini_batch_size
        for policy_id in models.keys():
            writer.add_scalar(f"losses/{policy_id}_value_loss", v_loss_epoch[policy_id] / num_minibatches, update)
            writer.add_scalar(f"losses/{policy_id}_policy_loss", pg_loss_epoch[policy_id] / num_minibatches, update)
            writer.add_scalar(f"losses/{policy_id}_entropy", entropy_epoch[policy_id] / num_minibatches, update)
            
            with torch.no_grad():
                approx_kl = (old_approx_kl + new_logprobs.mean()).item()
            writer.add_scalar(f"charts/{policy_id}_approx_kl", approx_kl, update)

        avg_return = np.mean(episodic_returns) if len(episodic_returns) > 0 else 0.0
        pbar.set_description(f"Update {update}, Avg Return: {avg_return:.2f}")
        writer.add_scalar("charts/avg_episodic_return", avg_return, update)

        if update > 0 and update % 50 == 0:
            policy_dir = 'policies'
            os.makedirs(policy_dir, exist_ok=True)
            for policy_id, model in models.items():
                ac_type = 1 if policy_id == 'ac1_policy' else 2
                save_path = os.path.join(policy_dir, f'L{args.level}_AC{ac_type}_{args.agent_mode}.pth')
                torch.save(model.state_dict(), save_path)
            print(f"\nSaved models at update {update}.")

    vec_env.close()
    writer.close()
    print("--- TRAINING COMPLETE ---")
