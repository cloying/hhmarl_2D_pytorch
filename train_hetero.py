# FILE: train_hetero.py (Cleaned, Final Version)

# --- Core Dependencies ---
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random

# --- Utility and Logging Dependencies ---
from torch.utils.tensorboard import SummaryWriter
import tqdm
from typing import cast
from gymnasium.spaces import Dict as DictSpace
from pathlib import Path

# --- Local Project Imports ---
from config import Config
from envs.env_hetero import LowLevelEnv
from models.ac_models_hetero import EscapeActorCritic, FightActorCritic


class VectorizedRolloutBuffer:
    """
    A rollout buffer optimized for data from multiple parallel environments.
    """

    def __init__(self, num_steps, num_workers, num_agents, obs_dims, act_dims, device):
        self.num_steps = num_steps
        self.num_workers = num_workers
        self.device = device
        self.obs = {i: torch.zeros((num_steps, num_workers, obs_dims[i])).to(device) for i in range(1, num_agents + 1)}
        self.actions = {i: torch.zeros((num_steps, num_workers, act_dims[i])).to(device) for i in
                        range(1, num_agents + 1)}
        self.logprobs = {i: torch.zeros((num_steps, num_workers)).to(device) for i in range(1, num_agents + 1)}
        self.rewards = {i: torch.zeros((num_steps, num_workers)).to(device) for i in range(1, num_agents + 1)}
        self.dones = {i: torch.zeros((num_steps, num_workers)).to(device) for i in range(1, num_agents + 1)}
        self.values = {i: torch.zeros((num_steps, num_workers)).to(device) for i in range(1, num_agents + 1)}
        self.step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values):
        """Adds a batch of transitions from all parallel workers to the buffer."""
        for agent_id in obs.keys():
            if agent_id in self.obs:
                self.obs[agent_id][self.step] = torch.tensor(obs[agent_id], dtype=torch.float32).to(self.device)
                self.actions[agent_id][self.step] = actions[agent_id]
                self.logprobs[agent_id][self.step] = logprobs[agent_id]
                agent_reward_slice = rewards[:, agent_id - 1] if rewards.ndim == 2 else rewards
                self.rewards[agent_id][self.step] = torch.tensor(agent_reward_slice, dtype=torch.float32).to(
                    self.device)
                self.dones[agent_id][self.step] = torch.tensor(dones, dtype=torch.float32).to(self.device)
                self.values[agent_id][self.step] = values[agent_id]

        self.step = (self.step + 1) % self.num_steps

    def compute_advantages_and_returns(self, next_values, next_dones, gamma, gae_lambda):
        """Computes GAE advantages and returns for all parallel trajectories."""
        self.advantages, self.returns = {}, {}
        for agent_id in self.values.keys():
            last_gae_lam = 0
            advantages = torch.zeros_like(self.rewards[agent_id]).to(self.device)
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_non_terminal = 1.0 - next_dones
                    next_value = next_values.get(agent_id, torch.zeros(self.num_workers).to(self.device))
                else:
                    next_non_terminal = 1.0 - self.dones[agent_id][t + 1]
                    next_value = self.values[agent_id][t + 1]

                delta = self.rewards[agent_id][t] + gamma * next_value * next_non_terminal - self.values[agent_id][t]
                advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

            self.advantages[agent_id] = advantages
            self.returns[agent_id] = advantages + self.values[agent_id]

    def get_mini_batches(self, batch_size):
        """Yields mini-batches of flattened, shuffled data."""
        num_samples = self.num_steps * self.num_workers
        indices = np.random.permutation(num_samples)

        flat_obs = {agent_id: obs_tensor.view(num_samples, -1) for agent_id, obs_tensor in self.obs.items()}
        flat_actions = {agent_id: act_tensor.view(num_samples, -1) for agent_id, act_tensor in self.actions.items()}
        flat_logprobs = {agent_id: lp_tensor.view(-1) for agent_id, lp_tensor in self.logprobs.items()}
        flat_advantages = {agent_id: adv_tensor.view(-1) for agent_id, adv_tensor in self.advantages.items()}
        flat_returns = {agent_id: ret_tensor.view(-1) for agent_id, ret_tensor in self.returns.items()}

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            mini_batch = {}
            for agent_id in self.obs.keys():
                mini_batch[agent_id] = {
                    'obs': flat_obs[agent_id][batch_indices],
                    'actions': flat_actions[agent_id][batch_indices],
                    'logprobs': flat_logprobs[agent_id][batch_indices],
                    'advantages': flat_advantages[agent_id][batch_indices],
                    'returns': flat_returns[agent_id][batch_indices],
                }
            yield mini_batch


if __name__ == "__main__":
    args = Config(0).get_arguments

    # --- Hyperparameters ---
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

    # --- Seeding for Reproducibility ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")
    print(f"--- Using device: {device} on {num_workers} parallel workers ---")

    run_name = f"L{args.level}_{args.agent_mode}_{args.num_agents}-vs-{args.num_opps}_seed{seed}_{int(time.time())}"
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir)
    plots_dir = Path(log_dir) / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # --- Create the Vectorized Environment ---
    env_fns = [lambda: gym.wrappers.RecordEpisodeStatistics(LowLevelEnv(args.env_config)) for _ in range(num_workers)]
    env = gym.vector.SyncVectorEnv(env_fns)

    models, optimizers = {}, {}

    obs_space = cast(DictSpace, env.single_observation_space)
    act_space = cast(DictSpace, env.single_action_space)

    agent_configs = {
        'ac1_policy': {
            'model_class': FightActorCritic if args.agent_mode == 'fight' else EscapeActorCritic,
            'obs_dim_own': obs_space.spaces[1].shape[0], 'obs_dim_other': obs_space.spaces[2].shape[0],
            'act_parts_own': len(act_space.spaces[1].nvec), 'act_parts_other': len(act_space.spaces[2].nvec),
            'actor_logits_dim': sum(act_space.spaces[1].nvec),
            'state_split': 12 if args.agent_mode == 'fight' else (7, 18),
        },
        'ac2_policy': {
            'model_class': FightActorCritic if args.agent_mode == 'fight' else EscapeActorCritic,
            'obs_dim_own': obs_space.spaces[2].shape[0], 'obs_dim_other': obs_space.spaces[1].shape[0],
            'act_parts_own': len(act_space.spaces[2].nvec), 'act_parts_other': len(act_space.spaces[1].nvec),
            'actor_logits_dim': sum(act_space.spaces[2].nvec),
            'state_split': 10 if args.agent_mode == 'fight' else (6, 18),
        }
    }

    for policy_id, cfg in agent_configs.items():
        model_kwargs = {
            'obs_dim_own': cfg['obs_dim_own'], 'obs_dim_other': cfg['obs_dim_other'],
            'act_parts_own': cfg['act_parts_own'], 'act_parts_other': cfg['act_parts_other'],
            'actor_logits_dim': cfg['actor_logits_dim'],
        }
        if cfg['model_class'] == FightActorCritic:
            model_kwargs['own_state_split_size'] = cfg['state_split']
        else:
            model_kwargs['own_state_split'] = cfg['state_split']

        model = cfg['model_class'](**model_kwargs)
        models[policy_id] = model.to(device)
        optimizers[policy_id] = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    # --- Training Initialization ---
    policy_map = {1: 'ac1_policy', 2: 'ac2_policy', 3: 'ac1_policy', 4: 'ac2_policy'}
    obs_dims = {i: obs_space.spaces[i].shape[0] for i in range(1, args.num_agents + 1)}
    act_dims_parts = {i: len(act_space.spaces[i].nvec) for i in range(1, args.num_agents + 1)}

    max_action_parts = 0
    for i in range(1, args.num_agents + 1):
        if act_dims_parts[i] > max_action_parts:
            max_action_parts = act_dims_parts[i]
    print(f"--- Max action parts: {max_action_parts} ---")

    buffer = VectorizedRolloutBuffer(num_steps, num_workers, args.num_agents, obs_dims, act_dims_parts, device)

    num_updates = total_timesteps // (num_steps * num_workers)

    worker_seeds = [seed + i for i in range(num_workers)]
    next_obs, _ = env.reset(seed=worker_seeds)

    next_done = torch.zeros(num_workers).to(device)

    print("--- STARTING TRAINING ---")
    pbar = tqdm.trange(1, num_updates + 1)
    for update in pbar:
        # --- A. Rollout Phase ---
        for step in range(num_steps):
            with torch.no_grad():
                actions, logprobs, values = {}, {}, {}
                obs_tensors = {agent_id: torch.tensor(next_obs[agent_id], dtype=torch.float32).to(device) for agent_id
                               in range(1, args.num_agents + 1)}

                for agent_id, obs_tensor in obs_tensors.items():
                    policy_id = policy_map[agent_id]
                    logits = models[policy_id].get_action_logits(obs_tensor)
                    split_sizes = list(act_space.spaces[agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    action_parts = [dist.sample() for dist in dists]
                    logprob = torch.sum(torch.stack([dist.log_prob(act) for dist, act in zip(dists, action_parts)]),
                                        dim=0)
                    actions[agent_id] = torch.stack(action_parts, dim=-1).cpu()
                    logprobs[agent_id] = logprob

                if 1 in obs_tensors and 2 in obs_tensors:
                    critic_obs_dict_1 = {'obs_1_own': obs_tensors[1], 'act_1_own': actions[1].to(device),
                                         'obs_2': obs_tensors[2], 'act_2': actions[2].to(device)}
                    val1 = models['ac1_policy'].get_value(critic_obs_dict_1)
                    critic_obs_dict_2 = {'obs_1_own': obs_tensors[2], 'act_1_own': actions[2].to(device),
                                         'obs_2': obs_tensors[1], 'act_2': actions[1].to(device)}
                    val2 = models['ac2_policy'].get_value(critic_obs_dict_2)
                    values = {1: val1, 2: val2}

            padded_actions_list = []
            for i in range(1, args.num_agents + 1):
                action_tensor = actions[i]
                num_parts = action_tensor.shape[1]

                if num_parts < max_action_parts:
                    padding_needed = max_action_parts - num_parts
                    padding = torch.zeros((action_tensor.shape[0], padding_needed), dtype=action_tensor.dtype)
                    padded_action = torch.cat([action_tensor, padding], dim=1)
                    padded_actions_list.append(padded_action.numpy())
                else:
                    padded_actions_list.append(action_tensor.numpy())

            env_actions = {i: padded_actions_list[i - 1] for i in range(1, args.num_agents + 1)}

            next_obs, agg_rewards, terminateds, truncateds, infos = env.step(env_actions)
            dones = np.logical_or(terminateds, truncateds)

            agent_rewards = np.zeros((num_workers, args.num_agents))
            per_worker_infos = infos.get("_", [{} for _ in range(num_workers)])
            final_infos = infos.get("final_info", [None for _ in range(num_workers)])

            for i in range(num_workers):
                info_source = final_infos[i] if dones[i] and final_infos[i] is not None else per_worker_infos[i]
                if "agent_rewards" in info_source:
                    reward_dict = info_source["agent_rewards"]
                    if isinstance(reward_dict, dict):
                        for agent_id, reward_val in reward_dict.items():
                            if 0 <= agent_id - 1 < args.num_agents:
                                agent_rewards[i, agent_id - 1] = reward_val

            buffer.add(next_obs, actions, logprobs, agent_rewards, dones, values)

        # --- B. Advantage Calculation ---
        with torch.no_grad():
            next_values = {}
            if 1 in next_obs and 2 in next_obs:
                final_obs_data = infos.get("final_observation", [None] * num_workers)

                last_obs_1_list, last_obs_2_list = [], []
                for i in range(num_workers):
                    if dones[i] and final_obs_data[i] is not None:
                        last_obs_1_list.append(final_obs_data[i].get(1, next_obs[1][i]))
                        last_obs_2_list.append(final_obs_data[i].get(2, next_obs[2][i]))
                    else:
                        last_obs_1_list.append(next_obs[1][i])
                        last_obs_2_list.append(next_obs[2][i])

                last_obs_1 = torch.tensor(np.array(last_obs_1_list), dtype=torch.float32).to(device)
                last_obs_2 = torch.tensor(np.array(last_obs_2_list), dtype=torch.float32).to(device)

                critic_obs_final_1 = {'obs_1_own': last_obs_1,
                                      'act_1_own': torch.zeros(num_workers, act_dims_parts[1], device=device),
                                      'obs_2': last_obs_2,
                                      'act_2': torch.zeros(num_workers, act_dims_parts[2], device=device)}
                critic_obs_final_2 = {'obs_1_own': last_obs_2,
                                      'act_1_own': torch.zeros(num_workers, act_dims_parts[2], device=device),
                                      'obs_2': last_obs_1,
                                      'act_2': torch.zeros(num_workers, act_dims_parts[1], device=device)}
                next_values = {
                    1: models['ac1_policy'].get_value(critic_obs_final_1),
                    2: models['ac2_policy'].get_value(critic_obs_final_2)
                }

            next_dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)

        buffer.compute_advantages_and_returns(next_values, next_dones_tensor, gamma, gae_lambda)

        # --- C. Learning Phase ---
        for epoch in range(num_update_epochs):
            for mini_batch in buffer.get_mini_batches(mini_batch_size):
                for policy_id, model in models.items():
                    agent_id_map = {'ac1_policy': 1, 'ac2_policy': 2}
                    other_agent_id_map = {'ac1_policy': 2, 'ac2_policy': 1}
                    agent_id, other_agent_id = agent_id_map[policy_id], other_agent_id_map[policy_id]

                    if agent_id not in mini_batch or other_agent_id not in mini_batch:
                        continue

                    batch_actor_obs = mini_batch[agent_id]['obs']
                    batch_critic_obs = {
                        'obs_1_own': mini_batch[agent_id]['obs'], 'act_1_own': mini_batch[agent_id]['actions'],
                        'obs_2': mini_batch[other_agent_id]['obs'], 'act_2': mini_batch[other_agent_id]['actions']
                    }
                    logits, new_values = model(batch_actor_obs, batch_critic_obs)

                    split_sizes = list(act_space.spaces[agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    new_logprobs = torch.sum(torch.stack(
                        [dist.log_prob(act) for dist, act in zip(dists, mini_batch[agent_id]['actions'].T)]), dim=0)
                    entropy = torch.sum(torch.stack([dist.entropy() for dist in dists]), dim=0)

                    logratio = new_logprobs - mini_batch[agent_id]['logprobs']
                    ratio = logratio.exp()
                    advantages = mini_batch[agent_id]['advantages']
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((new_values - mini_batch[agent_id]['returns']) ** 2).mean()
                    loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

                    optimizers[policy_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizers[policy_id].step()

        # --- D. Logging, Checkpointing, and Visualization ---
        avg_episodic_return = 0

        # Check if episode-end information is available
        if "final_info" in infos:
            # Create the list of final info dictionaries, filtering out any Nones
            final_infos_list = [info for info in infos["final_info"] if info is not None]

            if final_infos_list:
                # Calculate the average episodic return
                episodic_returns = [info["episode"]["r"].item() for info in final_infos_list if "episode" in info]
                if episodic_returns:
                    avg_episodic_return = np.mean(episodic_returns)

        # Log the primary metric
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, update)

        # --- DEFINITIVE FIX: Force the writer to save the buffered data to the file ---
        writer.flush()

        # Update the progress bar description
        pbar.set_description(f"Update {update}, Avg Return: {avg_episodic_return:.2f}")

        # Checkpointing and plotting logic (remains the same)
        if update > 0 and update % 50 == 0:
            print(f"\nSaving models at update {update}...")
            policy_dir = 'policies'
            os.makedirs(policy_dir, exist_ok=True)
            for policy_id, model in models.items():
                ac_type = 1 if policy_id == 'ac1_policy' else 2
                policy_name = f'L{args.level}_AC{ac_type}_{args.agent_mode}.pth'
                save_path = os.path.join(policy_dir, policy_name)
                torch.save(model.state_dict(), save_path)
            print("Models saved.")

            print(f"Generating plot for update {update}...")
            plot_path = plots_dir / f"update_{update:05d}_return_{avg_episodic_return:.2f}.png"
            try:
                env.call_at(0, "plot", plot_path)
                print(f"Plot saved to {plot_path}")
            except Exception as e:
                print(f"Could not generate plot: {e}")

    # --- Cleanup ---
    env.close()
    writer.close()
    print("--- TRAINING COMPLETE ---")