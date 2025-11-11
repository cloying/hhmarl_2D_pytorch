# FILE: train_hier.py (Corrected with Recurrent PPO)

# --- Dependencies ---
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm
from collections import defaultdict

# --- Local Imports ---
from config import Config
from envs.env_hier import HighLevelEnv
from models.ac_models_hier import CommanderActorCritic, OBS_DIM


# --- Recurrent PPO Rollout Buffer for Sequence-Based Training ---
class RecurrentRolloutBuffer:
    def __init__(self, num_steps, num_agents, obs_dim, hidden_size, device):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device
        self.obs = torch.zeros((num_steps, num_agents, obs_dim))
        self.actions = torch.zeros((num_steps, num_agents))
        self.logprobs = torch.zeros((num_steps, num_agents))
        self.rewards = torch.zeros((num_steps, num_agents))
        self.dones = torch.zeros((num_steps, num_agents))
        self.values = torch.zeros((num_steps, num_agents))
        self.actor_hiddens = torch.zeros((num_steps, num_agents, hidden_size))
        self.step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values, actor_hiddens):
        for i, agent_id in enumerate(sorted(obs.keys())):
            self.obs[self.step, i] = torch.from_numpy(obs[agent_id])
            self.actions[self.step, i] = actions[agent_id]
            self.logprobs[self.step, i] = logprobs[agent_id]
            self.rewards[self.step, i] = rewards.get(agent_id, 0.0)
            self.dones[self.step, i] = float(dones.get("__all__", False))
            self.values[self.step, i] = values[i]
            self.actor_hiddens[self.step, i] = actor_hiddens[i]
        self.step = (self.step + 1) % self.num_steps

    def compute_advantages_and_returns(self, next_value, next_done, gamma, gae_lambda):
        self.advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        self.returns = (self.advantages + self.values).to(self.device)
        self.advantages = self.advantages.to(self.device)

    def get_sequences(self, sequence_length):
        num_samples = self.num_steps * self.num_agents

        # Flatten agents and steps into one batch dimension
        flat_obs = self.obs.view(num_samples, -1)
        flat_actions = self.actions.view(num_samples)
        flat_logprobs = self.logprobs.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)
        flat_dones = self.dones.view(-1)
        # Get hidden states from the start of each trajectory
        flat_actor_hiddens = self.actor_hiddens.view(num_samples, -1)

        # --- FIX: Create sequences for RNN training ---
        for start in range(0, num_samples, sequence_length):
            end = start + sequence_length
            if end > num_samples: continue  # Drop the last partial sequence

            yield (
                flat_obs[start:end], flat_actions[start:end], flat_logprobs[start:end],
                flat_advantages[start:end], flat_returns[start:end],
                flat_dones[start:end], flat_actor_hiddens[start]  # Initial hidden state for the sequence
            )


# --- Main Training Script ---
if __name__ == "__main__":
    args = Config(1).get_arguments

    # Hyperparameters
    total_timesteps = 25_000_000;
    learning_rate = 1e-4;
    num_steps = 1024
    batch_size = 1000;
    mini_batch_size = 256;
    sequence_length = 64
    num_update_epochs = 10;
    gamma = 0.99;
    gae_lambda = 0.95
    clip_coef = 0.2;
    ent_coef = 0.01;
    vf_coef = 0.5;
    max_grad_norm = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    run_name = f"Commander_{args.num_agents}v{args.num_opps}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    env = HighLevelEnv(args.env_config)

    # Model and Optimizer (Parameter Sharing)
    action_dim = env.action_space.n
    model = CommanderActorCritic(obs_dim=OBS_DIM, action_dim=action_dim, num_agents=args.num_agents).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    buffer = RecurrentRolloutBuffer(num_steps, args.num_agents, OBS_DIM, model.hidden_size, device)
    num_updates = total_timesteps // (num_steps * args.num_agents)

    next_obs, _ = env.reset(seed=42)
    next_done = torch.zeros(1)
    # Hidden states shape: (num_layers, num_agents, hidden_size)
    next_actor_hidden = torch.zeros(1, args.num_agents, model.hidden_size, device=device)

    pbar = tqdm.trange(1, num_updates + 1)
    for update in pbar:
        # --- A. Rollout Phase ---
        for step in range(num_steps):
            obs_array = np.array([next_obs[i] for i in sorted(next_obs.keys())])
            obs_tensors = torch.from_numpy(obs_array).float().to(device)

            actions, logprobs, values = {}, {}, {}

            with torch.no_grad():
                # Get actions and values for all agents in one forward pass
                logits, value, new_actor_hidden = model(obs_tensors, next_actor_hidden)

                dist = torch.distributions.Categorical(logits=logits)
                action_samples = dist.sample()
                logprob_samples = dist.log_prob(action_samples)

                for i, agent_id in enumerate(sorted(next_obs.keys())):
                    actions[agent_id] = action_samples[i]
                    logprobs[agent_id] = logprob_samples[i]

            buffer.add(next_obs, actions, logprobs, {}, next_done, value, next_actor_hidden.squeeze(0))

            # Update hidden state for next step
            next_actor_hidden = new_actor_hidden

            env_actions = {i: a.cpu().item() for i, a in actions.items()}
            next_obs, rewards, terminateds, truncateds, info = env.step(env_actions)

            # --- FIX: Correctly handle rewards and dones ---
            buffer.rewards[buffer.step - 1] = torch.tensor([rewards.get(i, 0.0) for i in sorted(next_obs.keys())])
            next_done = torch.tensor(float(terminateds.get("__all__", False) or truncateds.get("__all__", False)))

            # Reset hidden states if episode ended
            if next_done:
                next_actor_hidden.zero_()

        # --- B. Advantage Calculation (GAE) ---
        with torch.no_grad():
            obs_array = np.array([next_obs[i] for i in sorted(next_obs.keys())])
            obs_tensors = torch.from_numpy(obs_array).float().to(device)
            _, next_value, _ = model(obs_tensors, next_actor_hidden)

        buffer.compute_advantages_and_returns(next_value, next_done, gamma, gae_lambda)

        # --- C. Learning Phase with Sequences ---
        for epoch in range(num_update_epochs):
            for obs, action, logprob, advantage, returns, dones, h_init in buffer.get_sequences(sequence_length):
                # Reshape initial hidden state for the sequence
                h_init = h_init.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size) -> (1, batch_size=1, hidden_size)

                logits, value, _ = model(obs.unsqueeze(1), h_init)  # Add sequence dimension for GRU
                logits = logits.squeeze(1)
                value = value.squeeze()

                dist = torch.distributions.Categorical(logits=logits)
                new_logprob = dist.log_prob(action)
                entropy = dist.entropy()

                logratio = new_logprob - logprob
                ratio = logratio.exp()

                norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                pg_loss = torch.max(-norm_advantage * ratio,
                                    -norm_advantage * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                v_loss = 0.5 * ((value - returns) ** 2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # --- D. Logging and Checkpointing ---
        avg_reward = buffer.rewards.mean().item() * args.num_agents
        writer.add_scalar("charts/avg_episodic_reward_estimate", avg_reward, update)
        pbar.set_description(f"Update {update}, Avg Reward: {avg_reward:.2f}")

        if update % 50 == 0:
            save_dir = "results"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"Commander_{args.num_agents}_vs_{args.num_opps}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"\nModel saved to {save_path}")

    env.close()
    writer.close()
    print("--- HIERARCHICAL TRAINING COMPLETE ---")