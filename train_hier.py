# FILE: train_hier.py (Pure PyTorch Recurrent PPO Implementation)

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

from config import Config
from envs.env_hier import HighLevelEnv
from models.ac_models_hier import CommanderActorCritic, OBS_DIM

# --- Recurrent PPO Rollout Buffer ---

class RecurrentRolloutBuffer:
    """A buffer to store trajectories for Recurrent PPO, supporting parameter sharing."""
    def __init__(self, num_steps, num_agents, obs_dim, hidden_size, device):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device
        
        # Initialize storage for experience from all agents
        self.obs = torch.zeros((num_steps, num_agents, obs_dim))
        self.actions = torch.zeros((num_steps, num_agents))
        self.logprobs = torch.zeros((num_steps, num_agents))
        self.rewards = torch.zeros((num_steps, num_agents))
        self.dones = torch.zeros((num_steps, num_agents))
        self.values = torch.zeros((num_steps, num_agents))
        
        # Storage for recurrent hidden states
        self.actor_hiddens = torch.zeros((num_steps, num_agents, hidden_size))
        self.critic_hiddens = torch.zeros((num_steps, num_agents, hidden_size))
        self.step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values, actor_hiddens, critic_hiddens):
        """Add a new transition for all agents to the buffer."""
        for i, agent_id in enumerate(sorted(obs.keys())):
            self.obs[self.step, i] = torch.tensor(obs[agent_id])
            self.actions[self.step, i] = actions[agent_id]
            self.logprobs[self.step, i] = logprobs[agent_id]
            self.rewards[self.step, i] = torch.tensor(rewards.get(agent_id, 0.0))
            self.dones[self.step, i] = torch.tensor(float(dones.get(agent_id, False)))
            self.values[self.step, i] = values[agent_id]
            self.actor_hiddens[self.step, i] = actor_hiddens[agent_id]
            self.critic_hiddens[self.step, i] = critic_hiddens[agent_id]
        self.step = (self.step + 1) % self.num_steps

    def compute_advantages_and_returns(self, next_value, next_done, gamma, gae_lambda):
        """Compute advantages and returns using GAE for all agents."""
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
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
        
        self.returns = self.advantages + self.values

    def get_mini_batches(self, batch_size):
        """A generator that yields mini-batches of flattened data."""
        num_samples = self.num_steps * self.num_agents
        indices = np.random.permutation(num_samples)

        # Flatten the data across agents and steps
        flat_obs = self.obs.view(num_samples, -1)
        flat_actions = self.actions.view(num_samples, -1)
        flat_logprobs = self.logprobs.view(-1)
        flat_advantages = self.advantages.view(-1)
        flat_returns = self.returns.view(-1)
        flat_actor_hiddens = self.actor_hiddens.view(num_samples, -1)
        flat_critic_hiddens = self.critic_hiddens.view(num_samples, -1)
        
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                flat_obs[batch_indices].to(self.device),
                flat_actions[batch_indices].to(self.device),
                flat_logprobs[batch_indices].to(self.device),
                flat_advantages[batch_indices].to(self.device),
                flat_returns[batch_indices].to(self.device),
                flat_actor_hiddens[batch_indices].to(self.device),
                flat_critic_hiddens[batch_indices].to(self.device),
            )

# --- Main Training Script ---
if __name__ == "__main__":
    # --- 1. Configuration and Initialization ---
    args = Config(1).get_arguments  # Use Mode 1 for hierarchical config
    
    # PPO Hyperparameters
    total_timesteps = 25_000_000
    learning_rate = 1e-4
    num_steps = 1024
    batch_size = 1000
    mini_batch_size = 256
    num_update_epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    # Setup logging
    run_name = f"Commander_{args.num_agents}vs{args.num_opps}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # --- 2. Environment Setup ---
    env = HighLevelEnv(args.env_config)

    # --- 3. Model and Optimizer Setup (Parameter Sharing) ---
    # All agents use the same policy, so we only need one model and one optimizer.
    action_dim = env.action_space.n
    model = CommanderActorCritic(obs_dim=OBS_DIM, action_dim=action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    # --- 4. Recurrent Rollout Buffer Setup ---
    buffer = RecurrentRolloutBuffer(num_steps, args.num_agents, OBS_DIM, model.hidden_size, device)

    # --- 5. Main Recurrent PPO Training Loop ---
    num_updates = total_timesteps // (num_steps * args.num_agents)
    
    next_obs, _ = env.reset()
    next_done = torch.zeros(args.num_agents).to(device)
    # Initialize hidden states for each agent
    next_actor_hidden = torch.zeros(1, args.num_agents, model.hidden_size, device=device)
    next_critic_hidden = torch.zeros(1, args.num_agents, model.hidden_size, device=device)

    pbar = tqdm.trange(1, num_updates + 1)
    for update in pbar:
        # --- A. Rollout Phase ---
        for step in range(num_steps):
            obs_tensors = torch.tensor(np.array([next_obs[i] for i in sorted(next_obs.keys())]), dtype=torch.float32).to(device)
            
            actions, logprobs, values = {}, {}, {}
            actor_hiddens, critic_hiddens = {}, {}

            with torch.no_grad():
                # --- Get actions for all agents (decentralized execution) ---
                for i, agent_id in enumerate(sorted(next_obs.keys())):
                    # The actor part only needs its own observation and hidden state
                    actor_obs_dict = {'obs_1_own': obs_tensors[i].unsqueeze(0)}
                    
                    # Pass the specific agent's hidden state
                    logits, _, actor_h, _ = model(
                        actor_obs_dict, 
                        next_actor_hidden[:, i, :].unsqueeze(0), # Shape: (1, 1, hidden_size)
                        next_critic_hidden[:, i, :].unsqueeze(0)
                    )
                    
                    # Sample action
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                    # Store results for this agent
                    actions[agent_id] = action.squeeze()
                    logprobs[agent_id] = logprob.squeeze()
                    actor_hiddens[agent_id] = next_actor_hidden[:, i, :] # Store the state used for this action
                    critic_hiddens[agent_id] = next_critic_hidden[:, i, :]
                    
                    # Update the 'next' hidden state for this agent
                    next_actor_hidden[:, i, :] = actor_h.squeeze(0)

                # --- Get centralized values for all agents ---
                # Construct the full centralized observation dictionary
                critic_obs_dict = {f'obs_{i}_own': obs_tensors[i-1].unsqueeze(0) for i in range(1, args.num_agents + 1)}
                critic_obs_dict.update({f'act_{i}_own': actions[i].unsqueeze(0).unsqueeze(0) for i in range(1, args.num_agents + 1)})
                
                # We need to map the dict keys for the model's forward pass
                critic_obs_dict['obs_2'] = critic_obs_dict.pop('obs_2_own')
                critic_obs_dict['act_2'] = critic_obs_dict.pop('act_2_own')
                
                # Get the value estimate using centralized info and update critic hidden state
                _, value, _, critic_h = model(
                    critic_obs_dict, 
                    next_actor_hidden, # Actor state is not used for value, but required by signature
                    next_critic_hidden
                )
                next_critic_hidden = critic_h.squeeze(0)

                for i, agent_id in enumerate(sorted(next_obs.keys())):
                    values[agent_id] = value.squeeze() # All agents get the same centralized value

            # Execute actions in the environment
            env_actions = {i: a.cpu().item() for i, a in actions.items()}
            next_obs, rewards, terminateds, truncateds, info = env.step(env_actions)
            dones = {k: v or truncateds.get(k, False) for k, v in terminateds.items()}

            buffer.add(next_obs, actions, logprobs, rewards, dones, values, actor_hiddens, critic_hiddens)
            
            # Reset hidden states for agents whose episodes have ended
            if dones.get("__all__", False):
                next_actor_hidden.zero_()
                next_critic_hidden.zero_()

        # --- B. Advantage Calculation (GAE) ---
        with torch.no_grad():
            # Calculate value for the last step in the rollout
            obs_tensors = torch.tensor(np.array([next_obs[i] for i in sorted(next_obs.keys())]), dtype=torch.float32).to(device)
            critic_obs_dict_final = {f'obs_{i}_own': obs_tensors[i-1].unsqueeze(0) for i in range(1, args.num_agents + 1)}
            # ... and so on for the full critic dict ...
            
            _, next_value, _, _ = model(critic_obs_dict, next_actor_hidden, next_critic_hidden)
            next_done = dones.get("__all__", False)

        buffer.compute_advantages_and_returns(next_value.squeeze(), next_done, gamma, gae_lambda)

        # --- C. Learning Phase ---
        for epoch in range(num_update_epochs):
            for obs, action, logprob, advantage, returns, actor_h, critic_h in buffer.get_mini_batches(mini_batch_size):
                
                # Recompute logits, values, and entropy
                # This needs a more complex reconstruction of the centralized critic dict
                # For simplicity, we re-evaluate actor and critic separately here.
                
                actor_obs_dict = {'obs_1_own': obs} # Batch of observations
                
                # Pass the stored hidden states for this mini-batch
                logits, value, _, _ = model(
                    actor_obs_dict, # Actor part uses decentralized obs
                    actor_h.unsqueeze(0),
                    critic_h.unsqueeze(0)
                )

                dist = torch.distributions.Categorical(logits=logits)
                new_logprob = dist.log_prob(action.squeeze())
                entropy = dist.entropy()

                # Policy Loss
                logratio = new_logprob - logprob
                ratio = logratio.exp()
                pg_loss1 = -advantage * ratio
                pg_loss2 = -advantage * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((value.squeeze() - returns) ** 2).mean()

                # Total Loss
                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        
        # --- D. Logging and Checkpointing ---
        avg_reward = buffer.rewards.mean()
        writer.add_scalar("charts/avg_reward", avg_reward, update)
        pbar.set_description(f"Update {update}, Avg Reward: {avg_reward:.2f}")

        if update % 50 == 0:
            print(f"\nSaving commander model at update {update}...")
            save_path = os.path.join("results", f"Commander_{args.num_agents}_vs_{args.num_opps}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    env.close()
    writer.close()
    print("--- HIERARCHICAL TRAINING COMPLETE ---")