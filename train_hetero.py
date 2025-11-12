# FILE: train_hetero.py (Definitive Fix for Reward Collection and TypeError)

import os
import time
import json
import imageio
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


# --- Utility Functions (Unchanged) ---
def find_latest_run_dir(base_dir="runs"):
    if not os.path.exists(base_dir): return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs: return None
    return max(subdirs, key=os.path.getmtime)


def save_checkpoint(run_dir, update, models, optimizers, global_step):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update}.pth")
    state = {
        'update': update, 'global_step': global_step,
        'models_state_dict': {pid: m.state_dict() for pid, m in models.items()},
        'optimizers_state_dict': {pid: o.state_dict() for pid, o in optimizers.items()},
    }
    torch.save(state, checkpoint_path)
    print(f"\nCheckpoint saved to {checkpoint_path}")


def load_checkpoint(run_dir, models, optimizers):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir): return 0, 0
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints: return 0, 0
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    for pid, model in models.items():
        model.load_state_dict(checkpoint['models_state_dict'][pid])
    for pid, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint['optimizers_state_dict'][pid])
    print(f"Resumed from checkpoint: {checkpoint_path}")
    return checkpoint['update'], checkpoint['global_step']


def log_to_json(run_dir, update, data):
    log_path = os.path.join(run_dir, "progress.json")
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = []
    log_data.append({'update': update, **data})
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)


def evaluate_and_visualize(run_dir, update, models, policy_map, args, device):
    print(f"\nGenerating visualization for update {update}...")
    eval_env = LowLevelEnv(args.env_config)
    frames = []
    for _ in range(1):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            actions = {}
            with torch.no_grad():
                for agent_id, agent_obs in obs.items():
                    policy_id = policy_map[agent_id]
                    obs_tensor = torch.from_numpy(agent_obs).float().to(device).unsqueeze(0)
                    logits = models[policy_id].get_action_logits(obs_tensor)
                    split_sizes = list(eval_env.action_space[agent_id].nvec)
                    action_dists = [torch.distributions.Categorical(logits=l) for l in
                                    logits.split(split_sizes, dim=-1)]
                    action_parts = torch.stack([dist.sample() for dist in action_dists], dim=-1)
                    actions[agent_id] = action_parts.cpu().numpy().squeeze()
            obs, _, terminated, truncated, _ = eval_env.step(actions)
            done = terminated or truncated
            frame_path = os.path.join(run_dir, "frame.png")
            eval_env.plot(frame_path)
            frames.append(imageio.imread(frame_path))
    eval_env.close()
    gif_path = os.path.join(run_dir, f'behavior_update_{update}.gif')
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"Saved visualization to {gif_path}")
    if os.path.exists(frame_path): os.remove(frame_path)


if __name__ == "__main__":
    args = Config(0).get_arguments

    total_timesteps = 10_000_000;
    learning_rate = 3e-4;
    num_steps = 2048
    num_update_epochs = 10;
    num_mini_batches = 32;
    gamma = 0.99
    gae_lambda = 0.95;
    clip_coef = 0.2;
    ent_coef = 0.01;
    vf_coef = 0.5
    max_grad_norm = 0.5;
    num_workers = args.num_workers
    batch_size = int(num_steps * num_workers)
    mini_batch_size = int(batch_size // num_mini_batches)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")
    run_name = f"L{args.level}_{args.agent_mode}_{args.num_agents}v{args.num_opps}_{int(time.time())}"
    run_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(run_dir)
    print(f"--- Training on {device} | Run: {run_name} ---")

    env_fns = [lambda: LowLevelEnv(args.env_config) for _ in range(num_workers)]
    vec_env = gym.vector.SyncVectorEnv(env_fns)

    agent_ids = sorted(list(vec_env.single_observation_space.keys()))
    models, optimizers, policy_map = {}, {}, {}
    for agent_id in agent_ids:
        is_ac1_type = (agent_id - 1) % 2 == 0
        policy_id = 'ac1_policy' if is_ac1_type else 'ac2_policy'
        policy_map[agent_id] = policy_id
        if policy_id not in models:
            obs_space = vec_env.single_observation_space;
            act_space = vec_env.single_action_space
            own_id, other_id = (1, 2) if is_ac1_type else (2, 1)
            ModelClass = FightActorCritic if args.agent_mode == 'fight' else EscapeActorCritic
            model_kwargs = {
                'obs_dim_own': obs_space[own_id].shape[0], 'obs_dim_other': obs_space[other_id].shape[0],
                'act_parts_own': len(act_space[own_id].nvec), 'act_parts_other': len(act_space[other_id].nvec),
                'actor_logits_dim': int(np.sum(act_space[own_id].nvec))
            }
            if ModelClass == FightActorCritic:
                model_kwargs['own_state_split_size'] = 12 if is_ac1_type else 10
            else:
                model_kwargs['own_state_split'] = (7, 18) if is_ac1_type else (6, 18)
            model = ModelClass(**model_kwargs).to(device)
            models[policy_id] = model
            optimizers[policy_id] = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    start_update, global_step = 0, 0
    if args.restore:
        latest_run = find_latest_run_dir()
        if latest_run:
            print(f"Attempting to resume from latest run: {latest_run}")
            start_update, global_step = load_checkpoint(latest_run, models, optimizers)
            run_dir = latest_run
            writer = SummaryWriter(run_dir, purge_step=start_update)


    class RolloutBuffer:
        def __init__(self):
            self.obs = {
                i: np.zeros((num_steps, num_workers, *vec_env.single_observation_space[i].shape), dtype=np.float32) for
                i in agent_ids}
            self.actions = {
                i: np.zeros((num_steps, num_workers, len(vec_env.single_action_space[i].nvec)), dtype=np.int64) for i in
                agent_ids}
            self.logprobs = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
            self.rewards = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
            self.dones = np.zeros((num_steps, num_workers))
            self.values = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
            self.step = 0

        def add(self, obs, actions, logprobs, rewards, dones, values):
            for agent_id in agent_ids:
                if agent_id in obs:
                    self.obs[agent_id][self.step] = obs[agent_id]
                    self.actions[agent_id][self.step] = actions[agent_id]
                    self.logprobs[agent_id][self.step] = logprobs[agent_id]
                    self.rewards[agent_id][self.step] = rewards[agent_id]
                    self.values[agent_id][self.step] = values[agent_id]
            self.dones[self.step] = dones
            self.step = (self.step + 1) % num_steps


    buffer = RolloutBuffer()
    num_updates = total_timesteps // batch_size
    next_obs, _ = vec_env.reset()
    next_done = np.zeros(num_workers)
    episodic_returns = deque(maxlen=50)

    pbar = tqdm.trange(start_update + 1, num_updates + 1)
    for update in pbar:
        for step in range(num_steps):
            global_step += num_workers
            with torch.no_grad():
                actions, logprobs, values = {}, {}, {}
                obs_tensors = {i: torch.from_numpy(next_obs[i]).to(device) for i in agent_ids}
                for agent_id, obs_tensor in obs_tensors.items():
                    policy_id = policy_map[agent_id]
                    logits = models[policy_id].get_action_logits(obs_tensor)
                    split_sizes = list(vec_env.single_action_space[agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    action_parts = [dist.sample() for dist in dists]
                    actions[agent_id] = torch.stack(action_parts, dim=-1).cpu().numpy()
                    logprobs[agent_id] = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, action_parts)]),
                                                   dim=0).cpu().numpy()

                for policy_id, model in models.items():
                    main_agent = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent = next(aid for aid in agent_ids if aid != main_agent)
                    critic_obs_dict = {
                        'obs_1_own': obs_tensors[main_agent],
                        'act_1_own': torch.from_numpy(actions[main_agent]).to(device, dtype=torch.float32),
                        'obs_2': obs_tensors[other_agent],
                        'act_2': torch.from_numpy(actions[other_agent]).to(device, dtype=torch.float32),
                    }
                    value = model.get_value(critic_obs_dict).cpu().numpy()
                    for aid, pid in policy_map.items():
                        if pid == policy_id: values[aid] = value

            new_next_obs, agg_rewards, terminateds, truncateds, infos = vec_env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            ##################################################################
            ###            DEFINITIVE FIX FOR REWARD COLLECTION            ###
            ##################################################################
            rewards_per_agent = {agent_id: np.zeros(num_workers) for agent_id in agent_ids}
            for i in range(num_workers):
                info_source = {}
                # Safely get info from either "final_info" or the top-level dict
                if dones[i]:
                    final_infos = infos.get("final_info", [])
                    if i < len(final_infos) and final_infos[i] is not None:
                        info_source = final_infos[i]
                else:
                    info_source = infos.get(i, {})

                if "agent_rewards" in info_source:
                    for agent_id, reward in info_source["agent_rewards"].items():
                        if agent_id in rewards_per_agent:
                            rewards_per_agent[agent_id][i] = reward
            ##################################################################

            buffer.add(next_obs, actions, logprobs, rewards_per_agent, next_done, values)
            next_obs, next_done = new_next_obs, dones

            if "final_info" in infos:
                for info in filter(None, infos.get("final_info", [])):
                    if "episode" in info:
                        episodic_returns.append(info["episode"]["r"][0])

        with torch.no_grad():
            next_values = {}
            next_obs_tensors = {i: torch.from_numpy(next_obs[i]).to(device) for i in agent_ids}
            for policy_id, model in models.items():
                main_agent = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                other_agent = next(aid for aid in agent_ids if aid != main_agent)
                dummy_actions = {i: torch.zeros(num_workers, len(vec_env.single_action_space[i].nvec), device=device,
                                                dtype=torch.float32) for i in agent_ids}
                critic_obs_dict = {
                    'obs_1_own': next_obs_tensors[main_agent], 'act_1_own': dummy_actions[main_agent],
                    'obs_2': next_obs_tensors[other_agent], 'act_2': dummy_actions[other_agent],
                }
                next_val = model.get_value(critic_obs_dict).cpu().numpy()
                for aid, pid in policy_map.items():
                    if pid == policy_id: next_values[aid] = next_val

        advantages, returns = {}, {}
        for agent_id in agent_ids:
            adv = np.zeros_like(buffer.rewards[agent_id])
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                next_non_terminal = 1.0 - (next_done if t == num_steps - 1 else buffer.dones[t + 1])
                next_val = next_values[agent_id] if t == num_steps - 1 else buffer.values[agent_id][t + 1]
                delta = buffer.rewards[agent_id][t] + gamma * next_val * next_non_terminal - buffer.values[agent_id][t]
                adv[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[agent_id] = adv
            returns[agent_id] = adv + buffer.values[agent_id]

        b_obs = {i: buffer.obs[i].reshape((-1, *vec_env.single_observation_space[i].shape)) for i in agent_ids}
        b_actions = {i: buffer.actions[i].reshape((-1, len(vec_env.single_action_space[i].nvec))) for i in agent_ids}
        b_logprobs = {i: buffer.logprobs[i].reshape(-1) for i in agent_ids}
        b_advantages = {i: advantages[i].reshape(-1) for i in agent_ids}
        b_returns = {i: returns[i].reshape(-1) for i in agent_ids}

        for epoch in range(num_update_epochs):
            inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size;
                mb_inds = inds[start:end]
                for policy_id, model in models.items():
                    main_agent = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent = next(aid for aid in agent_ids if aid != main_agent)
                    critic_obs_dict = {
                        'obs_1_own': torch.from_numpy(b_obs[main_agent][mb_inds]).to(device),
                        'act_1_own': torch.from_numpy(b_actions[main_agent][mb_inds]).to(device, dtype=torch.float32),
                        'obs_2': torch.from_numpy(b_obs[other_agent][mb_inds]).to(device),
                        'act_2': torch.from_numpy(b_actions[other_agent][mb_inds]).to(device, dtype=torch.float32)
                    }
                    logits, new_values = model(critic_obs_dict['obs_1_own'], critic_obs_dict)
                    split_sizes = list(vec_env.single_action_space[main_agent].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    new_logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, torch.from_numpy(
                        b_actions[main_agent][mb_inds]).to(device).T)]), dim=0)
                    entropy = torch.sum(torch.stack([d.entropy() for d in dists]), dim=0)
                    logratio = new_logprobs - torch.from_numpy(b_logprobs[main_agent][mb_inds]).to(device)
                    ratio = logratio.exp()
                    mb_adv = torch.from_numpy(b_advantages[main_agent][mb_inds]).to(device)
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
                    v_loss = 0.5 * (
                                (new_values - torch.from_numpy(b_returns[main_agent][mb_inds]).to(device)) ** 2).mean()
                    loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
                    optimizers[policy_id].zero_grad();
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm);
                    optimizers[policy_id].step()

        avg_return = np.mean(episodic_returns) if len(episodic_returns) > 0 else 0.0
        pbar.set_description(f"Update {update}, Avg Return: {avg_return:.2f}, Global Step: {global_step}")
        writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
        log_data = {'global_step': global_step, 'avg_return': float(avg_return), 'policy_loss': pg_loss.item(),
                    'value_loss': v_loss.item()}
        log_to_json(run_dir, update, log_data)

        if update > 0 and update % 30 == 0:
            save_checkpoint(run_dir, update, models, optimizers, global_step)
            evaluate_and_visualize(run_dir, update, models, policy_map, args, device)

    vec_env.close()
    writer.close()
    print("--- TRAINING COMPLETE ---")