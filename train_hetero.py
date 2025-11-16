# FILE: train_hetero.py (Final Version with Separated Actor/Critic Updates)

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
from collections import deque
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import tqdm
import imageio
from torch_geometric.data import Data, Batch

from config import Config
from envs.env_hetero import LowLevelEnv
from models.ac_models_hetero import Actor, GNN_Critic

# This is an optional but recommended fix for the graph break warnings.
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True


def linear_decay(current_step, total_steps, initial_value, final_value):
    if current_step >= total_steps: return final_value
    fraction = current_step / total_steps
    return initial_value + fraction * (final_value - initial_value)


def save_checkpoint(actors, critics, actor_optimizers, critic_optimizers, update, base_path):
    checkpoint_dir = Path(base_path) / "checkpoints" / f"update_{update}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for policy_id in actors.keys():
        torch.save(actors[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_actor.pth")
        torch.save(critics[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_critic.pth")
        torch.save(actor_optimizers[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_actor_optim.pth")
        torch.save(critic_optimizers[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_critic_optim.pth")
    print(f"\nSaved checkpoint at update {update} to {checkpoint_dir}")


def find_latest_checkpoint(base_path):
    checkpoint_dir = Path(base_path) / "checkpoints"
    if not checkpoint_dir.exists(): return None, 0
    checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("update_")]
    if not checkpoints: return None, 0
    latest_update = -1;
    latest_checkpoint_path = None
    for cp_path in checkpoints:
        try:
            update_num = int(cp_path.name.split('_')[-1])
            if update_num > latest_update:
                latest_update = update_num
                latest_checkpoint_path = cp_path
        except (ValueError, IndexError):
            continue
    return latest_checkpoint_path, latest_update


def load_checkpoint(actors, critics, actor_optimizers, critic_optimizers, path, device, restore_optimizers=True):
    print(f"Loading checkpoint from: {path}")
    for policy_id in actors.keys():
        actors[policy_id].load_state_dict(torch.load(path / f"{policy_id}_actor.pth", map_location=device))
        critics[policy_id].load_state_dict(torch.load(path / f"{policy_id}_critic.pth", map_location=device))
        if restore_optimizers:
            print("  - Restoring optimizer states.")
            actor_optimizers[policy_id].load_state_dict(
                torch.load(path / f"{policy_id}_actor_optim.pth", map_location=device))
            critic_optimizers[policy_id].load_state_dict(
                torch.load(path / f"{policy_id}_critic_optim.pth", map_location=device))
        else:
            print("  - Skipping optimizer state restoration for new curriculum level.")
    print("Checkpoint loaded successfully.")


def evaluate_and_render_episode(args, policy_map, actors, device, writer, global_step, num_steps):
    print(f"\n--- Rendering evaluation episode at step {global_step} ---")
    gif_dir = Path(writer.log_dir) / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    eval_env = LowLevelEnv(args.env_config)
    obs, info = eval_env.reset()
    done = False
    step_count = 0;
    total_reward = 0;
    frames = []
    while not done:
        actions = {}
        with torch.no_grad():
            for agent_id, agent_obs in obs.items():
                if agent_id not in policy_map: continue
                policy_id = policy_map[agent_id]
                obs_tensor = torch.from_numpy(agent_obs).unsqueeze(0).to(device)
                logits = actors[policy_id](obs_tensor)
                split_sizes = list(eval_env.action_space[agent_id].nvec)
                action_parts = [torch.argmax(part, dim=-1) for part in logits.split(split_sizes, dim=-1)]
                actions[agent_id] = torch.stack(action_parts, dim=-1).cpu().numpy()[0]
        obs, reward, terminated, truncated, info = eval_env.step(actions)
        done = terminated or truncated
        total_reward += reward
        frame_path = Path(writer.log_dir) / f"_temp_frame.png"
        eval_env.plot(frame_path)
        frames.append(imageio.imread(frame_path))
        os.remove(frame_path)
        step_count += 1
        if step_count > args.horizon + 5: break
    gif_path = gif_dir / f"update_{global_step // (args.num_workers * num_steps)}_{int(time.time())}.gif"
    imageio.mimsave(gif_path, frames, duration=100)
    print(f"--- Saved evaluation GIF to {gif_path} ---")
    eval_env.close()
    return total_reward


if __name__ == "__main__":
    args = Config(0).get_arguments

    total_timesteps = args.total_timesteps
    initial_learning_rate = args.learning_rate
    if args.restore:
        initial_learning_rate = 5e-6
        print(f"Restore mode enabled. Using smaller learning rate for fine-tuning: {initial_learning_rate}")

    num_steps = 2048;
    num_update_epochs = 10;
    num_mini_batches = 32;
    gamma = 0.99
    gae_lambda = 0.95;
    clip_coef = 0.2;
    ent_coef = args.ent_coef
    vf_coef = 0.5;
    max_grad_norm = 0.5
    num_workers = args.num_workers;
    batch_size = int(num_steps * num_workers)
    mini_batch_size = int(batch_size // num_mini_batches);
    num_updates = total_timesteps // batch_size
    seed = 42;
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")
    run_name = f"L{args.level}_{args.agent_mode}_{args.num_agents}v{args.num_opps}_GNN_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"--- Training on {device} with {num_workers} workers | Run: {run_name} ---")


    def make_env():
        return LowLevelEnv(args.env_config)


    env_fns = [make_env for _ in range(num_workers)]
    vec_env = gym.vector.SyncVectorEnv(env_fns)

    agent_ids = sorted(list(vec_env.single_observation_space.keys()))
    actors, critics, actor_optimizers, critic_optimizers, policy_map = {}, {}, {}, {}, {}

    for agent_id in agent_ids:
        is_ac1_type = (agent_id % 2) != 0
        policy_id = 'ac1_policy' if is_ac1_type else 'ac2_policy'
        policy_map[agent_id] = policy_id
        if policy_id not in actors:
            obs_space = vec_env.single_observation_space;
            act_space = vec_env.single_action_space
            own_id = 1 if is_ac1_type else 2
            actor = Actor(obs_dim_own=obs_space[own_id].shape[0],
                          actor_logits_dim=int(np.sum(act_space[own_id].nvec))).to(device)
            critic = GNN_Critic().to(device)
            if torch.__version__ >= "2.0.0":
                actor = torch.compile(actor);
                critic = torch.compile(critic)
            actors[policy_id] = actor;
            critics[policy_id] = critic
            actor_optimizers[policy_id] = optim.Adam(actor.parameters(), lr=initial_learning_rate, eps=1e-5)
            critic_optimizers[policy_id] = optim.Adam(critic.parameters(), lr=initial_learning_rate, eps=1e-5)

    start_update = 1
    if args.restore and args.restore_path:
        latest_checkpoint, latest_update = find_latest_checkpoint(args.restore_path)
        if latest_checkpoint:
            load_checkpoint(actors, critics, actor_optimizers, critic_optimizers, latest_checkpoint, device,
                            restore_optimizers=False)

    obs_buffers = {i: np.zeros((num_steps, num_workers, *vec_env.single_observation_space[i].shape), dtype=np.float32)
                   for i in agent_ids}
    action_buffers = {i: np.zeros((num_steps, num_workers, len(vec_env.single_action_space[i].nvec)), dtype=np.int64)
                      for i in agent_ids}
    logprob_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
    reward_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
    done_buffers = np.zeros((num_steps, num_workers))
    value_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
    graph_data_buffer = [[None for _ in range(num_workers)] for _ in range(num_steps)]

    global_step = 0
    next_obs, next_info = vec_env.reset(seed=seed)
    next_done = np.zeros(num_workers)
    episodic_returns = deque(maxlen=30)

    print("--- STARTING TRAINING ---")
    pbar = tqdm.trange(start_update, num_updates + 1)
    for update in pbar:
        new_lr = linear_decay(update, num_updates, initial_learning_rate, 1e-6)
        for policy_id in actors.keys():
            for param_group in actor_optimizers[policy_id].param_groups:
                param_group["lr"] = new_lr
            for param_group in critic_optimizers[policy_id].param_groups:
                param_group["lr"] = new_lr

        for step in range(num_steps):
            global_step += num_workers
            for agent_id in agent_ids: obs_buffers[agent_id][step] = next_obs[agent_id]
            done_buffers[step] = next_done
            current_shaping_scale = linear_decay(global_step, args.shaping_decay_timesteps, args.shaping_scale_initial,
                                                 args.shaping_scale_final)
            with torch.no_grad():
                actions, logprobs, values = {}, {}, {}
                obs_tensors = {i: torch.from_numpy(next_obs[i]).to(device) for i in agent_ids}
                for agent_id, obs_tensor in obs_tensors.items():
                    policy_id = policy_map[agent_id]
                    logits = actors[policy_id](obs_tensor)
                    split_sizes = list(vec_env.single_action_space[agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    action_parts = [dist.sample() for dist in dists]
                    actions[agent_id] = torch.stack(action_parts, dim=-1)
                    logprobs[agent_id] = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, action_parts)]),
                                                   dim=0)
                graph_data_list = [Data(**next_info["graph_data"][w]) for w in range(num_workers) if
                                   next_info["graph_data"][w] and next_info["graph_data"][w]['x'].nelement() > 0]
                graph_batch = Batch.from_data_list(graph_data_list).to(device) if graph_data_list else None
                for policy_id in actors.keys():
                    value = critics[policy_id](graph_batch) if graph_batch else torch.zeros(
                        len(graph_data_list) if graph_data_list else 0, device=device)
                    # Handle potential size mismatch if some envs returned empty graphs
                    if value.shape[0] != num_workers:
                        full_value = torch.zeros(num_workers, device=device)
                        valid_indices = [w for w, g in enumerate(next_info["graph_data"]) if
                                         g and g['x'].nelement() > 0]
                        full_value[valid_indices] = value
                        value = full_value
                    for aid, pid in policy_map.items():
                        if pid == policy_id:
                            values[aid] = value
            for agent_id in agent_ids:
                action_buffers[agent_id][step] = actions[agent_id].cpu().numpy()
                logprob_buffers[agent_id][step] = logprobs[agent_id].cpu().numpy()
                value_buffers[agent_id][step] = values[agent_id].cpu().numpy().flatten()
            for w in range(num_workers):
                graph_data_buffer[step][w] = next_info["graph_data"][w]
            env_actions = {i: a.cpu().numpy() for i, a in actions.items()}
            next_obs, _, terminateds, truncateds, next_info = vec_env.step(env_actions)
            next_done = np.logical_or(terminateds, truncateds)
            for i in range(num_workers):
                worker_rewards = next_info.get("agent_rewards", [{} for _ in range(num_workers)])[i]
                shaping_rewards = next_info.get("shaping_rewards", [{} for _ in range(num_workers)])[i]
                if worker_rewards or shaping_rewards:
                    for agent_id in agent_ids:
                        sparse_reward = worker_rewards.get(agent_id, 0.0)
                        dense_reward = shaping_rewards.get(agent_id, 0.0)
                        total_reward = sparse_reward + (dense_reward * current_shaping_scale)
                        reward_buffers[agent_id][step, i] = total_reward
            if "final_info" in next_info:
                for info in next_info["final_info"]:
                    if info and "episode" in info:
                        episodic_returns.append(info["episode"]["r"])
                        writer.add_scalar("charts/raw_episodic_return", info["episode"]["r"], global_step)

        with torch.no_grad():
            next_graph_data_list = [Data(**next_info["graph_data"][w]) for w in range(num_workers) if
                                    next_info["graph_data"][w] and next_info["graph_data"][w]['x'].nelement() > 0]
            next_graph_batch = Batch.from_data_list(next_graph_data_list).to(device) if next_graph_data_list else None
            next_values = {}
            for policy_id in actors.keys():
                next_val_tensor = critics[policy_id](next_graph_batch) if next_graph_batch else torch.zeros(
                    len(next_graph_data_list), device=device)
                if next_val_tensor.shape[0] != num_workers:
                    full_next_val = torch.zeros(num_workers, device=device)
                    valid_indices = [w for w, g in enumerate(next_info["graph_data"]) if g and g['x'].nelement() > 0]
                    full_next_val[valid_indices] = next_val_tensor
                    next_val_tensor = full_next_val
                next_val = next_val_tensor.cpu().numpy()
                for aid, pid in policy_map.items():
                    if pid == policy_id:
                        next_values[aid] = next_val

        advantages, returns = {}, {}
        for agent_id in agent_ids:
            agent_advantages = np.zeros_like(reward_buffers[agent_id])
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                next_non_terminal = 1.0 - (next_done if t == num_steps - 1 else done_buffers[t + 1])
                next_val_t = next_values[agent_id] if t == num_steps - 1 else value_buffers[agent_id][t + 1]
                delta = reward_buffers[agent_id][t] + gamma * next_val_t * next_non_terminal - value_buffers[agent_id][
                    t]
                agent_advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[agent_id] = agent_advantages
            returns[agent_id] = agent_advantages + value_buffers[agent_id]

        b_obs = {i: obs_buffers[i].reshape((-1, *vec_env.single_observation_space[i].shape)) for i in agent_ids}
        b_actions = {i: action_buffers[i].reshape((-1, len(vec_env.single_action_space[i].nvec))) for i in agent_ids}
        b_logprobs = {i: logprob_buffers[i].reshape(-1) for i in agent_ids}
        b_advantages = {i: advantages[i].reshape(-1) for i in agent_ids}
        b_returns = {i: returns[i].reshape(-1) for i in agent_ids}
        b_graph_data_list = [graph for step_graphs in graph_data_buffer for graph in step_graphs if
                             graph and graph['x'].nelement() > 0]

        if len(b_graph_data_list) != batch_size: continue
        inds = np.random.permutation(batch_size)

        for epoch in range(num_update_epochs):
            shuffled_graphs = [b_graph_data_list[i] for i in inds]
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = inds[start:end]
                mb_graph_data = [Data(**shuffled_graphs[i]) for i in range(start, end)]
                mb_graph_batch = Batch.from_data_list(mb_graph_data).to(device)

                for policy_id in actors.keys():
                    main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    mb_obs_own = torch.from_numpy(b_obs[main_agent_id][mb_inds]).to(device)
                    mb_act_own = torch.from_numpy(b_actions[main_agent_id][mb_inds]).to(device)
                    mb_adv = torch.from_numpy(b_advantages[main_agent_id][mb_inds]).to(device)
                    mb_ret = torch.from_numpy(b_returns[main_agent_id][mb_inds]).to(device)
                    mb_logp = torch.from_numpy(b_logprobs[main_agent_id][mb_inds]).to(device)

                    ### --- FIX: SEPARATE FORWARD AND BACKWARD PASSES --- ###

                    # --- Actor Loss Calculation ---
                    logits = actors[policy_id](mb_obs_own)
                    split_sizes = list(vec_env.single_action_space[main_agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    new_logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, mb_act_own.T)]), dim=0)
                    entropy = torch.sum(torch.stack([d.entropy() for d in dists]), dim=0)
                    logratio = new_logprobs - mb_logp
                    ratio = logratio.exp()
                    norm_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss1 = -norm_adv * ratio
                    pg_loss2 = -norm_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    actor_loss = pg_loss - ent_coef * entropy.mean()

                    # Actor Update
                    actor_optimizers[policy_id].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actors[policy_id].parameters(), max_grad_norm)
                    actor_optimizers[policy_id].step()

                    # --- Critic Loss Calculation (Separate Forward Pass) ---
                    new_values = critics[policy_id](mb_graph_batch)
                    v_loss = 0.5 * ((new_values.view(-1) - mb_ret) ** 2).mean()

                    # Critic Update
                    critic_optimizers[policy_id].zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critics[policy_id].parameters(), max_grad_norm)
                    critic_optimizers[policy_id].step()
                    ### --------------------------------------------------- ###

        avg_return = np.mean(episodic_returns) if len(episodic_returns) > 0 else 0.0
        pbar.set_description(f"Update {update}, Avg Return: {avg_return:.2f}, GS: {global_step}")
        writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
        writer.add_scalar("charts/learning_rate", new_lr, global_step)

        if update > 0 and update % args.checkpoint_interval == 0:
            save_checkpoint(actors, critics, actor_optimizers, critic_optimizers, update, args.log_path)

        if update > 0 and update % args.render_interval == 0:
            for model in actors.values(): model.eval()
            evaluate_and_render_episode(args, policy_map, actors, device, writer, global_step, num_steps)
            for model in actors.values(): model.train()

    policy_dir = os.path.join(args.log_path, 'policies')
    os.makedirs(policy_dir, exist_ok=True)
    for policy_id, model in actors.items():
        ac_type = 1 if policy_id == 'ac1_policy' else 2
        save_path = os.path.join(policy_dir, f'L{args.level}_AC{ac_type}_{args.agent_mode}.pth')
        torch.save(model.state_dict(), save_path)
    print(f"\nFinal policies saved to '{policy_dir}' directory.")

    vec_env.close()
    writer.close()
    print("--- TRAINING COMPLETE ---")