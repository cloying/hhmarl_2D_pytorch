# FILE: train_hetero.py (Corrected with Entropy Scheduling)

import os
import time
from collections import deque
from pathlib import Path
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data
import random

# Local Imports
import gymnasium as gym
from config import Config
from envs.env_hetero import LowLevelEnv
from models.ac_models_hetero import GNN_Critic, RecurrentActor


# --- Helper Function for Learning Rate Schedule ---
def get_learning_rate(global_step, initial_lr, total_timesteps):
    # (This function is unchanged)
    decay_start_fraction = 0.75
    decay_start_step = total_timesteps * decay_start_fraction
    if global_step < decay_start_step:
        return initial_lr
    else:
        decay_duration = total_timesteps - decay_start_step
        steps_into_decay = global_step - decay_start_step
        decay_progress = max(0.0, steps_into_decay / decay_duration)
        return initial_lr * (1.0 - decay_progress)


### --- NEW: Helper Function for Entropy Schedule --- ###
def get_entropy_coef(global_step, initial_ent, final_ent, total_timesteps):
    """Linearly decays the entropy coefficient over the entire training run."""
    progress = np.clip(global_step / total_timesteps, 0.0, 1.0)
    return initial_ent + progress * (final_ent - initial_ent)


# --- Other Helper Functions (evaluate_and_render_episode, etc. are unchanged) ---
def save_checkpoint(actors, critics, actor_optimizers, critic_optimizers, update, base_path):
    checkpoint_dir = Path(base_path) / "checkpoints" / f"update_{update}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for policy_id in actors.keys():
        torch.save(actors[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_actor.pth")
        torch.save(critics[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_critic.pth")
        torch.save(actor_optimizers[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_actor_optim.pth")
        torch.save(critic_optimizers[policy_id].state_dict(), checkpoint_dir / f"{policy_id}_critic_optim.pth")
    print(f"\nSaved checkpoint at update {update} to {checkpoint_dir}")


def evaluate_and_render_episode(args, policy_map, actors, device, writer, global_step):
    gif_dir = Path(writer.log_dir) / "gifs";
    gif_dir.mkdir(parents=True, exist_ok=True)
    eval_env = LowLevelEnv(args.env_config)
    obs, _ = eval_env.reset()
    hidden_states = {pid: torch.zeros(1, 1, model.hidden_size, device=device) for pid, model in actors.items()}
    done, frames = False, []
    while not done:
        actions = {}
        with torch.no_grad():
            for agent_id, agent_obs in obs.items():
                if agent_id not in policy_map: continue
                policy_id = policy_map[agent_id]
                obs_tensor = torch.from_numpy(agent_obs).float().unsqueeze(0).to(device)
                logits, new_hidden = actors[policy_id](obs_tensor, hidden_states[policy_id])
                hidden_states[policy_id] = new_hidden
                action_parts = [torch.argmax(part, dim=-1) for part in
                                logits.split(list(eval_env.action_space[agent_id].nvec), dim=-1)]
                actions[agent_id] = torch.stack(action_parts, dim=-1).cpu().numpy()[0]
        obs, _, terminated, truncated, _ = eval_env.step(actions)
        done = terminated or truncated
        frame_path = Path(writer.log_dir) / "_temp_frame.png";
        eval_env.plot(frame_path)
        frames.append(imageio.imread(frame_path));
        os.remove(frame_path)
        if len(frames) > args.horizon + 10: break
    gif_path = gif_dir / f"update_{global_step // (args.num_workers * 2048)}_{int(time.time())}.gif"
    imageio.mimsave(gif_path, frames, duration=120)
    eval_env.close()


def find_latest_checkpoint(base_path):
    checkpoint_dir = Path(base_path) / "checkpoints"
    if not checkpoint_dir.exists(): return None, 0
    updates = [int(p.name.split('_')[-1]) for p in checkpoint_dir.glob("update_*") if p.is_dir()]
    if not updates: return None, 0
    latest_update = max(updates)
    return checkpoint_dir / f"update_{latest_update}", latest_update


def load_checkpoint(actors, critics, actor_optimizers, critic_optimizers, path, device, restore_optimizers):
    print(f"Loading checkpoint from: {path}")
    for policy_id in actors.keys():
        actors[policy_id].load_state_dict(torch.load(path / f"{policy_id}_actor.pth", map_location=device))
        critics[policy_id].load_state_dict(torch.load(path / f"{policy_id}_critic.pth", map_location=device))
        if restore_optimizers:
            actor_optimizers[policy_id].load_state_dict(torch.load(path / f"{policy_id}_actor_optim.pth"))
            critic_optimizers[policy_id].load_state_dict(torch.load(path / f"{policy_id}_critic_optim.pth"))
    print("Checkpoint loaded.")


# --- Main Training Script ---
if __name__ == "__main__":
    args = Config(0).get_arguments
    # (Hyperparameter setup is unchanged)
    seed = args.seed;
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    initial_learning_rate = args.learning_rate
    num_steps = 2048;
    num_update_epochs = 10;
    num_mini_batches = 32
    gamma = 0.99;
    gae_lambda = 0.95;
    clip_coef = 0.2
    initial_ent_coef = args.ent_coef  # Use the initial value from config
    vf_coef = 0.5;
    max_grad_norm = 0.5
    num_workers = args.num_workers;
    batch_size = int(num_steps * num_workers)
    mini_batch_size = batch_size // num_mini_batches
    num_updates = args.total_timesteps // batch_size
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")
    writer = SummaryWriter(args.log_path)
    print(f"--- Training on {device} with {num_workers} workers | Logging to: {args.log_path} ---")

    # (Environment and model setup is unchanged)
    vec_env = gym.vector.SyncVectorEnv([lambda: LowLevelEnv(args.env_config) for _ in range(num_workers)])
    agent_ids = sorted(list(vec_env.single_observation_space.keys()))
    actors, critics, actor_optimizers, critic_optimizers, policy_map = {}, {}, {}, {}, {}
    for agent_id in agent_ids:
        is_ac1_type = (agent_id % 2) != 0;
        policy_id = 'ac1_policy' if is_ac1_type else 'ac2_policy'
        policy_map[agent_id] = policy_id
        if policy_id not in actors:
            obs_space = vec_env.single_observation_space;
            act_space = vec_env.single_action_space
            own_id = 1 if is_ac1_type else 2
            actor = RecurrentActor(obs_dim_own=obs_space[own_id].shape[0],
                                   actor_logits_dim=int(np.sum(act_space[own_id].nvec))).to(device)
            critic = GNN_Critic().to(device)
            actors[policy_id], critics[policy_id] = actor, critic
            actor_optimizers[policy_id] = optim.Adam(actor.parameters(), lr=initial_learning_rate, eps=1e-5)
            critic_optimizers[policy_id] = optim.Adam(critic.parameters(), lr=initial_learning_rate, eps=1e-5)

    # (Checkpoint loading is unchanged)
    start_update = 1
    if args.restore and args.restore_path:
        latest_checkpoint, latest_update = find_latest_checkpoint(args.restore_path)
        if latest_checkpoint:
            should_restore_optimizers = not args.reset_optimizers
            load_checkpoint(actors, critics, actor_optimizers, critic_optimizers, latest_checkpoint, device,
                            should_restore_optimizers)
            if should_restore_optimizers: start_update = latest_update + 1

    # (Buffer initialization is unchanged)
    obs_buffers = {i: np.zeros((num_steps, num_workers, *vec_env.single_observation_space[i].shape)) for i in agent_ids}
    action_buffers = {i: np.zeros((num_steps, num_workers, len(vec_env.single_action_space[i].nvec))) for i in
                      agent_ids}
    logprob_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids};
    reward_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
    value_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids};
    done_buffers = np.zeros((num_steps, num_workers))
    graph_data_buffer = [[None for _ in range(num_workers)] for _ in range(num_steps)]
    actor_hidden_buffers = {pid: np.zeros((num_steps, num_workers, actors[pid].hidden_size)) for pid in actors.keys()}

    # (Training loop start is unchanged)
    global_step = (start_update - 1) * batch_size
    next_obs, next_info = vec_env.reset(seed=seed)
    next_done = torch.zeros(num_workers, device=device)
    next_actor_hiddens = {pid: torch.zeros(1, num_workers, actors[pid].hidden_size, device=device) for pid in
                          actors.keys()}
    episodic_returns = deque(maxlen=30)
    print("--- STARTING TRAINING ---")
    pbar = tqdm.trange(start_update, num_updates + 1)
    for update in pbar:
        # --- MODIFICATION: Calculate new LR and Entropy Coef for this update ---
        new_lr = get_learning_rate(global_step, initial_learning_rate, args.total_timesteps)
        ent_coef = get_entropy_coef(global_step, initial_ent_coef, args.ent_coef_final,
                                    args.total_timesteps) if args.use_entropy_schedule else initial_ent_coef

        for optim_group in [actor_optimizers, critic_optimizers]:
            for optim in optim_group.values():
                for param_group in optim.param_groups: param_group['lr'] = new_lr

        vec_env.call("set_global_step", global_step)

        # (Rollout phase is unchanged)
        for step in range(num_steps):
            global_step += num_workers;
            obs_tensors = {i: torch.from_numpy(next_obs[i]).float().to(device) for i in agent_ids}
            for agent_id in agent_ids: obs_buffers[agent_id][step] = next_obs[agent_id]
            done_buffers[step] = next_done.cpu().numpy()
            for policy_id, h_state in next_actor_hiddens.items(): actor_hidden_buffers[policy_id][
                step] = h_state.squeeze(0).cpu().numpy()
            with torch.no_grad():
                actions, logprobs, values = {}, {}, {}
                for agent_id, obs_tensor in obs_tensors.items():
                    policy_id = policy_map[agent_id]
                    logits, new_hidden = actors[policy_id](obs_tensor, next_actor_hiddens[policy_id])
                    next_actor_hiddens[policy_id] = new_hidden
                    dists = [torch.distributions.Categorical(logits=l) for l in
                             logits.split(list(vec_env.single_action_space[agent_id].nvec), dim=-1)]
                    action_parts = [dist.sample() for dist in dists];
                    actions[agent_id] = torch.stack(action_parts, dim=-1)
                    logprobs[agent_id] = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, action_parts)]),
                                                   dim=0)
                graph_batch = Batch.from_data_list([Data(**g) for g in next_info["graph_data"] if g]).to(device)
                for policy_id in critics.keys():
                    value = critics[policy_id](graph_batch)
                    for aid, pid in policy_map.items():
                        if pid == policy_id: values[aid] = value
            for agent_id in agent_ids:
                action_buffers[agent_id][step] = actions[agent_id].cpu().numpy();
                logprob_buffers[agent_id][step] = logprobs[agent_id].cpu().numpy();
                value_buffers[agent_id][step] = values[agent_id].cpu().numpy().flatten()
            for w in range(num_workers): graph_data_buffer[step][w] = next_info["graph_data"][w]
            env_actions = {i: a.cpu().numpy() for i, a in actions.items()}
            next_obs, _, terminateds, truncateds, next_info = vec_env.step(env_actions)
            next_done = torch.from_numpy(np.logical_or(terminateds, truncateds)).to(device)
            all_sparse, all_dense = next_info.get("agent_rewards", [{} for _ in range(num_workers)]), next_info.get(
                "shaping_rewards", [{} for _ in range(num_workers)])
            for i in range(num_workers):
                sparse_rewards, dense_rewards = all_sparse[i] or {}, all_dense[i] or {}
                for agent_id in agent_ids: reward_buffers[agent_id][step, i] = sparse_rewards.get(agent_id,
                                                                                                  0.0) + dense_rewards.get(
                    agent_id, 0.0)
            if "final_info" in next_info:
                for info in next_info["final_info"]:
                    if info and "episode" in info: episodic_returns.append(info["episode"]["r"]); writer.add_scalar(
                        "charts/raw_episodic_return", info["episode"]["r"], global_step)
            for pid in next_actor_hiddens: next_actor_hiddens[pid] *= (1.0 - next_done.float()).view(1, -1, 1)

        # (Advantage calculation is unchanged)
        with torch.no_grad():
            next_values = {}
            next_graph_batch = Batch.from_data_list([Data(**g) for g in next_info["graph_data"] if g]).to(device)
            for policy_id in critics.keys():
                next_val = critics[policy_id](next_graph_batch).cpu().numpy().flatten()
                for aid, pid in policy_map.items():
                    if pid == policy_id: next_values[aid] = next_val
        advantages, returns = {i: np.zeros_like(reward_buffers[i]) for i in agent_ids}, {
            i: np.zeros_like(reward_buffers[i]) for i in agent_ids}
        for agent_id in agent_ids:
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - (done_buffers[t + 1] if t < num_steps - 1 else next_done.cpu().numpy())
                nextvals = value_buffers[agent_id][t + 1] if t < num_steps - 1 else next_values[agent_id]
                delta = reward_buffers[agent_id][t] + gamma * nextvals * nextnonterminal - value_buffers[agent_id][t]
                advantages[agent_id][t] = last_gae_lam = delta + gamma * gae_lambda * nextnonterminal * last_gae_lam
            returns[agent_id] = advantages[agent_id] + value_buffers[agent_id]

        # (Batch preparation is unchanged)
        b_obs = {i: obs_buffers[i].reshape((-1, *vec_env.single_observation_space[i].shape)) for i in agent_ids};
        b_actions = {i: action_buffers[i].reshape((-1, len(vec_env.single_action_space[i].nvec))) for i in agent_ids}
        b_logprobs = {i: logprob_buffers[i].reshape(-1) for i in agent_ids};
        b_advantages = {i: advantages[i].reshape(-1) for i in agent_ids};
        b_returns = {i: returns[i].reshape(-1) for i in agent_ids}
        b_graph_data = [g for step_graphs in graph_data_buffer for g in step_graphs if g]

        # --- MODIFICATION: The `ent_coef` used in the loss is now the scheduled value ---
        for epoch in range(num_update_epochs):
            inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size;
                mb_inds = inds[start:end]
                mb_graph_batch = Batch.from_data_list([Data(**b_graph_data[i]) for i in mb_inds]).to(device)
                for policy_id in critics.keys():
                    main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    mb_ret = torch.from_numpy(b_returns[main_agent_id][mb_inds]).to(device)
                    v_loss = 0.5 * ((critics[policy_id](mb_graph_batch).view(-1) - mb_ret) ** 2).mean()
                    critic_optimizers[policy_id].zero_grad();
                    v_loss.backward();
                    nn.utils.clip_grad_norm_(critics[policy_id].parameters(), max_grad_norm);
                    critic_optimizers[policy_id].step()
                for policy_id in actors.keys():
                    main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    mb_obs_own = torch.from_numpy(b_obs[main_agent_id][mb_inds]).float().to(device)
                    mb_act_own = torch.from_numpy(b_actions[main_agent_id][mb_inds]).to(device)
                    mb_adv = torch.from_numpy(b_advantages[main_agent_id][mb_inds]).to(device)
                    mb_logp = torch.from_numpy(b_logprobs[main_agent_id][mb_inds]).to(device)
                    h_start, h_worker = mb_inds // num_workers, mb_inds % num_workers
                    mb_hiddens = torch.from_numpy(actor_hidden_buffers[policy_id][h_start, h_worker]).float().unsqueeze(
                        0).to(device)
                    logits, _ = actors[policy_id](mb_obs_own, mb_hiddens)
                    dists = [torch.distributions.Categorical(logits=l) for l in
                             logits.split(list(vec_env.single_action_space[main_agent_id].nvec), dim=-1)]
                    new_logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, mb_act_own.T)]), dim=0)
                    entropy = torch.sum(torch.stack([d.entropy() for d in dists]), dim=0)
                    ratio = torch.exp(new_logprobs - mb_logp);
                    norm_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss1 = -norm_adv * ratio;
                    pg_loss2 = -norm_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    # The `ent_coef` variable here is the new scheduled value
                    actor_loss = torch.max(pg_loss1, pg_loss2).mean() - ent_coef * entropy.mean()
                    actor_optimizers[policy_id].zero_grad();
                    actor_loss.backward();
                    nn.utils.clip_grad_norm_(actors[policy_id].parameters(), max_grad_norm);
                    actor_optimizers[policy_id].step()

        # --- MODIFICATION: Log the new scheduled values to TensorBoard ---
        avg_return = np.mean(episodic_returns) if episodic_returns else 0.0
        pbar.set_description(
            f"Update {update}, Avg Return: {avg_return:.2f}, LR: {new_lr:.2e}, EntCoef: {ent_coef:.2e}")
        writer.add_scalar("charts/avg_episodic_return", avg_return, global_step);
        writer.add_scalar("charts/learning_rate", new_lr, global_step)
        writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)

        # (Checkpointing and final saving is unchanged)
        if update > 0 and update % args.checkpoint_interval == 0:
            save_checkpoint(actors, critics, actor_optimizers, critic_optimizers, update, args.log_path)
            for model in actors.values(): model.eval()
            evaluate_and_render_episode(args, policy_map, actors, device, writer, global_step)
            for model in actors.values(): model.train()
    final_policy_dir = os.path.join(args.log_path, 'policies')
    os.makedirs(final_policy_dir, exist_ok=True)
    for policy_id, model in actors.items():
        ac_type = 1 if policy_id == 'ac1_policy' else 2
        save_path = os.path.join(final_policy_dir, f'L3_Continuous_AC{ac_type}_fight.pth')
        torch.save(model.state_dict(), save_path)
    print(f"\nFinal policies saved to '{final_policy_dir}' directory.")
    vec_env.close();
    writer.close();
    print("--- TRAINING COMPLETE ---")