# FILE: train_hetero.py (Full Replacement - With Reward Schedule)

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

from config import Config
from envs.env_hetero import LowLevelEnv
from models.ac_models_hetero import EscapeActorCritic, FightActorCritic


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


def evaluate_and_render_episode(args, policy_map, models, device, writer, global_step):
    print("\nRendering an evaluation episode...")
    gif_dir = Path(writer.log_dir) / "gifs"
    frame_dir = gif_dir / f"update_{global_step // (args.num_workers * args.horizon)}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    eval_env = LowLevelEnv(args.env_config)
    obs, _ = eval_env.reset()
    done, step, total_reward, frames = False, 0, 0, []
    while not done:
        actions = {}
        with torch.no_grad():
            for agent_id, agent_obs in obs.items():
                if agent_id not in policy_map: continue
                policy_id = policy_map[agent_id]
                obs_tensor = torch.from_numpy(agent_obs).unsqueeze(0).to(device)
                logits = models[policy_id].get_action_logits(obs_tensor)
                split_sizes = list(eval_env.action_space[agent_id].nvec)
                dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                action_parts = [torch.argmax(dist.probs, dim=-1) for dist in dists]
                actions[agent_id] = torch.stack(action_parts, dim=-1).cpu().numpy()[0]
        obs, reward, terminated, truncated, info = eval_env.step(actions)
        done = terminated or truncated
        total_reward += reward
        frame_path = frame_dir / f"step_{step:03d}.png"
        eval_env.plot(frame_path)
        frames.append(imageio.imread(frame_path))
        step += 1
    gif_path = gif_dir / f"update_{global_step // (args.num_workers * args.horizon)}.gif"
    imageio.mimsave(gif_path, frames, duration=100)
    print(f"Saved evaluation GIF to {gif_path}")
    eval_env.close()
    return total_reward


if __name__ == "__main__":
    args = Config(0).get_arguments
    total_timesteps, initial_lr = args.total_timesteps, 1e-4
    num_steps, num_update_epochs, num_mini_batches = 2048, 10, 32
    gamma, gae_lambda, clip_coef = 0.99, 0.95, 0.1
    ent_coef, vf_coef, max_grad_norm = 0.025, 0.5, 0.5  # Increased ent_coef
    num_workers = args.num_workers
    render_every_n_updates, checkpoint_every_n_updates = 50, 25

    batch_size = int(num_steps * num_workers)
    mini_batch_size = int(batch_size // num_mini_batches)
    num_updates = total_timesteps // batch_size

    seed = 42;
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu > 0 else "cpu")
    run_name = f"L{args.level}_{args.agent_mode}_{args.num_agents}v{args.num_opps}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    print(f"--- Training on {device} with {num_workers} workers | Run: {run_name} ---")


    def make_env():
        return LowLevelEnv(args.env_config)


    vec_env = gym.vector.SyncVectorEnv([make_env for _ in range(num_workers)])

    agent_ids = sorted(list(vec_env.single_observation_space.keys()))
    models, optimizers, policy_map = {}, {}, {}
    for agent_id in agent_ids:
        is_ac1_type = (agent_id - 1) % 2 == 0;
        policy_id = 'ac1_policy' if is_ac1_type else 'ac2_policy'
        policy_map[agent_id] = policy_id
        if policy_id not in models:
            obs_space, act_space = vec_env.single_observation_space, vec_env.single_action_space
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
            optimizers[policy_id] = optim.Adam(model.parameters(), lr=initial_lr, eps=1e-5)

    start_update, global_step = 1, 0
    if args.restore and args.restore_path:
        latest_checkpoint_path = find_latest_checkpoint(args.restore_path)
        if latest_checkpoint_path:
            print(f"--- Loading latest checkpoint from: {latest_checkpoint_path} ---")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            for policy_id, model in models.items():
                if f'model_{policy_id}_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint[f'model_{policy_id}_state_dict'])
                    optimizers[policy_id].load_state_dict(checkpoint[f'optimizer_{policy_id}_state_dict'])
            print("Successfully loaded model weights for warm start.")
        else:
            print(
                f"WARNING: `restore` was True, but no valid checkpoint found in '{args.restore_path}'. Starting from scratch.")

    obs_buffers = {i: np.zeros((num_steps, num_workers, *vec_env.single_observation_space[i].shape), dtype=np.float32)
                   for i in agent_ids}
    action_buffers = {i: np.zeros((num_steps, num_workers, len(vec_env.single_action_space[i].nvec)), dtype=np.int64)
                      for i in agent_ids}
    logprob_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
    reward_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}
    done_buffers = np.zeros((num_steps, num_workers))
    value_buffers = {i: np.zeros((num_steps, num_workers)) for i in agent_ids}

    next_obs, _ = vec_env.reset(seed=seed)
    next_done = np.zeros(num_workers)
    episodic_returns = deque(maxlen=30)
    print("--- STARTING TRAINING ---")
    pbar = tqdm.trange(start_update, num_updates + 1)

    for update in pbar:
        # --- ANNEALING ---
        lr_frac = 1.0 - (update - 1.0) / num_updates
        current_lr = lr_frac * initial_lr
        for optimizer in optimizers.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

        ### --- NEW: Reward Schedule Calculation & Update --- ###
        current_shaping_scale = 1.0
        if args.use_reward_schedule:
            shaping_frac = 1.0 - (global_step / args.shaping_decay_timesteps)
            current_shaping_scale = max(args.shaping_scale_final, shaping_frac * args.shaping_scale_initial)

        # Set the attribute in all parallel environments
        vec_env.set_attr("current_shaping_scale", current_shaping_scale)

        writer.add_scalar("charts/learning_rate", current_lr, global_step)
        writer.add_scalar("charts/shaping_reward_scale", current_shaping_scale, global_step)
        ### --- END NEW SECTION --- ###

        # --- ROLLOUT ---
        for step in range(num_steps):
            global_step += num_workers;
            for agent_id in agent_ids: obs_buffers[agent_id][step] = next_obs[agent_id]
            done_buffers[step] = next_done
            with torch.no_grad():
                # ... (action selection logic is unchanged)
                actions, logprobs, values = {}, {}, {}
                obs_tensors = {i: torch.from_numpy(next_obs[i]).to(device) for i in agent_ids}
                for agent_id, obs_tensor in obs_tensors.items():
                    policy_id = policy_map[agent_id]
                    logits = models[policy_id].get_action_logits(obs_tensor)
                    split_sizes = list(vec_env.single_action_space[agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    action_parts = [dist.sample() for dist in dists]
                    actions[agent_id] = torch.stack(action_parts, dim=-1)
                    logprobs[agent_id] = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, action_parts)]),
                                                   dim=0)
                for policy_id, model in models.items():
                    main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent_id = next(aid for aid in agent_ids if aid != main_agent_id)
                    critic_obs_dict = {'obs_1_own': obs_tensors[main_agent_id],
                                       'act_1_own': actions[main_agent_id].float(),
                                       'obs_2': obs_tensors[other_agent_id], 'act_2': actions[other_agent_id].float()}
                    value = model.get_value(critic_obs_dict)
                    for aid, pid in policy_map.items():
                        if pid == policy_id: values[aid] = value
            for agent_id in agent_ids:
                action_buffers[agent_id][step] = actions[agent_id].cpu().numpy()
                logprob_buffers[agent_id][step] = logprobs[agent_id].cpu().numpy()
                value_buffers[agent_id][step] = values[agent_id].cpu().numpy().flatten()
            env_actions = {i: a.cpu().numpy() for i, a in actions.items()}
            next_obs, _, terminateds, truncateds, infos = vec_env.step(env_actions)
            next_done = np.logical_or(terminateds, truncateds)
            for i in range(num_workers):
                worker_rewards = infos.get("agent_rewards", [{} for _ in range(num_workers)])[i]
                if worker_rewards:
                    for agent_id in agent_ids: reward_buffers[agent_id][step, i] = worker_rewards.get(agent_id, 0.0)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info: episodic_returns.append(info["episode"]["r"])

        # --- GAE & LEARNING (unchanged) ---
        with torch.no_grad():  # ...
            next_values = {}
            next_obs_tensors = {i: torch.from_numpy(next_obs[i]).to(device) for i in agent_ids}
            for policy_id, model in models.items():
                main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                other_agent_id = next(aid for aid in agent_ids if aid != main_agent_id)
                dummy_actions = {i: torch.zeros(num_workers, len(vec_env.single_action_space[i].nvec), device=device,
                                                dtype=torch.float32) for i in agent_ids}
                critic_obs_dict = {'obs_1_own': next_obs_tensors[main_agent_id],
                                   'act_1_own': dummy_actions[main_agent_id], 'obs_2': next_obs_tensors[other_agent_id],
                                   'act_2': dummy_actions[other_agent_id]}
                next_val = model.get_value(critic_obs_dict).cpu().numpy().flatten()
                for aid, pid in policy_map.items():
                    if pid == policy_id: next_values[aid] = next_val
        advantages, returns = {}, {}
        for agent_id in agent_ids:  # ...
            agent_advantages = np.zeros_like(reward_buffers[agent_id])
            last_gae_lam = 0
            for t in reversed(range(num_steps)):
                next_non_terminal = 1.0 - (next_done if t == num_steps - 1 else done_buffers[t + 1])
                next_val = next_values[agent_id] if t == num_steps - 1 else value_buffers[agent_id][t + 1]
                delta = reward_buffers[agent_id][t] + gamma * next_val * next_non_terminal - value_buffers[agent_id][t]
                agent_advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[agent_id] = agent_advantages
            returns[agent_id] = agent_advantages + value_buffers[agent_id]
        b_obs = {i: obs_buffers[i].reshape((-1, *vec_env.single_observation_space[i].shape)) for i in agent_ids}
        b_actions = {i: action_buffers[i].reshape((-1, len(vec_env.single_action_space[i].nvec))) for i in agent_ids}
        b_logprobs = {i: logprob_buffers[i].reshape(-1) for i in agent_ids}
        b_advantages = {i: advantages[i].reshape(-1) for i in agent_ids}
        b_returns = {i: returns[i].reshape(-1) for i in agent_ids}
        for epoch in range(num_update_epochs):  # ...
            inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = inds[start:end]
                for policy_id, model in models.items():
                    main_agent_id = next(aid for aid, pid in policy_map.items() if pid == policy_id)
                    other_agent_id = next(aid for aid in agent_ids if aid != main_agent_id)
                    mb_obs_own = torch.from_numpy(b_obs[main_agent_id][mb_inds]).to(device);
                    mb_act_own = torch.from_numpy(b_actions[main_agent_id][mb_inds]).to(device)
                    mb_obs_other = torch.from_numpy(b_obs[other_agent_id][mb_inds]).to(device);
                    mb_act_other = torch.from_numpy(b_actions[other_agent_id][mb_inds]).to(device)
                    mb_adv = torch.from_numpy(b_advantages[main_agent_id][mb_inds]).to(device);
                    mb_ret = torch.from_numpy(b_returns[main_agent_id][mb_inds]).to(device)
                    mb_logp = torch.from_numpy(b_logprobs[main_agent_id][mb_inds]).to(device)
                    critic_obs_dict = {'obs_1_own': mb_obs_own, 'act_1_own': mb_act_own.float(), 'obs_2': mb_obs_other,
                                       'act_2': mb_act_other.float()}
                    logits, new_values = model(mb_obs_own, critic_obs_dict)
                    split_sizes = list(vec_env.single_action_space[main_agent_id].nvec)
                    dists = [torch.distributions.Categorical(logits=l) for l in logits.split(split_sizes, dim=-1)]
                    new_logprobs = torch.sum(torch.stack([d.log_prob(a) for d, a in zip(dists, mb_act_own.T)]), dim=0)
                    entropy = torch.sum(torch.stack([d.entropy() for d in dists]), dim=0)
                    logratio = new_logprobs - mb_logp;
                    ratio = logratio.exp()
                    norm_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss1 = -norm_adv * ratio;
                    pg_loss2 = -norm_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss = 0.5 * ((new_values.view(-1) - mb_ret) ** 2).mean()
                    loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
                    optimizers[policy_id].zero_grad();
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm);
                    optimizers[policy_id].step()

        # --- LOGGING & CHECKPOINTING (unchanged) ---
        avg_return = np.mean(episodic_returns) if len(episodic_returns) > 0 else 0.0
        pbar.set_description(f"Update {update}, Avg Return: {avg_return:.2f}, Global Step: {global_step}")
        writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
        if update > 0 and update % render_every_n_updates == 0:
            for model in models.values(): model.eval()
            rendered_reward = evaluate_and_render_episode(args, policy_map, models, device, writer, global_step)
            writer.add_scalar("charts/rendered_episode_reward", rendered_reward, global_step)
            for model in models.values(): model.train()
        if update > 0 and update % checkpoint_every_n_updates == 0:
            checkpoint_dir = os.path.join(args.log_path, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_data = {'update': update, 'global_step': global_step}
            for policy_id, model in models.items():
                checkpoint_data[f'model_{policy_id}_state_dict'] = model.state_dict()
                checkpoint_data[f'optimizer_{policy_id}_state_dict'] = optimizers[policy_id].state_dict()
            save_path = os.path.join(checkpoint_dir, f'checkpoint_update_{update}_step_{global_step}.pth')
            torch.save(checkpoint_data, save_path)

    final_policy_dir = 'policies';
    os.makedirs(final_policy_dir, exist_ok=True)
    for policy_id, model in models.items():
        ac_type = 1 if policy_id == 'ac1_policy' else 2
        save_path = os.path.join(final_policy_dir, f'L{args.level}_AC{ac_type}_{args.agent_mode}.pth')
        torch.save(model.state_dict(), save_path)
    print(f"\nFinal models for curriculum saved to '{final_policy_dir}' directory.")
    vec_env.close();
    writer.close()
    print("--- TRAINING COMPLETE ---")