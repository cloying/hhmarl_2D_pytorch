# FILE: evaluation.py (PyTorch-compatible and Corrected)

# --- Dependencies ---
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from config import Config

# --- Local Imports ---
# Import the gymnasium-compatible environment and the PyTorch models
from envs.env_hier import HighLevelEnv
from models.ac_models_hier import CommanderActorCritic  # The PyTorch model for the commander

# --- Constants ---
N_EVALS = 1000  # Number of episodes to run for evaluation
# Define the name of the saved model file (e.g., commander_model.pth)
MODEL_FILENAME = "Commander_3_vs_3.pth"


# --- Evaluation Function ---

def evaluate(args, env, policy_model, epoch, eval_stats, eval_log, device):
    """
    Runs a single evaluation episode.

    Args:
        args: Configuration arguments.
        env: The environment instance.
        policy_model: The trained PyTorch model (or None for low-level eval).
        epoch (int): The current evaluation episode number, used for logging.
        eval_stats (dict): A dictionary to accumulate statistics.
        eval_log (str): Path to the directory for saving logs and plots.
        device: The torch device (e.g., 'cuda' or 'cpu').
    """
    state, _ = env.reset()
    reward_sum = 0
    done = False
    step = 0

    # --- FIX: Initialize hidden states to None ---
    # This guarantees the variables exist even if not used (i.e., in low-level eval).
    actor_hidden, critic_hidden = None, None

    # If evaluating the commander, create the initial zero-tensor hidden state.
    if args.eval_hl and policy_model:
        # Shape: (num_layers, batch_size, hidden_size)
        actor_hidden = torch.zeros(1, 1, policy_model.hidden_size, device=device)
        critic_hidden = torch.zeros(1, 1, policy_model.hidden_size, device=device)  # Not used in eval

    while not done:
        actions = {}
        if args.eval_hl and policy_model:
            # --- High-Level (Commander) Policy Evaluation ---
            # The commander model makes a decision for each agent sequentially.
            for ag_id, ag_s in state.items():
                # Convert observation to a tensor and add a batch dimension
                obs_tensor = torch.from_numpy(ag_s).float().unsqueeze(0).to(device)

                # Get a deterministic action from the actor part of the model
                # We assume a get_action method is defined on the model for inference.
                # This method only runs the actor forward pass.
                # It takes the previous hidden state and returns the next one.
                with torch.no_grad():
                    action, actor_hidden = policy_model.get_action(obs_tensor, actor_hidden, deterministic=True)
                actions[ag_id] = action.cpu().item()
        else:
            # --- Low-Level Policy Evaluation ---
            # If no commander is involved, assign the 'fight' action (1).
            # The HighLevelEnv is designed to intercept this and use its pre-loaded
            # low-level 'fight' policies to execute the action.
            for n in range(1, args.num_agents + 1):
                actions[n] = 1

        # Step the environment
        state, rew, terminateds, truncateds, info = env.step(actions)
        done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        for r in rew.values():
            reward_sum += r

        step += 1

        # Accumulate stats from the info dictionary
        if info:
            for k, v in info.items():
                if k in eval_stats:
                    eval_stats[k] += v
            if "total_n_actions" in eval_stats:
                eval_stats["total_n_actions"] += (len(actions) if isinstance(actions, dict) else 1)

    # Plot the final state of the episode for visualization
    if epoch % 100 == 0 and eval_log:
        env.plot(Path(eval_log, f"Ep_{epoch}_Step_{step}_Rew_{round(reward_sum, 2)}.png"))


def postprocess_eval(ev, eval_file):
    """Calculates final metrics and saves them to a JSON file."""
    # This function is framework-agnostic and remains unchanged.
    win = (ev["agents_win"] / N_EVALS) * 100
    lose = (ev["opps_win"] / N_EVALS) * 100
    draw = (ev["draw"] / N_EVALS) * 100
    fight = (ev.get("agent_fight", 0) / (ev.get("agent_steps", 1) or 1)) * 100
    esc = (ev.get("agent_escape", 0) / (ev.get("agent_steps", 1) or 1)) * 100
    fight_opp = (ev.get("opp_fight", 0) / (ev.get("opp_steps", 1) or 1)) * 100
    esc_opp = (ev.get("opp_escape", 0) / (ev.get("opp_steps", 1) or 1)) * 100
    opp1 = (ev.get("opp1", 0) / (ev.get("agent_fight", 1) or 1)) * 100
    opp2 = (ev.get("opp2", 0) / (ev.get("agent_fight", 1) or 1)) * 100
    opp3 = (ev.get("opp3", 0) / (ev.get("agent_fight", 1) or 1)) * 100

    evals = {"win": win, "lose": lose, "draw": draw, "fight": fight, "esc": esc,
             "fight_opp": fight_opp, "esc_opp": esc_opp, "opp1": opp1, "opp2": opp2, "opp3": opp3}

    print("\n------ RESULTS ------")
    for k, v in evals.items():
        print(f"{k}: {round(v, 2)}")

    with open(eval_file, 'w') as file:
        json.dump(evals, file, indent=4)


# --- Main Execution Block ---

if __name__ == "__main__":
    t1 = time.time()

    # --- Setup ---
    args = Config(2).get_arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_base = os.path.join(os.getcwd(), 'results')
    # Path to the saved PyTorch model file
    model_path = os.path.join(log_base, MODEL_FILENAME)

    config_name = "Commander_" if args.eval_hl else "Low-Level_"
    config_name += f"{args.num_agents}-vs-{args.num_opps}"
    eval_log = os.path.join(log_base, f"EVAL_{config_name}")
    eval_file = os.path.join(eval_log, f"Metrics_{config_name}.json")
    if not os.path.exists(eval_log):
        os.makedirs(eval_log)

    # --- Environment and Model Initialization ---
    env = HighLevelEnv(args.env_config)

    policy_model = None
    if args.eval_hl:
        print(f"Loading high-level model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. Please ensure it is trained and saved correctly.")

        action_dim = env.action_space.n
        policy_model = CommanderActorCritic(obs_dim=OBS_DIM, action_dim=action_dim).to(device)
        policy_model.load_state_dict(torch.load(model_path, map_location=device))
        policy_model.eval()
    else:
        print("Evaluating pre-loaded low-level policies.")

    # --- Evaluation Loop ---
    eval_stats = {
        "agents_win": 0, "opps_win": 0, "draw": 0, "agent_fight": 0,
        "agent_escape": 0, "opp_fight": 0, "opp_escape": 0,
        "agent_steps": 0, "opp_steps": 0, "total_n_actions": 0,
        "opp1": 0, "opp2": 0, "opp3": 0
    }

    print(f"Running {N_EVALS} evaluation episodes...")
    iters = tqdm.trange(N_EVALS, leave=True)
    for n in iters:
        evaluate(args, env, policy_model, n, eval_stats, eval_log, device)

    # --- Post-processing and Results ---
    postprocess_eval(eval_stats, eval_file)

    print(f"\n------ TIME: {round(time.time() - t1, 3)} sec. ------")