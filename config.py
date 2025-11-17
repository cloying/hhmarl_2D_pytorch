# FILE: config.py (Set num_workers=10)

import argparse
import os


class Config(object):
    def __init__(self, mode: int):
        self.mode = mode
        parser = argparse.ArgumentParser(description='HHMARL2D Training Config')

        # --- Training & Curriculum ---
        parser.add_argument('--level', type=int, default=1, help='Curriculum learning level (1-5)')
        parser.add_argument('--agent_mode', type=str, default="fight", help='Low-level agent mode: "fight" or "escape"')
        parser.add_argument('--opponent_policy_path', type=str, default=None,
                            help='Path to the directory containing opponent .pth models for self-play (Level 4+).')

        # --- Multi-Agent Configuration ---
        parser.add_argument('--num_agents', type=int, default=2 if mode == 0 else 3, help='Number of trainable agents')
        parser.add_argument('--num_opps', type=int, default=2 if mode == 0 else 3, help='Number of opponents')
        parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

        # --- Environment & General Training Parameters ---
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode during training')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and save images')
        parser.add_argument('--restore', type=bool, default=False, help='Restore training from a saved model checkpoint')
        parser.add_argument('--restore_path', type=str, default=None, help='Full path to the model checkpoint to restore')
        parser.add_argument('--reset_optimizers', action='store_true', help='If restoring a model, force the optimizers to be reset.')

        # --- Hardware & Parallelization ---
        parser.add_argument('--gpu', type=int, default=1, help='Set to 1 to use GPU, 0 for CPU')
        ### --- MODIFICATION: Set default workers to 10 --- ###
        parser.add_argument('--num_workers', type=int, default=10, help='Number of parallel workers for data collection')
        ### ----------------------------------------------- ###

        # --- Algorithm Hyperparameters ---
        parser.add_argument('--total_timesteps', type=int, default=4_000_000, help='Total timesteps for the training run.')
        parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the Adam optimizer.')
        parser.add_argument('--map_size', type=float, default=0.3 if mode == 0 else 0.5, help='Map size in km (value * 100)')
        parser.add_argument('--ent_coef', type=float, default=0.015, help='Entropy coefficient for PPO loss.')
        parser.add_argument('--checkpoint_interval', type=int, default=15, help='Save a model checkpoint every N updates.')
        parser.add_argument('--render_interval', type=int, default=15, help='Render a GIF every N updates.')
        # --- Reward Shaping ---
        parser.add_argument('--glob_frac', type=float, default=0.2, help='Fraction of reward sharing between agents')
        parser.add_argument('--rew_scale', type=int, default=1, help='Global reward scaling factor for sparse rewards')
        parser.add_argument('--friendly_kill', type=bool, default=True, help='Whether friendly fire is possible')
        parser.add_argument('--friendly_punish', type=bool, default=False,
                            help='If friendly fire occurs, punish both agents')
        parser.add_argument('--kill_reward_bonus', type=float, default=3.0,
                            help='Additional sparse reward for destroying an enemy.')
        parser.add_argument('--firing_reward', type=float, default=0.0,
                            help='DEPRECATED: Use high_prob_kill_reward instead.')
        parser.add_argument('--ammo_penalty', type=float, default=-0.01,
                            help='Dense penalty for firing under bad conditions.')

        ### --- NEW: Tactical Reward Shaping Arguments --- ###
        parser.add_argument('--tail_chase_bonus', type=float, default=0.04,
                            help='Dense reward for maintaining a position behind the opponent.')
        parser.add_argument('--high_prob_kill_reward', type=float, default=0.9,
                            help='Dense reward for firing from an advantageous position.')
        parser.add_argument('--energy_advantage_bonus', type=float, default=0.0005,
                            help='Dense reward for having a higher speed than the opponent.')
        ### ------------------------------------------- ###
        parser.add_argument('--use_reward_schedule', type=bool, default=True, help='Enable adaptive scaling of shaping rewards over time.')
        parser.add_argument('--shaping_scale_initial', type=float, default=0.01, help='Initial multiplier for dense shaping rewards.')
        parser.add_argument('--shaping_scale_final', type=float, default=0.0, help='Final multiplier for shaping rewards.')
        parser.add_argument('--shaping_decay_timesteps', type=int, default=None, help='Timesteps until scale reaches final value. (Default: 50% of total_timesteps)')
        parser.add_argument('--eval_info', type=bool, default=True if mode == 2 else False, help='Provide eval statistic in step() function or not.')
        parser.add_argument('--eval_hl', type=bool, default=True, help='True=evaluation with Commander, False=evaluation of low-level policies')
        parser.add_argument('--eval_level_ag', type=int, default=5, help="Agent's pre-trained low-level policy")
        parser.add_argument('--eval_level_opp', type=int, default=4, help="Opponent's pre-trained low-level policy")
        parser.add_argument('--hier_action_assess', type=bool, default=True, help='Provide action rewards to guide hierarchical training.')
        parser.add_argument('--hier_opp_fight_ratio', type=int, default=75, help='Opponent fight policy selection probability [in %]')

        self.args = parser.parse_args()
        self.set_metrics()

    def set_metrics(self):
        if self.mode == 0: log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
        else: log_name = f'Commander_{self.args.num_agents}_vs_{self.args.num_opps}'
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', log_name)
        if not self.args.restore:
            if os.path.exists(os.path.join(self.args.log_path, "checkpoints")): self.args.restore, self.args.restore_path = True, self.args.log_path
            elif self.mode == 0 and self.args.level > 1:
                prev_level_path = os.path.join(os.path.dirname(__file__), 'results', f'L{self.args.level - 1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')
                if os.path.exists(prev_level_path): self.args.restore, self.args.restore_path = True, prev_level_path
        horizon_level = {1: 150, 2: 200, 3: 300, 4: 350, 5: 400}
        self.args.horizon = horizon_level.get(self.args.level, 500)
        if self.args.shaping_decay_timesteps is None: self.args.shaping_decay_timesteps = self.args.total_timesteps // 2
        self.args.total_num, self.args.env_config = self.args.num_agents + self.args.num_opps, {"args": self.args}

    @property
    def get_arguments(self):
        return self.args