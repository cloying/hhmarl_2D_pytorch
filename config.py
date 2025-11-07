
import argparse
import os
import datetime


class Config(object):
    """
    Manages all training, evaluation, and environment configurations.
    """

    def __init__(self, mode: int):
        self.mode = mode
        parser = argparse.ArgumentParser(description='HHMARL2D Training Config')

        # --- Training Mode Parameters ---
        parser.add_argument('--level', type=int, default=1, help='Curriculum learning level (1-5)')
        parser.add_argument('--horizon', type=int, default=500, help='Maximum steps per episode')
        parser.add_argument('--agent_mode', type=str, default="fight", help='Low-level agent mode: "fight" or "escape"')

        # --- Multi-Agent Configuration ---
        parser.add_argument('--num_agents', type=int, default=2 if mode == 0 else 3, help='Number of trainable agents')
        parser.add_argument('--num_opps', type=int, default=2 if mode == 0 else 3, help='Number of opponents')

        # --- Environment & General Training Parameters ---
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode during training')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and save images')
        parser.add_argument('--restore', type=bool, default=False,
                            help='Restore training from a saved model checkpoint')
        parser.add_argument('--restore_path', type=str, default=None,
                            help='Full path to the model checkpoint to restore')
        parser.add_argument('--log_name', type=str, default=None, help='Experiment name for logging')
        parser.add_argument('--log_path', type=str, default=None,
                            help='Full path to the directory for saving logs and models')

        # --- Hardware and Parallelization ---
        parser.add_argument('--gpu', type=float, default=0, help='Fraction of GPU to use (0 for CPU, 1 for full GPU)')
        parser.add_argument('--num_workers', type=int, default=4,
                            help='Number of parallel workers/threads for data collection')

        # --- Algorithm Hyperparameters ---
        parser.add_argument('--epochs', type=int, default=10000, help='Total number of training epochs')
        parser.add_argument('--batch_size', type=int, default=2000 if mode == 0 else 1000,
                            help='PPO training batch size')
        parser.add_argument('--mini_batch_size', type=int, default=256, help='PPO training mini-batch size')
        parser.add_argument('--map_size', type=float, default=0.3 if mode == 0 else 0.5,
                            help='Map size in km (value * 100)')

        # --- Reward Shaping Parameters ---
        parser.add_argument('--glob_frac', type=float, default=0, help='Fraction of reward sharing between agents')
        parser.add_argument('--rew_scale', type=int, default=1, help='Global reward scaling factor')
        parser.add_argument('--esc_dist_rew', type=bool, default=False,
                            help='Activate per-timestep reward for Escape Training')
        parser.add_argument('--hier_action_assess', type=bool, default=True,
                            help='Provide action rewards to guide hierarchical training')
        parser.add_argument('--friendly_kill', type=bool, default=True, help='Whether friendly fire is possible')
        parser.add_argument('--friendly_punish', type=bool, default=False,
                            help='If friendly fire occurs, punish both agents')

        # --- Evaluation-Specific Parameters ---
        parser.add_argument('--eval_info', type=bool, default=True if mode == 2 else False,
                            help='Provide detailed evaluation statistics in the step() info dict')
        parser.add_argument('--eval_hl', type=bool, default=True,
                            help='True=evaluation with Commander, False=evaluation of low-level policies')
        parser.add_argument('--eval_level_ag', type=int, default=5,
                            help="Agent's pre-trained low-level policy to use for evaluation")
        parser.add_argument('--eval_level_opp', type=int, default=4,
                            help="Opponent's pre-trained low-level policy to use for evaluation")
        parser.add_argument('--hier_opp_fight_ratio', type=int, default=75,
                            help='Opponent fight policy selection probability [in %]')

        self.args = parser.parse_args()
        self.set_metrics()

    def set_metrics(self):
        """
        Sets up logging paths and handles logic for restoring models automatically.
        """
        if self.mode == 0:
            log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
        else:
            log_name = f'Commander_{self.args.num_agents}_vs_{self.args.num_opps}'

        self.args.log_name = log_name
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if not self.args.restore and self.mode == 0:
            if self.args.agent_mode == "fight" and self.args.level > 1:
                prev_level_path = os.path.join(os.path.dirname(__file__), 'results',
                                               f'L{self.args.level - 1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')
                if os.path.exists(prev_level_path):
                    self.args.restore = True

        if self.args.restore and self.args.restore_path is None:
            if self.mode == 0:
                if self.args.agent_mode == "fight":
                    restore_dir = f'L{self.args.level - 1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
                else:
                    restore_dir = f'L3_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
                self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', restore_dir)
            else:
                raise NameError('Specify full restore path to Commander Policy.')

        if self.mode == 0:
            horizon_level = {1: 150, 2: 200, 3: 300, 4: 350, 5: 400}
            self.args.horizon = horizon_level.get(self.args.level, 500)
        else:
            self.args.horizon = 500

        if self.mode == 2 and self.args.eval_hl:
            self.args.eval_level_ag = self.args.eval_level_opp = 5

        self.args.eval = True if self.args.render else self.args.eval

        # --- FIX: Calculate and add total_num to the args object ---
        # This attribute is required by the environment to initialize its state.
        self.args.total_num = self.args.num_agents + self.args.num_opps

        # Pass all arguments to the environment config
        self.args.env_config = {"args": self.args}

    @property
    def get_arguments(self):
        """A property to easily access the parsed arguments object."""
        return self.args