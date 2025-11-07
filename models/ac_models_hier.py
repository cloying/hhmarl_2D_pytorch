# --- Dependencies ---
import torch
import torch.nn as nn
import numpy as np

# --- Constants ---
# These are derived from the environment and model architecture.
# It's good practice to define them in a central place.
N_OPP_HL = 2  # Number of opponents the commander can sense
OBS_OPP = 10  # Observation size for a single opponent
OBS_DIM = 14 + OBS_OPP * N_OPP_HL  # Total observation dimension for the commander agent


# --- Helper Function for Weight Initialization ---
def init_weights(m, gain=1.0):
    """
    Applies orthogonal initialization to Linear and GRU layers.
    This helps with training stability.
    """
    if isinstance(m, (nn.Linear, nn.GRU)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


# --- Actor-Critic Model for the Commander ---

class CommanderActorCritic(nn.Module):
    """
    A pure PyTorch implementation of the CommanderGRU model.

    This module implements an Actor-Critic architecture with Gated Recurrent Units (GRUs)
    for processing sequential observations. It features a centralized critic, meaning
    the critic's value estimate is based on the observations and actions of all friendly agents,
    while the actor's policy is based only on its own observation.
    """

    def __init__(self, obs_dim, action_dim, hidden_size=200):
        """
        Initializes the neural network layers.

        Args:
            obs_dim (int): The dimension of the commander's observation space.
            action_dim (int): The number of possible discrete actions for the commander.
            hidden_size (int): The size of the hidden state for the GRUs.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # --- Actor Network Layers ---
        # The actor's network processes the agent's own observations to decide on an action.
        self.actor_inp1 = nn.Linear(4, 50)  # Processes core agent state (pos, vel, heading)
        self.actor_inp2 = nn.Linear(N_OPP_HL * OBS_OPP, 200)  # Processes opponent information
        self.actor_inp3 = nn.Linear(10, 50)  # Processes friendly agent information
        self.actor_inp4 = nn.Linear(obs_dim, hidden_size)  # Processes the full observation for the RNN
        self.actor_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # --- Critic Network Layers ---
        # The critic's network processes observations and actions from ALL agents
        # to get a better estimate of the state value (Centralized Critic).
        self.critic_v1 = nn.Linear(obs_dim + 1, 100)  # Processes agent 1's obs + action
        self.critic_v2 = nn.Linear(obs_dim + 1, 100)  # Processes agent 2's obs + action
        self.critic_v3 = nn.Linear(obs_dim + 1, 100)  # Processes agent 3's obs + action
        self.critic_v4 = nn.Linear(3 * (obs_dim + 1), hidden_size)  # Processes combined info for the RNN
        self.critic_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # --- Shared and Output Layers ---
        # After RNN processing, features are passed through shared layers.
        self.shared_layer = nn.Sequential(
            nn.Linear(500, 500),
            nn.Tanh()
        )
        self.actor_out = nn.Linear(500, action_dim)  # Final layer for action logits
        self.critic_out = nn.Linear(500, 1)  # Final layer for state value

        # Apply orthogonal initialization for training stability
        self.apply(init_weights)

    def forward(self, obs_dict, actor_hidden, critic_hidden):
        """
        Performs the forward pass for both the actor and the critic.

        Args:
            obs_dict (dict): A dictionary of tensors containing observations and actions
                             for all agents (e.g., {'obs_1_own': ..., 'act_1_own': ...}).
            actor_hidden (torch.Tensor): The previous hidden state for the actor's GRU.
            critic_hidden (torch.Tensor): The previous hidden state for the critic's GRU.

        Returns:
            tuple: A tuple containing:
                - action_logits (torch.Tensor): The output logits for the policy.
                - value (torch.Tensor): The predicted state value from the critic.
                - new_actor_hidden (torch.Tensor): The next hidden state for the actor's GRU.
                - new_critic_hidden (torch.Tensor): The next hidden state for the critic's GRU.
        """
        # --- Actor Forward Pass ---
        obs_own = obs_dict['obs_1_own']  # Actor only uses its own observation

        # Process different parts of the observation through separate MLPs
        actor_x1 = torch.tanh(self.actor_inp1(obs_own[:, :4]))
        actor_x2 = torch.tanh(self.actor_inp2(obs_own[:, 4:4 + N_OPP_HL * OBS_OPP]))
        actor_x3 = torch.tanh(self.actor_inp3(obs_own[:, 4 + N_OPP_HL * OBS_OPP:]))

        # Combine processed features
        actor_x_combined = torch.cat((actor_x1, actor_x2, actor_x3), dim=1)

        # Prepare full observation for RNN
        x_full = torch.tanh(self.actor_inp4(obs_own))
        # Add a sequence dimension for the GRU: (batch_size, features) -> (batch_size, 1, features)
        x_full_seq = x_full.unsqueeze(1)

        # Pass through GRU
        rnn_out_act, new_actor_hidden = self.actor_rnn(x_full_seq, actor_hidden)
        rnn_out_act = rnn_out_act.squeeze(1)  # Remove sequence dimension

        # Combine all features and pass through shared layer and final actor layer
        final_actor_features = torch.cat((actor_x_combined, rnn_out_act), dim=1)
        action_logits = self.actor_out(self.shared_layer(final_actor_features))

        # --- Critic Forward Pass (Centralized) ---
        # The critic uses observations and actions from all agents for a better value estimate.
        v1_in = torch.cat((obs_dict["obs_1_own"], obs_dict["act_1_own"]), dim=1)
        v2_in = torch.cat((obs_dict["obs_2"], obs_dict["act_2"]), dim=1)
        v3_in = torch.cat((obs_dict["obs_3"], obs_dict["act_3"]), dim=1)
        v4_in = torch.cat((v1_in, v2_in, v3_in), dim=1)

        # Process each agent's info
        critic_z1 = torch.tanh(self.critic_v1(v1_in))
        critic_z2 = torch.tanh(self.critic_v2(v2_in))
        critic_z3 = torch.tanh(self.critic_v3(v3_in))

        # Combine processed features
        critic_z_combined = torch.cat((critic_z1, critic_z2, critic_z3), dim=1)

        # Prepare full centralized observation for RNN
        z_full = torch.tanh(self.critic_v4(v4_in))
        z_full_seq = z_full.unsqueeze(1)  # Add sequence dimension

        # Pass through GRU
        rnn_out_val, new_critic_hidden = self.critic_rnn(z_full_seq, critic_hidden)
        rnn_out_val = rnn_out_val.squeeze(1)  # Remove sequence dimension

        # Combine all features and pass through shared layer and final critic layer
        final_critic_features = torch.cat((critic_z_combined, rnn_out_val), dim=1)
        value = self.critic_out(self.shared_layer(final_critic_features))

        return action_logits, value, new_actor_hidden, new_critic_hidden