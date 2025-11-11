# FILE: models/ac_models_hier.py (Corrected and Generic)

# --- Dependencies ---
import torch
import torch.nn as nn
import numpy as np

# --- Constants ---
N_OPP_HL = 2
OBS_OPP = 10
OBS_DIM = 14 + OBS_OPP * N_OPP_HL


# --- Helper Function for Weight Initialization ---
def init_weights(m, gain=1.0):
    if isinstance(m, (nn.Linear, nn.GRU)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


# --- Actor-Critic Model for the Commander ---
class CommanderActorCritic(nn.Module):
    """
    A generic, pure PyTorch implementation of the CommanderGRU model.
    It supports a variable number of agents for centralized training.
    """

    def __init__(self, obs_dim, action_dim, num_agents, hidden_size=200):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents

        # --- Actor Network (Decentralized) ---
        # Each agent uses the same network weights but only on its own observation.
        self.actor_inp1 = nn.Linear(4, 50)
        self.actor_inp2 = nn.Linear(N_OPP_HL * OBS_OPP, 200)
        self.actor_inp3 = nn.Linear(10, 50)
        self.actor_inp4 = nn.Linear(obs_dim, hidden_size)
        self.actor_rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.actor_shared = nn.Sequential(nn.Linear(300 + hidden_size, 500), nn.Tanh())
        self.actor_out = nn.Linear(500, action_dim)

        # --- Critic Network (Centralized) ---
        # The critic processes the combined observations of ALL agents.
        critic_input_dim = num_agents * obs_dim
        self.critic_mlp = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.critic_out = nn.Linear(hidden_size, 1)

        self.apply(init_weights)

    def forward(self, obs_batch, actor_hidden):
        """
        Performs the forward pass for both actor and critic.

        Args:
            obs_batch (torch.Tensor): A batch of observations for all agents.
                                      Shape: (batch_size, num_agents, obs_dim)
            actor_hidden (torch.Tensor): The previous hidden state for the actor's GRU.
                                         Shape: (1, batch_size * num_agents, hidden_size)

        Returns:
            - action_logits (torch.Tensor): Shape (batch_size * num_agents, action_dim)
            - value (torch.Tensor): Shape (batch_size, num_agents, 1)
            - new_actor_hidden (torch.Tensor): Shape (1, batch_size * num_agents, hidden_size)
        """
        batch_size, num_agents, obs_dim = obs_batch.shape

        # Flatten agents into the batch dimension for parallel processing
        flat_obs = obs_batch.view(batch_size * num_agents, obs_dim)

        # --- Actor Forward Pass (Decentralized Execution) ---
        actor_x1 = torch.tanh(self.actor_inp1(flat_obs[:, :4]))
        actor_x2 = torch.tanh(self.actor_inp2(flat_obs[:, 4:4 + N_OPP_HL * OBS_OPP]))
        actor_x3 = torch.tanh(self.actor_inp3(flat_obs[:, 4 + N_OPP_HL * OBS_OPP:]))
        actor_x_combined = torch.cat((actor_x1, actor_x2, actor_x3), dim=1)

        x_full = torch.tanh(self.actor_inp4(flat_obs))
        x_full_seq = x_full.unsqueeze(1)  # Add sequence dimension for GRU

        # Reshape hidden state to (1, batch * agents, hidden)
        rnn_out_act, new_actor_hidden = self.actor_rnn(x_full_seq, actor_hidden)

        final_actor_features = torch.cat((actor_x_combined, rnn_out_act.squeeze(1)), dim=1)
        action_logits = self.actor_out(self.actor_shared(final_actor_features))

        # --- Critic Forward Pass (Centralized Value) ---
        # Concatenate all agent observations along the feature dimension
        central_obs = obs_batch.view(batch_size, -1)

        critic_features = self.critic_mlp(central_obs)
        value_flat = self.critic_out(critic_features)  # Shape: (batch_size, 1)

        # Broadcast the centralized value to each agent
        value = value_flat.repeat(1, num_agents)  # Shape: (batch_size, num_agents)

        return action_logits, value, new_actor_hidden