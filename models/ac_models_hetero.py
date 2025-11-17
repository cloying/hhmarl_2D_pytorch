# FILE: models/ac_models_hetero.py (Reworked with Recurrent Actor)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from envs.env_hetero import NODE_FEATURE_DIM

def init_weights(m, gain=1.0):
    """Orthogonal initialization for linear and GRU layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)

### --- NEW: Recurrent Actor Class --- ###
# This Actor uses a GRU to maintain a hidden state, giving it memory.
class RecurrentActor(nn.Module):
    def __init__(self, obs_dim_own, actor_logits_dim, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size

        # Pre-RNN processing layers
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim_own, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )

        # Recurrent layer
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Post-RNN layer to produce action logits
        self.actor_head = nn.Linear(hidden_size, actor_logits_dim)

        self.apply(init_weights)

    def forward(self, obs_own, hidden_state):
        """
        Forward pass for the recurrent actor.

        Args:
            obs_own (Tensor): Batch of agent observations.
                              Shape: (batch_size, obs_dim_own)
            hidden_state (Tensor): Previous hidden state of the GRU.
                                   Shape: (1, batch_size, hidden_size)

        Returns:
            logits (Tensor): Action logits. Shape: (batch_size, actor_logits_dim)
            new_hidden_state (Tensor): Next hidden state. Shape: (1, batch_size, hidden_size)
        """
        # Process observation through the initial MLP layers
        x = self.actor_net(obs_own)

        # Add a sequence length dimension of 1 for the GRU
        x = x.unsqueeze(1)

        # Pass through GRU
        gru_out, new_hidden_state = self.gru(x, hidden_state)

        # Remove sequence dimension and get the final action logits
        logits = self.actor_head(gru_out.squeeze(1))

        return logits, new_hidden_state

### --- Standalone GNN Critic Class (Unchanged but kept for completeness) --- ###
class GNN_Critic(nn.Module):
    def __init__(self, in_channels=NODE_FEATURE_DIM, hidden_channels=128, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.Tanh(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.value_mlp.apply(lambda m: init_weights(m))

    def forward(self, graph_batch):
        # Unpack the graph batch object from PyTorch Geometric
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        # Perform graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Global mean pooling aggregates node features into a single graph-level feature vector
        graph_embedding = global_mean_pool(x, batch)

        # Return the final value prediction
        return self.value_mlp(graph_embedding).squeeze(-1)