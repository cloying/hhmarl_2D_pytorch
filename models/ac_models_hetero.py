# FILE: models/ac_models_hetero.py (Corrected)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

### --- FIX: Define the constant here to break the circular import --- ###
NODE_FEATURE_DIM = 10
### -------------------------------------------------------------------- ###


def init_weights(m, gain=1.0):
    """Orthogonal initialization for linear and GRU layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)


class RecurrentActor(nn.Module):
    def __init__(self, obs_dim_own, actor_logits_dim, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim_own, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh()
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.actor_head = nn.Linear(hidden_size, actor_logits_dim)
        self.apply(init_weights)

    def forward(self, obs_own, hidden_state):
        x = self.actor_net(obs_own)
        x = x.unsqueeze(1)
        gru_out, new_hidden_state = self.gru(x, hidden_state)
        logits = self.actor_head(gru_out.squeeze(1))
        return logits, new_hidden_state


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
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        graph_embedding = global_mean_pool(x, batch)
        return self.value_mlp(graph_embedding).squeeze(-1)