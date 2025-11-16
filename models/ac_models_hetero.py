# FILE: models/ac_models_hetero.py (Final Modular Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from envs.env_hetero import NODE_FEATURE_DIM


def init_weights(m, gain=1.0):
    """Orthogonal initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


### --- Standalone Actor Class --- ###
# This single class is flexible. We will create two separate instances of it
# in the training script, each with the correct dimensions for AC1 and AC2.
class Actor(nn.Module):
    def __init__(self, obs_dim_own, actor_logits_dim):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim_own, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, actor_logits_dim)
        )
        self.actor_net.apply(lambda m: init_weights(m))

    def forward(self, obs_own):
        return self.actor_net(obs_own)


### --- Standalone GNN Critic Class --- ###
# This class is also flexible and will be instantiated for each policy.
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
        # Unpack the graph batch object
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        # Perform graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Readout: Aggregate node embeddings into a single graph embedding
        graph_embedding = global_mean_pool(x, batch)

        # Return the final value prediction
        return self.value_mlp(graph_embedding).squeeze(-1)