# FILE: models/ac_models_hetero.py (GNN Critic with Edge Features)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

# --- MODIFICATION: Define feature dimensions for nodes and new edges --- ###
NODE_FEATURE_DIM = 10
EDGE_FEATURE_DIM = 6


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


### --- MODIFICATION START: Create a GNN Layer that uses Edge Features --- ###
class EdgeGCNConv(MessagePassing):
    """
    A custom message passing layer that incorporates edge features into the message
    before aggregation.
    """

    def __init__(self, node_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')  # Use mean aggregation
        # MLP that processes the combined features of a source node and its edge to a target
        self.message_mlp = nn.Sequential(
            nn.Linear(node_channels + edge_channels, out_channels),
            nn.ReLU()
        )
        # MLP that updates a node's features based on its original features and aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(node_channels + out_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        # The propagate method will call message(), aggregate(), and update()
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, node_channels] (features of source nodes)
        # edge_attr has shape [E, edge_channels]
        # Combine them and pass through the message MLP
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels] (aggregated messages)
        # x has shape [N, node_channels] (features of target nodes)
        # Combine them and pass through the update MLP
        return self.update_mlp(torch.cat([x, aggr_out], dim=1))


class GNN_Critic(nn.Module):
    """
    The GNN Critic, now updated to use the EdgeGCNConv layer.
    """

    def __init__(self, in_channels=NODE_FEATURE_DIM, edge_channels=EDGE_FEATURE_DIM, hidden_channels=128,
                 out_channels=1):
        super().__init__()
        self.conv1 = EdgeGCNConv(in_channels, edge_channels, hidden_channels)
        self.conv2 = EdgeGCNConv(hidden_channels, edge_channels, hidden_channels)
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.Tanh(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.value_mlp.apply(lambda m: init_weights(m))

    def forward(self, graph_batch):
        x, edge_index, edge_attr, batch = graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch

        # Pass node and edge features into the convolutional layers
        x = self.conv1(x, edge_index, edge_attr)
        # For the second layer, the input node dimension is now hidden_channels
        x = self.conv2(x, edge_index, edge_attr)

        # Pool the node features of the entire graph to get a single graph embedding
        graph_embedding = global_mean_pool(x, batch)

        return self.value_mlp(graph_embedding).squeeze(-1)
### --- MODIFICATION END --- ###