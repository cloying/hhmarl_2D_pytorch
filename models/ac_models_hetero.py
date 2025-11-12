# FILE: models/ac_models_hetero.py (Improved Architecture)

import torch
import torch.nn as nn


def init_weights(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


SHARED_LAYER = lambda: nn.Sequential(nn.Linear(500, 500), nn.Tanh())


class EscapeActorCritic(nn.Module):
    def __init__(self, obs_dim_own, obs_dim_other, act_parts_own, act_parts_other, actor_logits_dim, own_state_split):
        super().__init__()
        s1, s2 = own_state_split

        # Actor head
        self.actor_inp1 = nn.Linear(s1, 150)
        self.actor_inp2 = nn.Linear(s2, 250)
        self.actor_inp3 = nn.Linear(obs_dim_own - s1 - s2, 100)
        self.actor_shared = SHARED_LAYER()
        self.actor_out = nn.Linear(500, actor_logits_dim)

        # Critic head
        critic_input_dim = obs_dim_own + act_parts_own + obs_dim_other + act_parts_other
        self.critic_inp = nn.Linear(critic_input_dim, 500)
        self.critic_shared = SHARED_LAYER()
        self.critic_out = nn.Linear(500, 1)

        self.apply(lambda m: init_weights(m))

    def get_action_logits(self, obs_own):
        s1, s2 = self.actor_inp1.in_features, self.actor_inp2.in_features
        _inp1, _inp2, _inp3 = obs_own.split([s1, s2, self.actor_inp3.in_features], dim=-1)

        x1 = torch.tanh(self.actor_inp1(_inp1))
        x2 = torch.tanh(self.actor_inp2(_inp2))
        x3 = torch.tanh(self.actor_inp3(_inp3))

        x = torch.cat((x1, x2, x3), dim=1)
        actor_features = self.actor_shared(x)
        return self.actor_out(actor_features)

    def get_value(self, critic_obs_dict):
        critic_input = torch.cat([
            critic_obs_dict['obs_1_own'], critic_obs_dict['act_1_own'],
            critic_obs_dict['obs_2'], critic_obs_dict['act_2']
        ], dim=-1)
        critic_features = torch.tanh(self.critic_inp(critic_input))
        critic_features = self.critic_shared(critic_features)
        return self.critic_out(critic_features).squeeze(-1)

    def forward(self, actor_obs, critic_obs_dict):
        return self.get_action_logits(actor_obs), self.get_value(critic_obs_dict)


class FightActorCritic(nn.Module):
    def __init__(self, obs_dim_own, obs_dim_other, act_parts_own, act_parts_other, actor_logits_dim,
                 own_state_split_size):
        super().__init__()
        # Actor head (simplified non-recurrent version for now)
        self.actor_net = nn.Sequential(
            nn.Linear(obs_dim_own, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, actor_logits_dim)
        )

        # Critic head
        critic_input_dim = obs_dim_own + act_parts_own + obs_dim_other + act_parts_other
        self.critic_net = nn.Sequential(
            nn.Linear(critic_input_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.apply(lambda m: init_weights(m))

    def get_action_logits(self, obs_own):
        return self.actor_net(obs_own)

    def get_value(self, critic_obs_dict):
        critic_input = torch.cat([
            critic_obs_dict['obs_1_own'], critic_obs_dict['act_1_own'],
            critic_obs_dict['obs_2'], critic_obs_dict['act_2']
        ], dim=-1)
        return self.critic_net(critic_input).squeeze(-1)

    def forward(self, actor_obs, critic_obs_dict):
        return self.get_action_logits(actor_obs), self.get_value(critic_obs_dict)