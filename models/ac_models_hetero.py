# FILE: models/ac_models_hetero.py (Corrected and Final Version)

import torch
import torch.nn as nn


def init_weights(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


SHARED_LAYER = lambda: nn.Sequential(nn.Linear(500, 500), nn.Tanh())


class EscapeActorCritic(nn.Module):
    # --- FIX: Updated __init__ signature for clarity ---
    def __init__(self, obs_dim_own, obs_dim_other, act_parts_own, act_parts_other, actor_logits_dim, own_state_split):
        super().__init__()
        s1, s2 = own_state_split
        self.inp1 = nn.Linear(s1, 150)
        self.inp2 = nn.Linear(s2, 250)
        self.inp3 = nn.Linear(5, 100)
        self.shared_layer = SHARED_LAYER()
        # Actor head now uses the explicit logits dimension
        self.actor_out = nn.Linear(500, actor_logits_dim)

        # Critic head now uses the explicit action parts dimension
        critic_input_dim = obs_dim_own + act_parts_own + obs_dim_other + act_parts_other
        self.critic_inp = nn.Linear(critic_input_dim, 500)
        self.critic_shared = SHARED_LAYER()
        self.critic_out = nn.Linear(500, 1)

        self.apply(init_weights)

    def get_action_logits(self, obs_own):
        split1, split2 = self.inp1.in_features, self.inp2.in_features
        _inp1, _inp2, _inp3 = obs_own.split([split1, split2, self.inp3.in_features], dim=-1)
        x1 = torch.tanh(self.inp1(_inp1))
        x2 = torch.tanh(self.inp2(_inp2))
        x3 = torch.tanh(self.inp3(_inp3))
        x = torch.cat((x1, x2, x3), dim=1)
        actor_features = self.shared_layer(x)
        return self.actor_out(actor_features)

    def get_value(self, critic_obs_dict):
        critic_input = torch.cat([
            critic_obs_dict['obs_1_own'], critic_obs_dict['act_1_own'],
            critic_obs_dict['obs_2'], critic_obs_dict['act_2']
        ], dim=1)
        critic_features = torch.tanh(self.critic_inp(critic_input))
        critic_features = self.critic_shared(critic_features)
        return self.critic_out(critic_features).squeeze(-1)

    def forward(self, actor_obs, critic_obs_dict):
        logits = self.get_action_logits(actor_obs)
        value = self.get_value(critic_obs_dict)
        return logits, value


class FightActorCritic(nn.Module):
    # --- FIX: Updated __init__ signature for clarity ---
    def __init__(self, obs_dim_own, obs_dim_other, act_parts_own, act_parts_other, actor_logits_dim,
                 own_state_split_size):
        super().__init__()
        self.ss_agent = own_state_split_size

        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 2, batch_first=True)
        self.shared_layer = SHARED_LAYER()

        self.actor_inp1 = nn.Linear(self.ss_agent, 200)
        self.actor_inp2 = nn.Linear(obs_dim_own - self.ss_agent, 200)
        self.actor_inp3 = nn.Linear(obs_dim_own, 100)
        self.actor_out = nn.Linear(500, actor_logits_dim)

        # Critic layers now use the correct action parts dimensions
        critic_v1_dim = obs_dim_own + act_parts_own
        critic_v2_dim = obs_dim_other + act_parts_other
        self.critic_v1 = nn.Linear(critic_v1_dim, 175)
        self.critic_v2 = nn.Linear(critic_v2_dim, 175)
        self.critic_v3 = nn.Linear(critic_v1_dim + critic_v2_dim, 150)
        self.critic_shared = SHARED_LAYER()
        self.critic_out = nn.Linear(500, 1)

        self.apply(init_weights)

    def get_action_logits(self, obs_own):
        _inp1, _inp2 = obs_own.split([self.ss_agent, self.actor_inp2.in_features], dim=-1)
        x1 = torch.tanh(self.actor_inp1(_inp1))
        x2 = torch.tanh(self.actor_inp2(_inp2))
        x = torch.cat((x1, x2), dim=1)
        x_full = torch.tanh(self.actor_inp3(obs_own))
        x_ft = x_full.unsqueeze(1)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full_norm = nn.functional.normalize(x_full + x_att.squeeze(1))
        final_actor_features = torch.cat((x, x_full_norm), dim=1)
        return self.actor_out(self.shared_layer(final_actor_features))

    def get_value(self, critic_obs_dict):
        v1_in = torch.cat((critic_obs_dict['obs_1_own'], critic_obs_dict['act_1_own']), dim=1)
        v2_in = torch.cat((critic_obs_dict['obs_2'], critic_obs_dict['act_2']), dim=1)
        v3_in = torch.cat((v1_in, v2_in), dim=1)
        y1 = torch.tanh(self.critic_v1(v1_in))
        y2 = torch.tanh(self.critic_v2(v2_in))
        y = torch.cat((y1, y2), dim=1)
        y_full = torch.tanh(self.critic_v3(v3_in))
        y_ft = y_full.unsqueeze(1)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full_norm = nn.functional.normalize(y_full + y_att.squeeze(1))
        final_critic_features = torch.cat((y, y_full_norm), dim=1)
        return self.critic_out(self.critic_shared(final_critic_features)).squeeze(-1)

    def forward(self, actor_obs, critic_obs_dict):
        logits = self.get_action_logits(actor_obs)
        value = self.get_value(critic_obs_dict)
        return logits, value