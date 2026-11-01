"""Dimension-configurable neural networks; no robot or task dependencies."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, act=nn.SiLU, zero_last=False, last_gain=None):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend((nn.Linear(in_dim, width), act()))
            in_dim = width
        last = nn.Linear(in_dim, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DynamicsMLP(nn.Module):
    """Predict a state-adapter-defined delta from features, action and context."""

    def __init__(self, *, feat_dim, act_dim, xi_dim, hidden, out_dim):
        super().__init__()
        self.mlp = MLP(feat_dim + act_dim + xi_dim, hidden, out_dim, zero_last=True)

    def forward(self, feat, action, xi_norm):
        return self.mlp(torch.cat((feat, action, xi_norm), -1))


class GoalEncoder(nn.Module):
    def __init__(self, *, scale, hidden, latent_dim):
        super().__init__()
        self.mlp = MLP(len(scale), hidden, latent_dim)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, goal):
        return torch.tanh(self.mlp(goal * self.scale))


class ActionDecoder(nn.Module):
    def __init__(self, *, scale, hidden, latent_dim, action_dim, last_gain):
        super().__init__()
        self.mlp = MLP(latent_dim + len(scale), hidden, action_dim, last_gain=last_gain)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, latent, history):
        return self.mlp(torch.cat((latent, history * self.scale), -1))


class GoalConditionedPolicy(nn.Module):
    def __init__(
        self,
        *,
        goal_scale,
        obs_scale,
        encoder_hidden,
        decoder_hidden,
        latent_dim,
        action_dim,
        last_gain,
        interface="motion",
    ):
        super().__init__()
        self.default_interface = interface
        self.encoders = nn.ModuleDict(
            {
                interface: GoalEncoder(
                    scale=goal_scale,
                    hidden=encoder_hidden,
                    latent_dim=latent_dim,
                )
            }
        )
        self.decoder = ActionDecoder(
            scale=obs_scale,
            hidden=decoder_hidden,
            latent_dim=latent_dim,
            action_dim=action_dim,
            last_gain=last_gain,
        )

    def forward(self, obs_hist, goal, interface=None):
        interface = self.default_interface if interface is None else interface
        return self.decoder(self.encoders[interface](goal), obs_hist)


class ValueNetwork(nn.Module):
    def __init__(self, *, feat_dim, goal_scale, context_dim, error_dim, hidden):
        super().__init__()
        self.mlp = MLP(feat_dim + len(goal_scale) + context_dim + error_dim, hidden, 1)
        self.register_buffer("goal_scale", torch.as_tensor(goal_scale, dtype=torch.float32))

    def forward(self, feat, goal, context, error):
        return self.mlp(torch.cat((feat, goal * self.goal_scale, context, error), -1)).squeeze(-1)


class TerminationHead(nn.Module):
    """Binary logits from scaled signed and absolute task error features."""

    def __init__(self, *, scales, hidden):
        super().__init__()
        self.mlp = MLP(2 * len(scales), hidden, 1)
        self.register_buffer("scale", torch.tensor(scales, dtype=torch.float32))

    def forward(self, features):
        x = features * self.scale
        return self.mlp(torch.cat((x, x.abs()), -1)).squeeze(-1)
