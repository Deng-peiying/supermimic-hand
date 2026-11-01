"""SuperMimic WM：显式积分动力学 + 学习终止头；观测和奖励解析计算。"""

import torch
from torch import nn
from torch.nn import functional as F

from supermimic.algorithms.networks import DynamicsMLP, TerminationHead

from .state_spec import CONTEXT_DIM, DELTA_DIM, FEAT_DIM, canonical, error_vector, integrate


class WorldModel(nn.Module):
    def __init__(self, hidden=(1024, 1024, 1024), dt=0.02, drop_distance=0.1):
        super().__init__()
        self.dt = dt
        self.dynamics = DynamicsMLP(
            feat_dim=FEAT_DIM, act_dim=22, xi_dim=CONTEXT_DIM, hidden=hidden, out_dim=DELTA_DIM
        )
        self.term_head = TerminationHead(scales=(1 / drop_distance,) * 3, hidden=(64, 64))

    def forward(self, state, action, xi=None):
        if xi is None:
            xi = torch.ones((*state.shape[:-1], CONTEXT_DIM), device=state.device)
        return integrate(state, self.dynamics(canonical(state), action, (xi - 1) / 0.2), self.dt)

    def termination_logits(self, state, reference):
        return self.term_head(state[..., 44:47] - reference[..., 44:47])

    def loss(self, state, action, next_state):
        return error_vector(self(state, action), next_state).square().mean()

    def window_loss(self, states, actions, xi, done, refs):
        state = states[0]
        state_losses, term_losses = [], []
        for t in range(actions.shape[0]):
            state = self(state, actions[t], xi)
            err = error_vector(state, states[t + 1]).square()
            state_losses.append(err.mean())
            term_losses.append(
                F.binary_cross_entropy_with_logits(self.termination_logits(state, refs[t]), done[t])
            )
        loss = torch.stack(state_losses).mean() * DELTA_DIM + torch.stack(term_losses).mean()
        return loss, {
            "nse1": state_losses[0].detach(),
            "nseK": state_losses[-1].detach(),
            "termination_bce": torch.stack(term_losses).mean().detach(),
        }
