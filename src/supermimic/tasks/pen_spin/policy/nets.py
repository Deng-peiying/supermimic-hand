"""任务策略组装：公共残差 Actor + Prima1 动作约束。"""

import torch
from torch import nn

from supermimic.algorithms.models import ResidualActor
from supermimic.algorithms.networks import MLP
from supermimic.robots.prima1.config import MAX_TARGET_DELTA, RESIDUAL_LIMIT

from ..envs.mdp.observations import HISTORY
from ..world_model.state_spec import CONTEXT_DIM, FEAT_DIM, canonical
from .reference import FUTURE_FRAMES


class Actor(ResidualActor):
    def __init__(self, limits):
        super().__init__(
            limits,
            history_steps=HISTORY,
            residual_limit=RESIDUAL_LIMIT,
            max_target_delta=MAX_TARGET_DELTA,
        )
        self.net[0] = nn.Linear(HISTORY * 66 + FUTURE_FRAMES * 22 + 1, 128)

    def forward(self, history, reference_q, phase):
        # 整个未来窗口用于决策；残差仍加在下一帧关节参考上。
        residual = self.residual_limit * torch.tanh(
            self.net(torch.cat((history.flatten(1), reference_q.flatten(1), phase), -1))
        )
        previous = history[:, -1, 44:66]
        target = torch.clamp(
            reference_q[:, 0] + residual,
            previous - self.max_target_delta,
            previous + self.max_target_delta,
        )
        return torch.clamp(target, self.limits[:, 0], self.limits[:, 1])


class Critic(nn.Module):
    """非对称价值网络：笔真值 + Actor 历史；不参与真机推理。"""

    def __init__(self, hidden=(256, 256, 256)):
        super().__init__()
        self.net = MLP(FEAT_DIM * 2 + HISTORY * 66 + 1 + CONTEXT_DIM, hidden, 1, zero_last=True)

    def forward(self, state, history, reference, phase, xi):
        x = torch.cat(
            (canonical(state), history.flatten(1), canonical(reference), phase, (xi - 1) / 0.2), -1
        )
        return self.net(x).squeeze(-1)
