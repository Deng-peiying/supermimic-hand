"""Actor 的接口没有任何物体状态；世界模型与奖励允许使用仿真真值。"""

import torch
import torch.nn.functional as F
from torch import nn

HISTORY = 30  # 0.6 秒；速度由连续两次关节位置差分得到。
STATE_SCALE = [1.0] * 22 + [10.0] * 22 + [0.1] * 3 + [1.0] * 4 + [1.0] * 3 + [10.0] * 3


def append_history(history, q, previous_q, executed_target, dt=0.02):
    sample = torch.cat((q, ((q - previous_q) / dt / 10).clamp(-5, 5), executed_target), -1)
    return torch.cat((history[:, 1:], sample[:, None]), 1)


def initial_history(q):
    # 填充静止启动历史，不用未来参考或仿真 qvel 伪造传感器历史。
    return torch.cat((q, torch.zeros_like(q), q), -1)[:, None].repeat(1, HISTORY, 1)


class Actor(nn.Module):
    def __init__(self, limits):
        super().__init__()
        self.register_buffer("limits", torch.as_tensor(limits, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(HISTORY * 66 + 23, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 22),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, history, reference_q, phase):
        # reference_q/phase 来自内部固定片段时钟，绝不是笔的测量信息。
        residual = 0.15 * torch.tanh(
            self.net(torch.cat((history.flatten(1), reference_q, phase), -1))
        )
        previous = history[:, -1, 44:66]
        target = torch.clamp(reference_q + residual, previous - 0.12, previous + 0.12)
        return torch.clamp(target, self.limits[:, 0], self.limits[:, 1])


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(STATE_SCALE))
        self.net = nn.Sequential(
            nn.Linear(79, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 57)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, action):
        # 学习归一化状态增量；特权笔状态只在这里出现。
        result = state + self.scale * self.net(torch.cat((state / self.scale, action), -1))
        quat = F.normalize(result[..., 47:51], dim=-1, eps=1e-6)
        return torch.cat((result[..., :47], quat, result[..., 51:]), -1)

    def loss(self, state, action, next_state):
        prediction = self(state, action)
        # 采样时也连续化了符号；这里处理任意 q/-q 等价表示。
        sign = torch.where(
            (prediction[..., 47:51] * next_state[..., 47:51]).sum(-1, keepdim=True) < 0, -1.0, 1.0
        )
        target = torch.cat(
            (next_state[..., :47], next_state[..., 47:51] * sign, next_state[..., 51:]), -1
        )
        return ((prediction - target) / self.scale).square().mean()


def pen_axis(quat):
    w, x, y, z = F.normalize(quat, dim=-1).unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), -1)


def survival_probability(state, reference):
    """真实环境 10 cm 失效阈值的平滑近似，仅用于想象中的可微终止。"""
    distance = (state[..., 44:47] - reference[..., 44:47]).norm(dim=-1)
    return torch.sigmoid((0.10 - distance) / 0.005)


def mimic_reward(state, reference, target, previous):
    """圆柱笔忽略绕自身轴旋转；位置和轴方向是主奖励，关节跟踪为软约束。"""
    q_error = (state[..., :22] - reference[..., :22]).square().mean(-1)
    qd_error = (state[..., 22:44] - reference[..., 22:44]).square().mean(-1)
    p_error = (state[..., 44:47] - reference[..., 44:47]).square().sum(-1)
    dot = (pen_axis(state[..., 47:51]) * pen_axis(reference[..., 47:51])).sum(-1)
    axis_error = (1 - dot.square()).clamp(0, 1)  # 无标记圆柱两端对称。
    reward = (
        torch.exp(-q_error / 0.15**2)
        + 0.2 * torch.exp(-qd_error / 3**2)
        + 3 * torch.exp(-p_error / 0.025**2)
        + 2 * torch.exp(-axis_error / 0.2)
    )
    # 平滑距离惩罚在掉笔时仍有梯度，防止仅靠接近零的指数奖励。
    reward = reward - 20 * p_error - 0.02 * ((target - previous) / 0.12).square().mean(-1)
    return reward
