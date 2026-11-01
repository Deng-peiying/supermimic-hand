"""奖励原始值：真实仿真评估与 WM 想象共用，不在公式里混入权重。"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from supermimic.core.state import RewardInputs as BaseRewardInputs
from supermimic.core.state import StateLayout
from supermimic.robots.prima1.config import STATE_LAYOUT


@dataclass
class RewardInputs(BaseRewardInputs):
    """Prima1 defaults; other embodiments pass an explicit layout to core.RewardInputs."""

    layout: StateLayout = STATE_LAYOUT


def pen_axis(quat: torch.Tensor) -> torch.Tensor:
    """笔局部 X 轴转到掌心基座系；四元数为 wxyz。"""
    w, x, y, z = F.normalize(quat, dim=-1).unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), -1)


def joint_position_tracking_exp(data: RewardInputs, std: float) -> torch.Tensor:
    """关节位置软跟踪；std 单位 rad，先对 22 关节平方误差取均值。"""
    _positive(std)
    error = (data.current("joint_pos") - data.desired("joint_pos")).square().mean(-1)
    return torch.exp(-error / std**2)


def joint_velocity_tracking_exp(data: RewardInputs, std: float) -> torch.Tensor:
    """关节速度软跟踪；std 单位 rad/s。"""
    _positive(std)
    error = (data.current("joint_vel") - data.desired("joint_vel")).square().mean(-1)
    return torch.exp(-error / std**2)


def pen_position_error_l2(data: RewardInputs) -> torch.Tensor:
    """笔中心距离平方（m²）；返回正代价，由负权重产生惩罚。"""
    return (data.current("object_pos") - data.desired("object_pos")).square().sum(-1)


def pen_position_tracking_exp(data: RewardInputs, std: float) -> torch.Tensor:
    """笔中心跟踪；std 单位 m。"""
    _positive(std)
    return torch.exp(-pen_position_error_l2(data) / std**2)


def pen_axis_tracking_exp(data: RewardInputs, error_scale: float) -> torch.Tensor:
    """无标记笔两端等价：误差 1-(轴点积)²，忽略绕自身长轴旋转。

    error_scale 是无量纲分母，不是弧度标准差，不能再平方。
    """
    _positive(error_scale)
    dot = (pen_axis(data.current("object_quat")) * pen_axis(data.desired("object_quat"))).sum(-1)
    error = (1 - dot.square()).clamp(0, 1)
    return torch.exp(-error / error_scale)


def action_delta_l2(data: RewardInputs, scale: float) -> torch.Tensor:
    """相邻执行目标归一化平方差均值；scale 单位 rad/控制步。"""
    _positive(scale)
    return ((data.target - data.previous_target) / scale).square().mean(-1)


def _positive(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Reward error scales must be positive and finite")
