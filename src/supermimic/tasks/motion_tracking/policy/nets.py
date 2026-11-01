"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.algorithms.networks import GoalConditionedPolicy, ValueNetwork
from supermimic.robots.g1.state import NUM_JOINTS
from ..world_model.heads import OBS_TERMS

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = NUM_JOINTS


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)  # 编码器隐藏层维度
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)  # 解码器隐藏层维度
    latent_dim: int = LATENT_DIM  # 潜在表示维度
    history: int = 10  # 历史观测步数
    # 观测特征缩放除数：[关节位置、关节速度、基座角速度、投影重力、历史动作]
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # 目标特征缩放除数：[关节位置29维、关节速度29维、重力3维、机体系角速度3维、水平线速度3维、高度1维]
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01  # 解码器末层初始化增益
    goal_dim: int = GOAL_DIM  # 目标特征维度：680，启用锚定观测后增加9维


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)  # 两层隐藏层维度


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:  # 生成观测缩放向量
    parts = []  # 存储各观测项的缩放向量
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):  # 遍历观测维度与缩放系数
        parts.append(torch.full((cfg.history * d,), 1.0 / s))  # 为每项历史观测生成缩放系数
    return torch.cat(parts)  # 按特征项拼接，与展平后的历史观测顺序一致


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:  # 生成目标缩放向量
    sizes = (NUM_JOINTS, NUM_JOINTS, 3, 3, 3, 1)  # 单帧目标特征维度
    per_frame = torch.cat(
        [torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)]
    )  # 生成68维缩放向量
    k, extra = divmod(goal_dim, 68)  # 计算完整参考帧数及额外维度
    return torch.cat([per_frame.repeat(k), torch.ones(extra)])  # 拼接各帧及额外特征的缩放系数


class Policy(GoalConditionedPolicy):
    """Assemble shared networks with the tracking task's observation/goal scaling."""

    def __init__(self, cfg: PolicyCfg):
        super().__init__(
            goal_scale=_goal_scale_vector(cfg, cfg.goal_dim),
            obs_scale=_obs_scale_vector(cfg),
            encoder_hidden=cfg.encoder_hidden,
            decoder_hidden=cfg.decoder_hidden,
            latent_dim=cfg.latent_dim,
            action_dim=ACT_DIM,
            last_gain=cfg.decoder_last_gain,
        )
        self.cfg = cfg


class Critic(ValueNetwork):
    """Assemble a shared value network with privileged tracking error features."""

    def __init__(self, cfg: CriticCfg, feat_dim=71, goal_dim=None, xi_dim=33, pcfg=None):
        pcfg = pcfg or PolicyCfg()
        super().__init__(
            feat_dim=feat_dim,
            goal_scale=_goal_scale_vector(pcfg, goal_dim or pcfg.goal_dim),
            context_dim=xi_dim,
            error_dim=4,
            hidden=cfg.hidden,
        )
