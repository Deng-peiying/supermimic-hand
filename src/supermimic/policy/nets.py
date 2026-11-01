"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import OBS_TERMS

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


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
    hidden: tuple[int, ...] = (256, 128) # 两层隐藏层维度


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor: # 生成观测缩放向量
    parts = [] # 存储各观测项的缩放向量
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale): # 遍历观测维度与缩放系数
        parts.append(torch.full((cfg.history * d,), 1.0 / s)) # 为每项历史观测生成缩放系数
    return torch.cat(parts) # 按特征项拼接，与展平后的历史观测顺序一致


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor: # 生成目标缩放向量
    sizes = (29, 29, 3, 3, 3, 1) # 单帧目标特征维度
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)]) # 生成68维缩放向量
    k, extra = divmod(goal_dim, 68) # 计算完整参考帧数及额外维度
    return torch.cat([per_frame.repeat(k), torch.ones(extra)]) # 拼接各帧及额外特征的缩放系数


class MotionEncoder(nn.Module): # 参考运动编码器
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__() # 初始化父类
        goal_dim = goal_dim or cfg.goal_dim # 确定目标输入维度
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim) # 创建目标编码网络
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim)) # 注册目标缩放向量

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale)) # 缩放目标并编码为潜在特征


class Decoder(nn.Module):  # 962维->29维
    def __init__(self, cfg: PolicyCfg, obs_dim: int = 930):
        super().__init__()
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, ACT_DIM, last_gain=cfg.decoder_last_gain) # 创建动作解码网络
        self.register_buffer("scale", _obs_scale_vector(cfg)) # 注册历史观测缩放向量

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1)) # 拼接潜在特征与历史观测，输出动作


class Policy(nn.Module):
    """动作 = 解码器(运动编码器(目标), 历史观测)；后续可注册其他目标接口（附录B方案）。"""

    def __init__(self, cfg: PolicyCfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg) # 创建共享动作解码器

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        return self.decoder(self.encoders[interface](goal), obs_hist)


class Critic(nn.Module):
    """价值函数：71维标准状态、680维目标、33维物理参数和4维根部误差，输出标量；每种接口独立配置。"""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(-1)
"""
Canonical State	71	机器人完整物理状态
[
    oot_pos      	3	0–2	骨盆三维位置 \(x,y,z\)
    root_quat	    4	3–6	骨盆姿态四元数
    root_lin_vel	3	7–9	骨盆三轴线速度
    root_ang_vel	3	10–12	骨盆三轴角速度
    joint_pos	    29	13–41	29个关节角度
    joint_vel	    29	42–70	29个关节角速度
    合计	        71	  0–70
]


obs["policy"]	930	最近10帧本体观测
[
    joint_pos	    29	290	相对默认姿态的关节位置
    joint_vel	    29	290	关节速度
    base_ang_vel	3	30	基座角速度
    projected_gravity	3	30	基座坐标系下的重力方向
    actions	29	290	历史关节动作
]

obs["goal"]	680	未来10帧参考运动
[
    joint_pos	29	290	目标关节位置
    joint_vel	29	290	目标关节速度
    grav	3	30	参考姿态下的重力方向
    ang_vel_b	3	30	参考机体系角速度
    lin_vel_h	3	30	参考水平坐标系线速度
    height	1	10	参考根部高度
]

MotionEncoder 输出	32	目标运动潜在特征
Decoder 输入	962	32维潜在特征 + 930维历史观测
Policy 输出	29	29个关节动作
xi	33	域随机化物理参数
[
friction	    1	足部摩擦参数
encoder_bias	29	29个关节的编码器偏差
com_offset	    3	躯干质心在X/Y/Z方向的偏移
]

e_root	    4	根部跟踪误差特征
Critic 输入	788	71 + 680 + 33 + 4
Critic 输出	1	状态价值估计



"""
