"""使用 dataclass 统一管理全部训练配置，支持 tyro 命令行参数解析。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .reward_cfg import TrackingRewardCfg

from .imagination.policy_trainer import PolicyTrainerCfg
from .policy.nets import CriticCfg, PolicyCfg
from .world_model.wm_trainer import WMTrainerCfg
from .world_model.world_model import WorldModelCfg


@dataclass # 管“仿真环境怎么运行”
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")  # NPZ 动作数据目录
    clips: list[str] | None = None  # 指定动作片段；None 表示全部
    limit: int | None = None  # 最多加载的片段数；None 不限制
    manifest: Path | None = None  # 筛选清单 CSV，包含 name、keep 字段
    library: Path | None = None  # 预打包的 .pt 动作库，优先于 npz_dir

@dataclass 
class EnvCfg:
    num_envs: int = 4096  # 并行仿真环境数量
    episode_length_s: float = 10.0  # 单回合最大时长（秒）
    action_noise_std: float = 0.1  # 动作噪声标准差
    obs_noise: bool = True  # 启用观测噪声
    reset_noise: bool = True  # 启用重置随机扰动
    domain_randomization: bool = True  # 启用域随机化
    sampling_mode: str = "return"  # 经验采样模式：按回报采样
    replay_capacity: int = 1000  # 经验回放容量
    future_frames: int = 10  # 未来参考动作帧数
    compact_library: bool = False  # 是否使用紧凑动作库
    anchor_obs: bool = False  # 是否启用锚定观测



@dataclass # 管“训练过程怎么安排”
class ScheduleCfg:
    total_iterations: int = 320_000  # 总环境交互步数（D12 方案 A）
    updates_per_env_step: float = 0.4  # UTD：每环境步的更新次数
    policy_updates_per_wm_update: int = 1  # 每次世界模型更新对应的策略更新次数
    warmup_iterations: int = 40  # 首次更新前预热 40 环境步，填充 k 步序列窗口
    lr_final_ratio: float = 0.03  # 最终学习率与初始学习率之比
    lr_decay_fraction: float = 250_000 / 320_000  # 学习率衰减阶段占总训练步数的比例
    log_interval: int = 50  # 日志记录间隔（环境步）
    checkpoint_interval: int = 4000  # 模型检查点保存间隔（环境步）
    max_hours: float | None = None  # 最长训练时间（小时）；None 不限制


@dataclass # tyro 可以根据 Python 的 dataclass 自动生成命令行参数接口。
# field() 用于配置 dataclass 中某个字段的初始化行为。
# data: DataCfg = field(default_factory=DataCfg),表示 data 的默认值需要通过调用 DataCfg() 来创建
# default=...：直接使用一个默认值。
# default_factory=...：每次创建实例时，调用指定函数生成一个新的默认值
class TrainConfig:
    run_name: str = "debug"  # 实验名称
    log_root: Path = Path("logs")  # 日志目录,相对路径
    seed: int = 0  # 随机种子
    device: str = "cuda:0"  # 计算设备
    data: DataCfg = field(default_factory=DataCfg)  # 数据配置
    rewards: TrackingRewardCfg = field(default_factory=TrackingRewardCfg)
    env: EnvCfg = field(default_factory=EnvCfg)  # 仿真环境配置
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)  # 公共网络与 G1 状态适配的组装配置
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)  # 世界模型优化配置
    policy: PolicyCfg = field(default_factory=PolicyCfg)  # 公共策略网络的维度、缩放和隐藏层配置
    critic: CriticCfg = field(default_factory=CriticCfg)  # 公共价值网络的隐藏层配置
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)  # 策略训练配置 src\supermimic\imagination\policy_trainer.py
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)  # 训练调度
    resume: Path | None = None  # 恢复训练的检查点
