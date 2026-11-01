"""阶段、硬终止、软存活与奖励组装。"""

from dataclasses import dataclass, field

from supermimic.core.rewards import TensorRewardTermCfg

from ..reward_cfg import pen_spin_rewards


@dataclass
class PenSpinEnvCfg:
    stage: str = "short"
    rewards: dict[str, TensorRewardTermCfg] = field(default_factory=pen_spin_rewards)
    # 与 G1/Dex 一致：权重后的奖励乘控制周期；真实与想象共用管理器。
    scale_rewards_by_dt: bool = True
    drop_distance_m: float = 0.10
    survival_softness_m: float = 0.005
    hold_steps: int = 100
    short_steps: int = 50

    def __post_init__(self):
        if self.stage not in ("hold", "short", "full"):
            raise ValueError(f"未知阶段: {self.stage}")
        if self.drop_distance_m <= 0 or self.survival_softness_m <= 0:
            raise ValueError("终止距离和软化尺度必须为正")

    def describe(self):
        return {
            "task": "prima1_left_pen_spin",
            "inherits_rewards_from": [],
            "stage": self.stage,
            "scale_rewards_by_dt": self.scale_rewards_by_dt,
            "rewards": {name: term.describe() for name, term in self.rewards.items()},
            "terminations": {
                "drop_distance_m": self.drop_distance_m,
                "hold_steps": self.hold_steps,
                "short_steps": self.short_steps,
                "full_steps": "number of reference frames - 1",
            },
            "imagination": {
                "termination": "learned_distance_error_head",
                "termination_feature_scale_m": self.drop_distance_m,
                "reference_end": "stop_bootstrap_and_freeze_state",
            },
        }


def make_env_cfg(stage="short") -> PenSpinEnvCfg:
    return PenSpinEnvCfg(stage=stage)
