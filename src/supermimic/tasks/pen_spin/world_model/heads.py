"""无额外学习奖励头：用任务奖励函数解析计算 WM 预测状态的总奖励。"""

from supermimic.core.rewards import TensorRewardManager

from ..envs.mdp.rewards import RewardInputs
from ..envs.pen_spin_env_cfg import make_env_cfg


def mimic_reward(state, reference, target, previous):
    """计算总奖励；训练代码使用 env.reward_manager 获得分项值。"""
    cfg = make_env_cfg()
    manager = TensorRewardManager(cfg.rewards, dt=0.02, scale_by_dt=cfg.scale_rewards_by_dt)
    return manager.compute(RewardInputs(state, reference, target, previous)).total
