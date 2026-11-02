"""六项基线奖励 + 翻转角速度跟踪；无父任务奖励继承。"""

from supermimic.core.rewards import TensorRewardTermCfg

from .envs.mdp import rewards as mdp


def pen_spin_rewards() -> dict[str, TensorRewardTermCfg]:
    # 六项基线参数保持不变；第七项提供翻转方向与速度目标。
    terms = {
        "joint_position": TensorRewardTermCfg(
            func=mdp.joint_position_tracking_exp, weight=1.0, params={"std": 0.15}
        ),
        "joint_velocity": TensorRewardTermCfg(
            func=mdp.joint_velocity_tracking_exp, weight=0.2, params={"std": 3.0}
        ),
        "pen_position": TensorRewardTermCfg(
            func=mdp.pen_position_tracking_exp, weight=3.0, params={"std": 0.025}
        ),
        "pen_axis": TensorRewardTermCfg(
            func=mdp.pen_axis_tracking_exp, weight=2.0, params={"error_scale": 0.2}
        ),
        "pen_distance": TensorRewardTermCfg(func=mdp.pen_position_error_l2, weight=-20.0),
        "action_delta": TensorRewardTermCfg(
            func=mdp.action_delta_l2, weight=-0.02, params={"scale": 0.12}
        ),
        "pen_flip_velocity": TensorRewardTermCfg(
            func=mdp.pen_flip_velocity_tracking, weight=0.5, params={"std": 3.0}
        ),
    }

    # 持久化身份保留迁移前的值，目录调整不等于奖励定义发生变化。
    # 这只是检查点元数据，不用于动态导入；实际函数位于 envs/mdp/rewards.py。
    for term in terms.values():
        term.identifier = f"supermimic.tasks.pen_spin.mdp.rewards.{term.func.__name__}"
    return terms
