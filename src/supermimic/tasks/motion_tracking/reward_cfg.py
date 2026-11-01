"""Single reward parameter source for simulation and imagination."""

from dataclasses import dataclass


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3  # 身体位置误差尺度
    std_body_ori: float = 0.4  # 身体姿态误差尺度
    std_body_lin_vel: float = 1.0  # 身体线速度误差尺度
    std_body_ang_vel: float = 3.14  # 身体角速度误差尺度
    std_root_pos: float = 0.3  # 根部位置误差尺度
    std_root_ori: float = 0.4  # 根部姿态误差尺度
    w_body_pos: float = 1.0  # 身体位置奖励权重
    w_body_ori: float = 1.0  # 身体姿态奖励权重
    w_body_lin_vel: float = 1.0  # 身体线速度奖励权重
    w_body_ang_vel: float = 1.0  # 身体角速度奖励权重
    w_root_pos: float = 0.5  # 根部位置奖励权重
    w_root_ori: float = 0.5  # 根部姿态奖励权重
    w_action_rate: float = -0.1  # 动作变化惩罚权重
    w_joint_limit: float = -2.0  # 关节限位惩罚权重
    dt: float = 0.02  # 控制时间步长

    @property  # 将方法作为只读属性访问
    def max_step_reward(self) -> float:
        return self.dt * (self.w_body_pos + self.w_body_ori + self.w_body_lin_vel + self.w_body_ang_vel + self.w_root_pos + self.w_root_ori)  # 单步最大正奖励

