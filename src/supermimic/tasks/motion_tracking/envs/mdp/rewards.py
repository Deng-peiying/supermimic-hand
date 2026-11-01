"""
基于 PyTorch 的 BeyondMimic 解析式跟踪奖励（D4/D7方案）。
同一套可微实现用于仿真侧（测试、日志）和世界模型想象侧（奖励计算）。
逐项对应 mjlab 的 tasks/tracking/mdp/rewards.py，包括参考身体的重新锚定：
仅对齐机器人锚点的 XY 位置和偏航角（Yaw），详见 MotionCommand.update_relative_body_poses。

输入为跟踪身体部位的世界坐标状态，顺序与 TRACKED_BODIES 一致，
锚点是其中一个身体部位（anchor_index）。
奖励已乘以各项权重和时间步长 dt，与 mjlab 的奖励管理器一致（scale_rewards_by_dt=True）。
"""

from __future__ import annotations

from ...reward_cfg import TrackingRewardCfg

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_box_minus, quat_inv, quat_mul, yaw_quat




def reanchor_reference(
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_anchor_pos_w: torch.Tensor,
    ref_anchor_quat_w: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将参考身体对齐至机器人锚点的XY位置和偏航角。与 mjlab 的 update_relative_body_poses 一致：
    Z高度、横滚角和俯仰角保留参考运动的值。形状：身体(N,B,·)，锚点(N,·)。
    """
    nb = ref_body_pos_w.shape[-2]  # 参考身体数量
    delta_pos = robot_anchor_pos_w.clone()  # 复制机器人锚点位置
    delta_pos = torch.cat([delta_pos[..., :2], ref_anchor_pos_w[..., 2:3]], dim=-1)  # 机器人XY与参考Z组合
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(ref_anchor_quat_w)))  # 计算锚点偏航旋转差
    delta_ori_b = delta_ori[..., None, :].expand(*delta_ori.shape[:-1], nb, 4)  # 扩展到所有身体部位
    rel_quat = quat_mul(delta_ori_b, ref_body_quat_w)  # 旋转参考身体姿态
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori_b, ref_body_pos_w - ref_anchor_pos_w[..., None, :])  # 对齐参考身体位置
    return rel_pos, rel_quat  # 返回重新锚定的位姿


def _sq_quat_err(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """计算四元数之间的测地旋转角平方，零误差处平滑。"""
    v = quat_box_minus(q1, q2)  # 计算旋转误差向量
    return (v * v).sum(-1)  # 返回旋转误差平方



def tracking_reward_terms(
    *,
    robot_body_pos_w: torch.Tensor,
    robot_body_quat_w: torch.Tensor,
    robot_body_lin_vel_w: torch.Tensor,
    robot_body_ang_vel_w: torch.Tensor,
    ref_body_pos_w: torch.Tensor,
    ref_body_quat_w: torch.Tensor,
    ref_body_lin_vel_w: torch.Tensor,
    ref_body_ang_vel_w: torch.Tensor,
    anchor_index: int,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_limits: torch.Tensor,  # (29, 2) or (N, 29, 2)
    cfg: TrackingRewardCfg,
) -> dict[str, torch.Tensor]:
    """返回各项加权并乘以dt的奖励(N,)，以及总奖励total。"""
    ra_pos = robot_body_pos_w[..., anchor_index, :]  # 机器人锚点位置
    ra_quat = robot_body_quat_w[..., anchor_index, :]  # 机器人锚点姿态
    ma_pos = ref_body_pos_w[..., anchor_index, :]  # 参考锚点位置
    ma_quat = ref_body_quat_w[..., anchor_index, :]  # 参考锚点姿态
    rel_pos, rel_quat = reanchor_reference(ref_body_pos_w, ref_body_quat_w, ma_pos, ma_quat, ra_pos, ra_quat)  # 对齐参考身体

    e_root_pos = ((ma_pos - ra_pos) ** 2).sum(-1)  # 锚点位置平方误差
    e_root_ori = _sq_quat_err(ma_quat, ra_quat)  # 锚点姿态平方误差
    e_body_pos = ((rel_pos - robot_body_pos_w) ** 2).sum(-1).mean(-1)  # 平均身体位置平方误差
    e_body_ori = _sq_quat_err(rel_quat, robot_body_quat_w).mean(-1)  # 平均身体姿态平方误差
    e_body_lin = ((ref_body_lin_vel_w - robot_body_lin_vel_w) ** 2).sum(-1).mean(-1)  # 平均线速度平方误差
    e_body_ang = ((ref_body_ang_vel_w - robot_body_ang_vel_w) ** 2).sum(-1).mean(-1)  # 平均角速度平方误差


    lo = soft_limits[..., 0]  # 关节软下限
    hi = soft_limits[..., 1]  # 关节软上限
    out = (-(joint_pos - lo)).clamp(min=0.0) + (joint_pos - hi).clamp(min=0.0)  # 关节越界幅度
    joint_limit = out.sum(-1)  # 汇总关节限位惩罚量
    action_rate = ((action - prev_action) ** 2).sum(-1)  # 相邻动作平方差

    dt = cfg.dt
    terms = {
        "motion_global_root_pos": dt * cfg.w_root_pos * torch.exp(-e_root_pos / cfg.std_root_pos**2),  # 全局锚点位置奖励
        "motion_global_root_ori": dt * cfg.w_root_ori * torch.exp(-e_root_ori / cfg.std_root_ori**2),  # 全局锚点姿态奖励
        "motion_body_pos": dt * cfg.w_body_pos * torch.exp(-e_body_pos / cfg.std_body_pos**2),  # 身体位置奖励
        "motion_body_ori": dt * cfg.w_body_ori * torch.exp(-e_body_ori / cfg.std_body_ori**2),  # 身体姿态奖励
        "motion_body_lin_vel": dt * cfg.w_body_lin_vel * torch.exp(-e_body_lin / cfg.std_body_lin_vel**2),  # 身体线速度奖励
        "motion_body_ang_vel": dt * cfg.w_body_ang_vel * torch.exp(-e_body_ang / cfg.std_body_ang_vel**2),  # 身体角速度奖励
        "action_rate_l2": dt * cfg.w_action_rate * action_rate,  # 动作变化惩罚
        "joint_limit": dt * cfg.w_joint_limit * joint_limit,  # 关节限位惩罚
    }
    total = terms["motion_global_root_pos"]  # 初始化总奖励
    for k, v in terms.items():  # 遍历各项奖励
        if k != "motion_global_root_pos":  # 跳过已计入项
            total = total + v  # 累加奖励
    terms["total"] = total  # 保存总奖励
    return terms  # 返回奖励字典
