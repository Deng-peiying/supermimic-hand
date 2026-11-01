"""将单步想象计算封装为编译函数（方案3.6：降低内核启动开销）。
ImagStep 将世界模型动力学、解析正向运动学奖励、终止预测和下一步无噪声观测整合到一次调用中，使 torch.compile 能融合
逐刚体正向运动学计算中的大量小内核。参考运动数据预先从动作库提取，以普通张量传入，确保编译图形状固定。
"""

from __future__ import annotations

import torch

from supermimic.world_model.heads import ReferenceFrame, analytic_observation
from supermimic.world_model.world_model import WorldModel

REWARD_TERM_NAMES = (
    "motion_global_root_pos",  # 全局位置跟踪奖励
    "motion_global_root_ori",  # 全局姿态跟踪奖励
    "motion_body_pos",  # 身体相对位置跟踪奖励
    "motion_body_ori",  # 身体相对姿态跟踪奖励
    "motion_body_lin_vel",  # 身体线速度跟踪奖励
    "motion_body_ang_vel",  # 身体角速度跟踪奖励
    "action_rate_l2",  # 动作变化惩罚
    "joint_limit",  # 关节限位惩罚
)


class ImagStep:
    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, compile_mode: str | None = None): # 初始化单步想象计算器
        self.wm = wm # 保存世界模型
        self.q0 = default_joint_pos # 保存默认关节位置
        self._fn = self._step # 默认使用原始单步计算函数
        if compile_mode: # 启用编译
            mode = None if compile_mode == "default" else compile_mode # 选择编译模式
            self._fn = torch.compile(self._step, mode=mode, dynamic=False) # 编译单步计算函数

    def _step(self, s, a, a_prev, xi, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat):
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat)  # 构建参考运动帧
        s_next = self.wm.step(s, a, xi) # 世界模型预测下一状态
        bodies = self.wm.heads.robot_bodies(s_next) # 正向运动学计算各刚体状态
        terms = self.wm.heads.reward(s_next, a, a_prev, ref, bodies) # 计算各项奖励
        r = terms["total"] # 获取总奖励
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0) # 堆叠8项奖励，形状(8,n)
        logits = self.wm.termination_logits(s_next, ref) # 预测终止信号
        obs_clean = analytic_observation(s_next, a, xi, self.q0)  # 生成无噪声观测
        ai = self.wm.heads.anchor_index # 获取锚点身体索引
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)  # 拼接锚点位置和姿态，形状(n,7)
        return s_next, r, term_vec, logits, obs_clean, anchor_pose  # 返回单步想象结果

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame):
        return self._fn(s, a, a_prev, xi, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w)
