"""Paper terminations: root (pelvis) height deviation and tilt; clip end as time-out."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from ...envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def bad_root_height(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.25) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))  # 获取运动指令
    return (cmd.root_pos_w[:, 2] - cmd.robot_root_pos_w[:, 2]).abs() > threshold  # 根部高度误差超过阈值


def bad_root_tilt(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.8) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))  # 获取运动指令
    g = cmd.robot.data.gravity_vec_w  # 世界坐标系重力向量
    ref = quat_apply_inverse(cmd.root_quat_w, g)[:, 2]  # 参考根部坐标系的重力Z分量
    rob = quat_apply_inverse(cmd.robot_root_quat_w, g)[:, 2]  # 机器人根部坐标系的重力Z分量
    return (ref - rob).abs() > threshold  # 重力Z分量差超过阈值


def clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    cmd = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))  # 获取运动指令
    return cmd.clip_end  # 返回参考动作片段结束标志