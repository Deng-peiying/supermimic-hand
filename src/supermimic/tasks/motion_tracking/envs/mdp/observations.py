from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from ...envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """返回未来K帧参考运动特征，形状为(N, K×68)，D5方案。"""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))  # 获取运动指令并标注类型
    return command.goal  # 返回未来参考运动特征
