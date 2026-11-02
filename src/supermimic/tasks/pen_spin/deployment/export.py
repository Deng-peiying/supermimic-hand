"""只导出 Actor、关节参考与接口说明，不导出笔真值或世界模型。"""

import json

import numpy as np
import torch

from supermimic.robots.prima1.config import MAX_TARGET_DELTA, NUM_JOINTS, RESIDUAL_LIMIT

from ..envs.mdp.observations import HISTORY, initial_history
from ..policy.reference import FUTURE_FRAMES, future_joint_reference


def export(actor, clip, env, out):
    actor = actor.cpu().eval()
    q = torch.tensor(clip.q[None, 0])
    first_end = int(clip.ends[0]) if hasattr(clip, 'clips') else env.length
    future = future_joint_reference(torch.tensor(clip.q), torch.tensor([0]), first_end, env.stage)
    scripted = torch.jit.trace(actor, (initial_history(q), future, torch.zeros(1, 1)))
    scripted.save(str(out / "actor.ts"))
    # 部署包只含 Actor 和已知参考关节，不包含世界模型及笔轨迹。
    np.savez_compressed(out / "joint_reference.npz", q=clip.q, dt=clip.dt)
    if hasattr(clip, 'clips'):
        np.savez_compressed(out / 'reference_boundaries.npz', starts=clip.starts, ends=clip.ends)
    schema = {
        **clip.provenance(),
        "history_shape": [1, HISTORY, 3 * NUM_JOINTS],
        "history_channels": [
            "q_rad[22]",
            "clip((q-q_previous)/dt/10,-5,5)[22]",
            "executed_target_rad[22]",
        ],
        "interface_version": 4,
        "future_reference_shape": [1, FUTURE_FRAMES, NUM_JOINTS],
        "future_reference_offsets": list(range(1, FUTURE_FRAMES + 1)),
        "reference_padding": "repeat last stage frame; hold repeats current frame",
        "inputs": ["history", "future_reference_q", "current_frame/stage_length"],
        "output": "22 joint targets in rad, dataset joint order",
        "stage": env.stage,
        "stage_length": env.length,
        "max_target_delta_rad": MAX_TARGET_DELTA,
        "residual_limit_rad": RESIDUAL_LIMIT,
        "hardware_ready": False,
        "needs_hardware_validation": [
            "SDK joint mapping/direction/zero",
            "position feedback timing",
            "PD gains and torque limits",
            "watchdog and emergency stop",
        ],
    }
    (out / "actor_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
