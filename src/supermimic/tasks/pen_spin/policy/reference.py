"""已知手关节未来参考；不读取笔状态。末端重复末帧，不循环拼接。"""

import torch

FUTURE_FRAMES = 10


def future_joint_reference(reference_q, frame, length, stage, ends=None):
    """返回 (..., 10, 22)，从当前帧的下一帧开始；hold 固定当前帧。"""
    offsets = torch.arange(1, FUTURE_FRAMES + 1, device=frame.device)
    indices = frame[..., None] + (0 * offsets if stage == "hold" else offsets)
    indices = indices.clamp(max=min(length, len(reference_q) - 1))
    if ends is not None:
        indices = torch.minimum(indices, ends[frame][..., None])
    return reference_q[indices, :22]


def reference_end(env, frame):
    return env.reference_ends[frame] if hasattr(env, 'reference_ends') else env.length


def reference_phase(env, frame):
    if hasattr(env, 'reference_starts'):
        start, end = env.reference_starts[frame], env.reference_ends[frame]
        return ((frame - start).float() / (end - start).clamp_min(1))[..., None]
    return frame[..., None].float() / env.length


def future_for_env(env, frame):
    return future_joint_reference(env.ref[:, :22], frame, env.length, env.stage,
                                  getattr(env, 'reference_ends', None))
