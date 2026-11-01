"""策略驱动的真实接触采集；保存实际执行动作，禁止跨 reset 拼接转移。"""

import numpy as np
import torch

from supermimic.robots.prima1.config import NUM_JOINTS, STATE_LAYOUT

from ..envs.mdp.observations import append_history, initial_history
from ..policy.reference import future_joint_reference
from .reference import tensor


@torch.no_grad()
def collect(env, actor, steps, device, rng, noise=0.04, random_start=False):
    records = []
    state = env.reset()
    history = initial_history(tensor(state[None, STATE_LAYOUT.index("joint_pos")], device))
    reference_q = tensor(env.clip.q, device)
    for _ in range(steps):
        frame = env.frame
        phase = tensor([[frame / env.length]], device)
        action = (
            actor(
                history,
                future_joint_reference(
                    reference_q, torch.tensor([frame], device=device), env.length, env.stage
                ),
                phase,
            )[0]
            .cpu()
            .numpy()
        )
        next_state, done, _ = env.step(action + rng.normal(0, noise, NUM_JOINTS))
        if (
            next_state[STATE_LAYOUT.index("object_quat")] @ state[STATE_LAYOUT.index("object_quat")]
            < 0
        ):
            next_state[STATE_LAYOUT.index("object_quat")] *= -1
        # terminal 转移保留；reset 后的状态绝不作为它的 next_state。
        records.append(
            (
                state.copy(),
                env.previous.copy(),
                next_state.copy(),
                history[0].cpu().numpy().copy(),
                frame,
                done,
            )
        )
        history = append_history(
            history,
            tensor(next_state[None, STATE_LAYOUT.index("joint_pos")], device),
            tensor(state[None, STATE_LAYOUT.index("joint_pos")], device),
            tensor(env.previous[None], device),
            env.clip.dt,
        )
        state = next_state
        if done:
            start = int(rng.integers(env.length)) if random_start and env.stage != "hold" else 0
            state = env.reset(start)
            history = initial_history(tensor(state[None, STATE_LAYOUT.index("joint_pos")], device))
    return records


def batch(records, count, device, rng):
    indices = rng.integers(len(records), size=count)
    cols = list(zip(*(records[int(i)] for i in indices)))
    return [tensor(np.stack(c), device) for c in cols]
