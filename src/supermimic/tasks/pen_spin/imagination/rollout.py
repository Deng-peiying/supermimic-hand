"""非对称 Actor-Critic 的可微 rollout；参考结束精确屏蔽，不自动首尾拼接。"""

from dataclasses import dataclass

import torch

from ..data.replay import history_from_start
from ..envs.mdp.observations import append_history
from ..envs.mdp.rewards import RewardInputs
from ..policy.reference import future_for_env, reference_end, reference_phase


@dataclass
class ImagRollout:
    states: torch.Tensor
    histories: torch.Tensor
    frames: torch.Tensor
    rewards: torch.Tensor
    dhat: torch.Tensor
    survival: torch.Tensor
    active: torch.Tensor


def imagine(wm, actor, env, start, horizon):
    s, frame = start.state[:, :57], start.frame.clone()
    history = history_from_start(start, env.clip.dt)
    states, histories, frames = [s], [history], [frame]
    rewards, dhats, active_masks = [], [], []
    survival = [torch.ones_like(frame, dtype=torch.float32)]
    for _ in range(horizon):
        active = (
            (frame < reference_end(env, frame)).float() if env.stage != "hold" else torch.ones_like(frame).float()
        )
        nxt = frame if env.stage == "hold" else torch.minimum(frame + 1, torch.as_tensor(reference_end(env, frame), device=frame.device))
        # 参考已经结束的样本不再向前预测。仅乘0不能消除NaN/Inf，
        # 也不应让终止后的任意状态进入Critic输入。
        live = active.bool()
        action = history[:, -1, 44:66].clone()
        prediction = s.clone()
        if live.any():
            action[live] = actor(
                history[live],
                future_for_env(env, frame[live]),
                reference_phase(env, frame[live]),
            )
            prediction[live] = wm(s[live], action[live], start.xi[live])
        reward = env.reward_manager.compute(
            RewardInputs(prediction, env.ref[nxt], action, history[:, -1, 44:66])
        ).total
        done = torch.sigmoid(wm.termination_logits(prediction, env.ref[nxt]))
        if env.stage != "hold":
            done = torch.where(nxt >= reference_end(env, frame), torch.ones_like(done), done)
        rewards.append(reward * active)
        dhats.append(done)
        active_masks.append(active)
        survival.append(survival[-1] * (1 - done))
        updated_history = append_history(
            history, prediction[:, :22], s[:, :22], action, env.clip.dt
        )
        history = torch.where(live[:, None, None], updated_history, history)
        s, frame = prediction, nxt
        states.append(s)
        histories.append(history)
        frames.append(frame)
    return ImagRollout(
        *map(torch.stack, (states, histories, frames, rewards, dhats, survival, active_masks))
    )
