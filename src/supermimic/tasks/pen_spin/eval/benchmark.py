"""单片段确定性物理评估，保留分项奖励与失败定义。"""

import json
from pathlib import Path

import numpy as np
import torch

from supermimic.robots.prima1.config import STATE_LAYOUT

from ..data.reference import references, tensor
from ..envs.mdp.observations import append_history, initial_history
from ..envs.mdp.rewards import RewardInputs, pen_axis
from ..policy.reference import future_joint_reference


@torch.no_grad()
def evaluate(env, actor, device, output=None, baseline=False):
    state = env.reset()
    history = initial_history(tensor(state[None, STATE_LAYOUT.index("joint_pos")], device))
    ref = references(env.clip, env.stage, device)
    rewards, errors, angles, states, targets = [], [], [], [state.copy()], []
    raw_terms = {name: [] for name in env.cfg.rewards}
    weighted_terms = {name: [] for name in env.cfg.rewards}
    done, info = False, {}
    while not done:
        frame = env.frame
        nxt = frame if env.stage == "hold" else frame + 1
        previous = env.previous.copy()
        if baseline:
            action = env.clip.q[nxt]
        else:
            action = (
                actor(
                    history,
                    future_joint_reference(
                        ref[:, :22], torch.tensor([frame], device=device), env.length, env.stage
                    ),
                    tensor([[frame / env.length]], device),
                )[0]
                .cpu()
                .numpy()
            )
        next_state, done, info = env.step(action)
        s = tensor(next_state[None], device)
        result = env.reward_manager.compute(
            RewardInputs(
                s,
                ref[None, nxt],
                tensor(env.previous[None], device),
                tensor(previous[None], device),
            )
        )
        rewards.append(float(result.total))
        for name, values in raw_terms.items():
            values.append(float(result.raw[name]))
            weighted_terms[name].append(float(result.weighted[name]))
        errors.append(info["position_error_m"])
        dot = (
            (
                pen_axis(s[:, STATE_LAYOUT.index("object_quat")])
                * pen_axis(ref[None, nxt, STATE_LAYOUT.index("object_quat")])
            )
            .sum()
            .abs()
            .clamp(0, 1)
        )
        angles.append(float(torch.acos(dot).cpu()) * 180 / np.pi)
        history = append_history(
            history,
            s[:, STATE_LAYOUT.index("joint_pos")],
            tensor(state[None, STATE_LAYOUT.index("joint_pos")], device),
            tensor(env.previous[None], device),
            env.clip.dt,
        )
        state = next_state
        states.append(state.copy())
        targets.append(env.previous.copy())
    metrics = {
        "steps": len(rewards),
        "duration_s": len(rewards) * env.clip.dt,
        "dropped": info["dropped"],
        "completed_without_drop": not info["dropped"] and info["truncated"],
        "tracking_failed": info["tracking_failed"],
        "completion_criterion": "time_limit_without_tracking_distance_failure",
        "physical_drop_detected": None,  # No contact-based drop detector yet.
        "spin_success": None,  # Requires a separately validated rotation criterion.
        "mean_position_error_m": float(np.mean(errors)),
        "max_position_error_m": max(errors),
        "mean_axis_error_deg": float(np.mean(angles)),
        "mean_reward": float(np.mean(rewards)),
        "return": float(np.sum(rewards)),
        "reward_raw_mean": {name: float(np.mean(values)) for name, values in raw_terms.items()},
        "reward_weighted_mean": {
            name: float(np.mean(values)) for name, values in weighted_terms.items()
        },
        "reward_term_return": {
            name: float(np.sum(values)) for name, values in weighted_terms.items()
        },
        "baseline": baseline,
        "start_frame": 0,
        "initial_pen_velocity": "zero",
        "stage": env.stage,
    }
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / "task_config.json").write_text(
            json.dumps(env.cfg.describe(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            output.with_suffix(".npz"),
            states=np.asarray(states),
            targets=np.asarray(targets),
            dt=env.clip.dt,
        )
        output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
