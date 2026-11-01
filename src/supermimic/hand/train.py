"""单片段 HandMimic：真实接触采集 → 世界模型监督学习 → 短窗口 BPTT。

首轮实现采用截断有限时域目标，没有 G1 的 critic bootstrap；不会把 PPO
冒充 SuperMimic，也不会对 MuJoCo 接触求导。学习梯度只经过拟合的世界模型。
"""

import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .data import DEFAULT_ASSET, DEFAULT_CLIP, Clip
from .env import HandEnv
from .networks import (
    Actor,
    WorldModel,
    append_history,
    initial_history,
    mimic_reward,
    pen_axis,
    survival_probability,
)


def tensor(value, device):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)


def references(clip, stage, device):
    ref = np.zeros((len(clip.q), 57), dtype=np.float32)
    ref[:, :22], ref[:, 22:44], ref[:, 44:51] = clip.q, clip.qd, clip.pose
    if stage == "hold":
        ref[:, 22:44] = 0
    return tensor(ref, device)


@torch.no_grad()
def collect(env, actor, steps, device, rng, noise=0.04, random_start=False):
    records = []
    state = env.reset()
    history = initial_history(tensor(state[None, :22], device))
    for _ in range(steps):
        frame = env.frame
        next_frame = frame if env.stage == "hold" else frame + 1
        phase = tensor([[frame / env.length]], device)
        action = (
            actor(history, tensor(env.clip.q[None, next_frame], device), phase)[0].cpu().numpy()
        )
        next_state, done, _ = env.step(action + rng.normal(0, noise, 22))
        if next_state[47:51] @ state[47:51] < 0:
            next_state[47:51] *= -1
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
            tensor(next_state[None, :22], device),
            tensor(state[None, :22], device),
            tensor(env.previous[None], device),
        )
        state = next_state
        if done:
            start = int(rng.integers(env.length)) if random_start and env.stage != "hold" else 0
            state = env.reset(start)
            history = initial_history(tensor(state[None, :22], device))
    return records


def batch(records, count, device, rng):
    indices = rng.integers(len(records), size=count)
    cols = list(zip(*(records[int(i)] for i in indices)))
    return [tensor(np.stack(c), device) for c in cols]


@torch.no_grad()
def validate_rollout(wm, records, horizon, device):
    """用真实动作检验多步开环误差，窗口不能跨越 reset。"""
    starts = [
        i
        for i in range(len(records) - horizon + 1)
        if not any(records[j][5] for j in range(i, i + horizon - 1))
    ]
    if not starts:
        return {"windows": 0, "position_error_m": None}
    prediction = tensor(np.stack([records[i][0] for i in starts]), device)
    errors = []
    for h in range(horizon):
        action = tensor(np.stack([records[i + h][1] for i in starts]), device)
        target = tensor(np.stack([records[i + h][2] for i in starts]), device)
        prediction = wm(prediction, action)
        errors.append(float((prediction[:, 44:47] - target[:, 44:47]).norm(dim=-1).mean()))
    return {"windows": len(starts), "position_error_m": errors}


@torch.no_grad()
def evaluate(env, actor, device, output=None, baseline=False):
    state = env.reset()
    history = initial_history(tensor(state[None, :22], device))
    ref = references(env.clip, env.stage, device)
    rewards, errors, angles, states, targets = [], [], [], [state.copy()], []
    done, info = False, {}
    while not done:
        frame = env.frame
        nxt = frame if env.stage == "hold" else frame + 1
        previous = env.previous.copy()
        if baseline:
            action = env.clip.q[nxt]
        else:
            action = (
                actor(history, ref[None, nxt, :22], tensor([[frame / env.length]], device))[0]
                .cpu()
                .numpy()
            )
        next_state, done, info = env.step(action)
        s = tensor(next_state[None], device)
        rewards.append(
            float(
                mimic_reward(
                    s,
                    ref[None, nxt],
                    tensor(env.previous[None], device),
                    tensor(previous[None], device),
                )
            )
        )
        errors.append(info["position_error_m"])
        dot = (pen_axis(s[:, 47:51]) * pen_axis(ref[None, nxt, 47:51])).sum().abs().clamp(0, 1)
        angles.append(float(torch.acos(dot).cpu()) * 180 / np.pi)
        history = append_history(
            history, s[:, :22], tensor(state[None, :22], device), tensor(env.previous[None], device)
        )
        state = next_state
        states.append(state.copy())
        targets.append(env.previous.copy())
    metrics = {
        "steps": len(rewards),
        "duration_s": len(rewards) * env.clip.dt,
        "dropped": info["dropped"],
        "completed_without_drop": not info["dropped"] and info["truncated"],
        "mean_position_error_m": float(np.mean(errors)),
        "max_position_error_m": max(errors),
        "mean_axis_error_deg": float(np.mean(angles)),
        "mean_reward": float(np.mean(rewards)),
        "return": float(np.sum(rewards)),
        "baseline": baseline,
        "start_frame": 0,
        "initial_pen_velocity": "zero",
        "stage": env.stage,
    }
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output.with_suffix(".npz"),
            states=np.asarray(states),
            targets=np.asarray(targets),
            dt=env.clip.dt,
        )
        output.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def train(args):
    if (
        min(
            args.iterations,
            args.collect_steps,
            args.wm_steps,
            args.policy_steps,
            args.batch_size,
            args.horizon,
        )
        < 1
    ):
        raise ValueError("训练次数必须为正")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    clip = Clip(args.clip, args.asset)
    env = HandEnv(clip, args.stage)
    actor, wm = Actor(env.limits).to(device), WorldModel().to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=3e-4)
    replay = deque(maxlen=5000)
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt["provenance"] != clip.provenance() or ckpt["stage"] != args.stage:
            raise ValueError("resume 必须匹配数据、资产和阶段；跨阶段使用 --initialize")
        actor.load_state_dict(ckpt["actor"])
        wm.load_state_dict(ckpt["wm"])
        actor_opt.load_state_dict(ckpt["actor_optimizer"])
        wm_opt.load_state_dict(ckpt["wm_optimizer"])
        replay.extend(ckpt["replay"])
        rng.bit_generator.state = ckpt["numpy_rng"]
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        start = ckpt["iteration"] + 1
    elif args.initialize:
        ckpt = torch.load(args.initialize, map_location=device, weights_only=False)
        if ckpt["provenance"] != clip.provenance():
            raise ValueError("initialize 数据或模型不匹配")
        actor.load_state_dict(ckpt["actor"])
        wm.load_state_dict(ckpt["wm"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint.pt").exists() and not args.resume:
        raise FileExistsError("输出已有 checkpoint，请换目录或显式 resume")
    (out / "provenance.json").write_text(
        json.dumps(clip.provenance(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    baseline = evaluate(env, actor, device, out / "baseline", baseline=True)
    print("baseline", json.dumps(baseline), flush=True)
    ref = references(clip, args.stage, device)
    for iteration in range(start, start + args.iterations):
        replay.extend(
            collect(env, actor, args.collect_steps, device, rng, random_start=args.random_start)
        )
        # 整批独立采集的验证转移不进入训练 buffer，避免随机拆相邻帧造成泄漏。
        validation = collect(
            env,
            actor,
            max(32, args.collect_steps // 4),
            device,
            np.random.default_rng(100000 + iteration),
            noise=0.02,
        )
        wm.requires_grad_(True)
        for _ in range(args.wm_steps):
            state, action, next_state, _, _, _ = batch(replay, args.batch_size, device, rng)
            wm_loss = wm.loss(state, action, next_state)
            if not torch.isfinite(wm_loss):
                raise FloatingPointError("world model loss")
            wm_opt.zero_grad()
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            wm_opt.step()
        with torch.no_grad():
            vs, va, vn, _, _, _ = batch(validation, len(validation), device, rng)
            validation_loss = float(wm.loss(vs, va, vn))
        rollout_error = validate_rollout(wm, validation, args.horizon, device)
        wm.requires_grad_(False)
        for _ in range(args.policy_steps):
            state, _, _, history, frame, _ = batch(replay, args.batch_size, device, rng)
            frame = frame.long()
            loss = torch.zeros((), device=device)
            alive = torch.ones_like(frame, dtype=torch.float32)
            # 初态取真实 replay；短视野和小残差限制策略利用模型误差。
            for h in range(args.horizon):
                active = (
                    (frame < env.length).float()
                    if args.stage != "hold"
                    else torch.ones_like(frame).float()
                )
                nxt = frame if args.stage == "hold" else (frame + 1).clamp(max=env.length)
                action = actor(history, ref[nxt, :22], frame[:, None].float() / env.length)
                prediction = wm(state, action)
                reward = mimic_reward(prediction, ref[nxt], action, history[:, -1, 44:66])
                loss = loss - 0.97**h * (reward * active * alive).mean() / args.horizon
                # 真实仿真掉笔后会终止；想象中不能继续领取掉笔后的奖励。
                alive = alive * survival_probability(prediction, ref[nxt])
                history = append_history(history, prediction[:, :22], state[:, :22], action)
                state, frame = prediction, nxt
            if not torch.isfinite(loss):
                raise FloatingPointError("policy loss")
            actor_opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            if not torch.isfinite(grad):
                raise FloatingPointError("policy gradient")
            actor_opt.step()
        metrics = evaluate(env, actor, device, out / "evaluation")
        row = {
            "iteration": iteration,
            "replay_size": len(replay),
            "wm_train_loss": float(wm_loss.detach()),
            "wm_validation_loss": validation_loss,
            "policy_loss": float(loss.detach()),
            "wm_open_loop": rollout_error,
            "policy_grad_norm": float(grad),
            "evaluation": metrics,
        }
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        ckpt = {
            "actor": actor.state_dict(),
            "wm": wm.state_dict(),
            "stage": args.stage,
            "actor_optimizer": actor_opt.state_dict(),
            "wm_optimizer": wm_opt.state_dict(),
            "iteration": iteration,
            "provenance": clip.provenance(),
            "replay": list(replay),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
        }
        torch.save(ckpt, out / "checkpoint.tmp")
        (out / "checkpoint.tmp").replace(out / "checkpoint.pt")
    export(actor, clip, env, out)


def export(actor, clip, env, out):
    actor = actor.cpu().eval()
    q = torch.tensor(clip.q[None, 0])
    scripted = torch.jit.trace(actor, (initial_history(q), q, torch.zeros(1, 1)))
    scripted.save(str(out / "actor.ts"))
    # 部署包只含 Actor 和已知参考关节，不包含世界模型及笔轨迹。
    np.savez_compressed(out / "joint_reference.npz", q=clip.q, dt=clip.dt)
    schema = {
        **clip.provenance(),
        "history_shape": [1, 30, 66],
        "history_channels": [
            "q_rad[22]",
            "clip((q-q_previous)/dt/10,-5,5)[22]",
            "executed_target_rad[22]",
        ],
        "inputs": ["history", "next_reference_q", "current_frame/stage_length"],
        "output": "22 joint targets in rad, dataset joint order",
        "stage": env.stage,
        "stage_length": env.length,
        "max_target_delta_rad": 0.12,
        "residual_limit_rad": 0.15,
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "evaluate"])
    p.add_argument("--clip", default=str(DEFAULT_CLIP))
    p.add_argument("--asset", default=str(DEFAULT_ASSET))
    p.add_argument("--stage", choices=["hold", "short", "full"], default="short")
    p.add_argument("--output", default="runs/hand/short")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--collect-steps", type=int, default=512)
    p.add_argument("--wm-steps", type=int, default=200)
    p.add_argument("--policy-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--random-start", action="store_true")
    init = p.add_mutually_exclusive_group()
    init.add_argument("--resume")
    init.add_argument("--initialize")
    p.add_argument("--checkpoint")
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == "train":
        train(args)
    else:
        env = HandEnv(Clip(args.clip, args.asset), args.stage)
        actor = Actor(env.limits).to(args.device)
        if not args.baseline:
            if not args.checkpoint:
                p.error("evaluate 需要 --checkpoint 或 --baseline")
            ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
            if ckpt["provenance"] != env.clip.provenance() or ckpt["stage"] != args.stage:
                raise ValueError("评估配置不匹配")
            actor.load_state_dict(ckpt["actor"])
        print(
            json.dumps(
                evaluate(env, actor, args.device, Path(args.output) / "evaluation", args.baseline),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
