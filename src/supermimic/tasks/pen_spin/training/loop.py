"""完整模型式训练调度：并行采集、k步 WM、BPTT Actor、GAE Critic、余弦学习率。"""

import copy
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch

from supermimic.training.schedule import cosine_lr

from ..data import Clip
from ..data.replay import HandReplay
from ..data.sampling import ReturnSampler
from ..deployment.export import export
from ..envs.mdp.observations import HISTORY
from ..envs.parallel_env import ParallelHandEnv
from ..imagination.policy_trainer import PolicyTrainer
from ..policy.nets import Actor, Critic
from ..policy.reference import future_joint_reference
from ..world_model.wm_trainer import WorldModelTrainer
from ..world_model.world_model import WorldModel
from .checkpoint import check_task_config
from .logging import format_progress


class Trainer:
    def __init__(self, cfg):
        for key in (
            "num_envs",
            "iterations",
            "wm_steps",
            "wm_window",
            "batch_size",
            "policy_batch_size",
            "horizon",
            "critic_epochs",
            "critic_minibatches",
            "policy_steps",
            "log_interval",
            "checkpoint_interval",
        ):
            if getattr(cfg, key) < 1:
                raise ValueError(f"{key} 必须为正")
        if (
            cfg.replay_capacity <= max(HISTORY + 2, cfg.wm_window + 2)
            or cfg.warmup <= cfg.wm_window
        ):
            raise ValueError("replay容量或预热不足以覆盖历史/WM窗口")
        if not 0 < cfg.gamma < 1 or cfg.updates_per_env_step <= 0:
            raise ValueError("gamma/更新比例非法")
        self.cfg = cfg
        torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.out = Path(cfg.output)
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "checkpoint.pt").exists() and not cfg.resume:
            raise FileExistsError("已有检查点，请换输出目录或明确 resume")
        self.clip = Clip(cfg.clip, cfg.asset)
        self.env = ParallelHandEnv(
            self.clip,
            cfg.num_envs,
            cfg.device,
            stage=cfg.stage,
            action_noise=cfg.action_noise,
            nconmax=cfg.nconmax,
            njmax=cfg.njmax,
            domain_randomization=cfg.domain_randomization,
        )
        self.actor = Actor(self.env.limits).to(cfg.device)
        self.critic = Critic().to(cfg.device)
        self.wm = WorldModel(
            (cfg.wm_hidden,) * 3, self.clip.dt, drop_distance=self.env.cfg.drop_distance_m
        ).to(cfg.device)
        self.replay = HandReplay(cfg.num_envs, cfg.replay_capacity, cfg.device)
        self.sampler = ReturnSampler(
            cfg.num_envs,
            self.env.length,
            cfg.device,
            cfg.gamma,
            cfg.sampling,
            max_step_reward=sum(max(0, term.weight) for term in self.env.cfg.rewards.values())
            * self.env.reward_manager.scale,
        )
        self.wm_trainer = WorldModelTrainer(self.wm, self.env.ref, cfg)
        self.policy_trainer = PolicyTrainer(self.wm, self.actor, self.critic, self.env, cfg)
        self.iteration, self.updates, self.budget = 0, 0, 0.0
        self.stop = False
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location=cfg.device, weights_only=False)
            if (
                ck.get("format") != "prima1-supermimic-v5"
                or ck["provenance"] != self.clip.provenance()
            ):
                raise ValueError("无法恢复不同算法版本或数据的检查点")
            check_task_config(ck, self.env)
            # 允许运行位置和最终预算改变；算法、观测、物理配置必须一致。
            allowed = {
                "output",
                "resume",
                "iterations",
                "log_interval",
                "checkpoint_interval",
                "device",
                "threads",
                "clip",
                "asset",
            }
            if any(ck["config"].get(k) != v for k, v in asdict(cfg).items() if k not in allowed):
                raise ValueError("恢复配置不匹配，请使用 initialize 开始新实验")
            for name in ("actor", "wm", "critic"):
                getattr(self, name).load_state_dict(ck[name])
            self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
            self.policy_trainer.opt.load_state_dict(ck["actor_opt"])
            self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
            self.policy_trainer.target_critic.load_state_dict(ck["target_critic"])
            self.iteration, self.updates = ck["iteration"], ck["updates"]
            self.sampler.value.copy_(ck["bin_return"])
            self.sampler.memory.copy_(ck["bin_memory"])
        elif cfg.initialize:
            ck = torch.load(cfg.initialize, map_location=cfg.device, weights_only=False)
            if ck.get("format") not in ("prima1-supermimic-v4", "prima1-supermimic-v5"):
                raise ValueError("未来10帧Actor接口已改变，旧检查点不可initialize")
            if ck["provenance"] != self.clip.provenance():
                raise ValueError("initialize 数据不匹配")
            self.actor.load_state_dict(ck["actor"])
        for name, data in [
            ("config", asdict(cfg)),
            ("task_config", self.env.cfg.describe()),
            ("provenance", self.clip.provenance()),
        ]:
            (self.out / f"{name}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(
            "READY",
            json.dumps(
                {
                    "envs": cfg.num_envs,
                    "state_dim": 57,
                    "history": HISTORY,
                    "wm_batch": cfg.batch_size,
                    "policy_batch": cfg.policy_batch_size,
                    "horizon": cfg.horizon,
                    "clip_frames": len(self.clip.q),
                }
            ),
            flush=True,
        )

    def save(self, numbered=False):
        checkpoint = {
            "format": "prima1-supermimic-v5",
            "iteration": self.iteration,
            "updates": self.updates,
            "config": asdict(self.cfg),
            "stage": self.cfg.stage,
            "provenance": self.clip.provenance(),
            "task_config": self.env.cfg.describe(),
            "actor": self.actor.state_dict(),
            "wm": self.wm.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.policy_trainer.target_critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(),
            "actor_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "bin_return": self.sampler.value,
            "bin_memory": self.sampler.memory,
            "resume_semantics": "restore models/optimizers/sampler; reset environments and refill replay",
        }
        target = self.out / (f"ckpt_{self.iteration:07d}.pt" if numbered else "checkpoint.pt")
        temp = target.with_suffix(".tmp")
        torch.save(checkpoint, temp)
        temp.replace(target)

    def run(self):
        cfg, env = self.cfg, self.env
        ids = torch.arange(cfg.num_envs, device=cfg.device)
        if cfg.stage != "hold":
            env.reset(ids, self.sampler.sample(cfg.num_envs))
        start_iteration = self.iteration
        t0, tlog = time.monotonic(), time.monotonic()
        metrics, episode_metrics = {}, []
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        max_reward = sum(max(0, term.weight) for term in env.cfg.rewards.values())
        max_reward *= env.reward_manager.scale
        for iteration in range(self.iteration + 1, cfg.iterations + 1):
            phase_start = time.monotonic()
            with torch.no_grad():
                packed, frame, xi = env.packed_state(), env.frame.clone(), env.xi.clone()
                action = self.actor(
                    env.history,
                    future_joint_reference(env.ref[:, :22], frame, env.length, env.stage),
                    frame[:, None].float() / env.length,
                )
                _, reward, failed, ended = env.step(action)
                self.replay.record(
                    ids,
                    packed,
                    action,
                    env.previous,
                    xi,
                    reward.total,
                    frame,
                    env.packed_state(),
                    env.frame,
                    failed,
                    ended,
                )
                stats = self.sampler.record(reward.total, frame, failed | ended)
                if stats:
                    stats["failures"] = int(failed.sum())
                    stats["reference_ends"] = int((ended & ~failed).sum())
                    episode_metrics.append(stats)
                reset_ids = ids[failed | ended]
                frames = (
                    torch.zeros_like(reset_ids)
                    if cfg.stage == "hold"
                    else self.sampler.sample(len(reset_ids))
                )
                env.reset(reset_ids, frames)
            timers["sim"] += time.monotonic() - phase_start
            self.iteration = iteration
            if iteration - start_iteration >= cfg.warmup:
                self.budget += cfg.updates_per_env_step
                while self.budget >= 1:
                    lr = cosine_lr(iteration, cfg.iterations, cfg.lr)
                    for optimizer in (
                        self.wm_trainer.opt,
                        self.policy_trainer.opt,
                        self.policy_trainer.critic_opt,
                    ):
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    phase_start = time.monotonic()
                    metrics["wm"] = self.wm_trainer.update(self.replay)
                    timers["wm"] += time.monotonic() - phase_start
                    phase_start = time.monotonic()
                    for _ in range(cfg.policy_steps):
                        metrics["policy"] = self.policy_trainer.update(self.replay)
                    timers["policy"] += time.monotonic() - phase_start
                    self.updates += 1
                    self.budget -= 1
            if iteration % cfg.log_interval == 0 or iteration == start_iteration + 1:
                now = time.monotonic()
                episodes = sum(x["episodes"] for x in episode_metrics)
                row = {
                    "iteration": iteration,
                    "hours": (now - t0) / 3600,
                    "updates": self.updates,
                    "iterations_per_s": (
                        1 if iteration == start_iteration + 1 else cfg.log_interval
                    )
                    / (now - tlog),
                    "episodes": episodes,
                    "failure_count": sum(x["failures"] for x in episode_metrics),
                    "reference_end_count": sum(x["reference_ends"] for x in episode_metrics),
                    "train_reward_mean": sum(
                        x["mean_reward"] * x["episodes"] for x in episode_metrics
                    )
                    / max(1, episodes),
                    "gpu_peak_gb": torch.cuda.max_memory_allocated(cfg.device) / 1e9,
                    "reward_terms": {k: float(v.mean()) for k, v in reward.weighted.items()},
                    "capacity": env.check_capacity(),
                    "timing": dict(timers),
                    **metrics,
                }
                print(format_progress(row, max_reward, cfg.wm_window), flush=True)
                with (self.out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                tlog, episode_metrics = now, []
                timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
            if iteration % cfg.checkpoint_interval == 0:
                self.save(numbered=True)
                self.save()
            if self.stop:
                break
        self.save(numbered=True)
        self.save()
        export(copy.deepcopy(self.actor), self.clip, env, self.out)


def train(cfg):
    trainer = Trainer(cfg)

    def stop(signum, frame):
        trainer.stop = True
        print(f"Stop requested ({signum}); saving after current iteration.", flush=True)

    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        trainer.run()
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)
        trainer.env.close()
