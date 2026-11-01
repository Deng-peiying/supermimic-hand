"""完整 SuperMimic 转笔实验配置；CLI 与 Python 共用显式默认值。"""

import argparse
from dataclasses import dataclass, fields

from .data import DEFAULT_ASSET, DEFAULT_CLIP


@dataclass
class TrainConfig:
    mode: str = "train"
    clip: str = str(DEFAULT_CLIP)
    asset: str = str(DEFAULT_ASSET)
    stage: str = "full"
    output: str = "runs/hand/full"
    device: str = "cuda:0"
    seed: int = 42
    threads: int = 4
    num_envs: int = 4096
    iterations: int = 320000
    replay_capacity: int = 256
    wm_steps: int = 10
    wm_window: int = 8
    wm_hidden: int = 512
    policy_steps: int = 1
    batch_size: int = 1024
    policy_batch_size: int = 512
    horizon: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_target_tau: float = 0.005  # 每次策略更新后 EMA；避免同一价值网络快速自举漂移。
    lr: float = 2e-4
    updates_per_env_step: float = 0.4
    warmup: int = 40
    log_interval: int = 100
    checkpoint_interval: int = 4000
    action_noise: float = 0.02
    domain_randomization: bool = True
    sampling: str = "return"
    compile: bool = False
    nconmax: int = 64
    njmax: int = 256
    resume: str | None = None
    initialize: str | None = None
    checkpoint: str | None = None
    baseline: bool = False


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "evaluate", "describe"])
    defaults = TrainConfig()
    init = p.add_mutually_exclusive_group()
    choices = {"stage": ["hold", "short", "full"], "sampling": ["return", "uniform", "start"]}
    for field in fields(defaults):
        if field.name == "mode":
            continue
        value = getattr(defaults, field.name)
        kwargs = {"default": value}
        if isinstance(value, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(value) if value is not None else str
        if field.name in choices:
            kwargs["choices"] = choices[field.name]
        group = init if field.name in ("resume", "initialize") else p
        group.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    return p
