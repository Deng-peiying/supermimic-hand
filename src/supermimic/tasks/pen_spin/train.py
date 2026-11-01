"""Prima1 完整 SuperMimic：GPU并行训练 / CPU独立物理评估 / 配置查看。"""

import json
from pathlib import Path

import torch

from .config import TrainConfig, make_parser
from .data import Clip
from .envs.env import HandEnv
from .envs.pen_spin_env_cfg import make_env_cfg
from .eval.benchmark import evaluate
from .policy.nets import Actor
from .training.checkpoint import check_task_config
from .training.loop import train


def main():
    p = make_parser()
    args = TrainConfig(**vars(p.parse_args()))
    if args.mode == "describe":
        print(json.dumps(make_env_cfg(args.stage).describe(), ensure_ascii=False, indent=2))
        return
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
            check_task_config(ckpt, env)
            actor.load_state_dict(ckpt["actor"])
        print(
            json.dumps(
                evaluate(env, actor, args.device, Path(args.output) / "evaluation", args.baseline),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
