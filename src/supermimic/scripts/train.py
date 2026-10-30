"""训练 SuperMimic。运行示例：
  uv run sm-train --run-name e1_walk --data.clips walk1_subject1 --env.num-envs 1024
"""

from __future__ import annotations

import os

# 在 torchrun 分布式启动模式下，必须在 CUDA/Warp 初始化之前，将每个进程（rank）固定到各自的 GPU。
# 原因：MuJoCo Warp 和 mjlab 的仿真始终运行在当前进程可见的第一块 GPU 上，
# 因此仅设置 cfg.device 并不足以保证每个进程使用正确的 GPU。
# 如果机器上的 GPU 数量少于进程数量，则跳过 GPU 独占绑定。
# 例如，在单 GPU 冒烟测试中，多个进程可以通过 gloo 后端共享 cuda:0
def _n_gpus() -> int:
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return 0


# 如果预先设置了 CUDA_VISIBLE_DEVICES，例如 "0,2,3"（跳过故障 GPU），
# 则根据 LOCAL_RANK 从该列表中选择对应的 GPU。
_preset = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
_avail = _preset if _preset else [str(i) for i in range(_n_gpus())]
if "LOCAL_RANK" in os.environ and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) <= len(_avail) and (not _preset or len(_preset) > 1):
    os.environ["CUDA_VISIBLE_DEVICES"] = _avail[int(os.environ["LOCAL_RANK"])]
    os.environ["SM_GPU_PINNED"] = "1"
    # 为每个进程分配独立的一组 CPU 核心。 
    # 原因：训练循环在 CPU 端主要受单线程性能限制。
    # 如果多个进程共享 CPU 核心，会产生执行时间抖动（jitter）。
    # 而分布式训练中的同步 All-Reduce 操作会把这种抖动转化为等待时间，
    # 从而降低整体训练效率。
    try:
        cores = sorted(os.sched_getaffinity(0))
        n_local = int(os.environ["LOCAL_WORLD_SIZE"])
        per = len(cores) // n_local
        if per >= 2:
            r = int(os.environ["LOCAL_RANK"])
            os.sched_setaffinity(0, set(cores[r * per : (r + 1) * per]))
    except (OSError, ValueError):
        pass

import tyro

from supermimic.config import TrainConfig


def main() -> None:
    cfg = tyro.cli(TrainConfig)
    from supermimic.training.loop import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    main()
