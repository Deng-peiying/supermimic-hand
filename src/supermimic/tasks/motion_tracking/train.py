"""训练 SuperMimic。运行示例：
  uv run sm-run g1_tracking --run-name e1_walk --data.clips walk1_subject1 --env.num-envs 1024
"""

from __future__ import annotations

import os

# 在 torchrun 分布式启动模式下，必须在 CUDA/Warp 初始化之前，将每个进程（rank）固定到各自的 GPU。
# 原因：MuJoCo Warp 和 mjlab 的仿真始终运行在当前进程可见的第一块 GPU 上，

def _n_gpus() -> int:
    try:
        return len(os.listdir("/proc/driver/nvidia/gpus")) # 如果读取 Linux 目录时发生特定的操作系统错误，就交给 except 处理
    except OSError:
        return 0


_preset = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d] #读取环境变量 CUDA_VISIBLE_DEVICES
_avail = _preset if _preset else [str(i) for i in range(_n_gpus())] # 确定可用 GPU 列表
if "LOCAL_RANK" in os.environ and int(os.environ.get("LOCAL_WORLD_SIZE", "1")) <= len(_avail) and (not _preset or len(_preset) > 1):
    # 判断是否需要 GPU 绑定
    # "LOCAL_RANK" in os.environ	当前是否存在本地进程编号
    # LOCAL_WORLD_SIZE <= len(_avail)	当前机器上的本地进程数量不超过候选 GPU 数量
    # not _preset or len(_preset) > 1	没有预设 GPU，或者预设了多张 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = _avail[int(os.environ["LOCAL_RANK"])] 
    os.environ["SM_GPU_PINNED"] = "1" # 设置 GPU 绑定标记
    # 为每个进程分配独立的一组 CPU 核心。 
    # 原因：训练循环在 CPU 端主要受单线程性能限制。
    # 如果多个进程共享 CPU 核心，会产生执行时间抖动（jitter）。
    # 而分布式训练中的同步 All-Reduce 操作会把这种抖动转化为等待时间，
    # 从而降低整体训练效率。
    try:
        cores = sorted(os.sched_getaffinity(0)) # 获取当前进程允许使用的 CPU 核心，os.sched_getaffinity(0) 返回当前进程被允许在哪些逻辑 CPU 上执行
        n_local = int(os.environ["LOCAL_WORLD_SIZE"])
        per = len(cores) // n_local # 计算每个进程分配多少 CPU 核心
        if per >= 2: # 判断是否具有足够的 CPU 核心
            r = int(os.environ["LOCAL_RANK"]) # 获取当前进程编号
            os.sched_setaffinity(0, set(cores[r * per : (r + 1) * per]))  # 真正进行 CPU 绑定
    except (OSError, ValueError):
        pass

import tyro

from .config import TrainConfig


def main() -> None:
    cfg = tyro.cli(TrainConfig) # 解析训练配置，tyro 可以根据 Python 的 dataclass 自动生成命令行参数接口 -> cfg.seed
    from .training.loop import Trainer # 导入训练器

    Trainer(cfg).run() # 创建训练器并执行训练


if __name__ == "__main__":
    main()
