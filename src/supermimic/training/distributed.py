"""数据并行训练辅助工具（计划 D13 / 6.3）。
使用 torchrun 启动，每张 GPU 对应一个独立进程：
  torchrun --nproc_per_node 4 -m supermimic.scripts.run g1_tracking --run-name e5_ws ...
每个进程（rank）独立拥有自己的仿真器（包含 num_envs 个并行环境）、经验回放缓冲区（replay buffer）和采样器（sampler）。
世界模型（world model）、策略网络（policy）和价值评估网络（critic）在各个进程中分别维护一份副本。
在反向传播（backward）之后，通过显式的 all-reduce 操作对各进程的梯度进行聚合并求平均。
不使用 DDP（DistributedDataParallel）封装器，因此不会影响已经通过 torch.compile 编译的函数。
学习率按照 sqrt(world_size) 进行缩放。
论文中的设置：使用 2 张 GPU 时，学习率乘以 sqrt(2)。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed() -> tuple[int, int, int]:
    """返回全局进程编号、本地进程编号、总进程数；通过 torchrun 启动时初始化 NCCL。"""
    if not is_distributed(): # 非分布式模式
        return 0, 0, 1 # 单进程：rank=0，local_rank=0，world_size=1
    rank = int(os.environ["RANK"]) # 获取全局进程编号
    world = int(os.environ["WORLD_SIZE"]) # 获取总进程数
    pinned = os.environ.get("SM_GPU_PINNED") == "1"  # 检查是否已在 train.py 中绑定 GPU
    # 已绑定时每个进程只看到一张 GPU（cuda:0）；否则取模以支持单 GPU 冒烟测试
    local_rank = 0 if pinned else int(os.environ["LOCAL_RANK"]) % max(torch.cuda.device_count(), 1)
    torch.cuda.set_device(local_rank)
    # NCCL 不支持多个进程共享同一张 GPU；单 GPU 多进程测试回退到 Gloo
    one_gpu_per_rank = pinned or torch.cuda.device_count() >= world
    backend = os.environ.get("SM_DIST_BACKEND") or ("nccl" if one_gpu_per_rank else "gloo")
    dist.init_process_group(backend, rank=rank, world_size=world) # 初始化分布式通信组 
    return rank, local_rank, world # 返回进程信息


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


@torch.no_grad()
def allreduce_grads(module: torch.nn.Module) -> None:
    """Average gradients across ranks (flattened into one buffer per dtype)."""
    if world_size() == 1:
        return
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


@torch.no_grad()
def broadcast_params(module: torch.nn.Module, src: int = 0) -> None:
    if world_size() == 1:
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def allreduce_mean(x: float) -> float:
    if world_size() == 1:
        return x
    t = torch.tensor([x], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t)


def finalize() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
