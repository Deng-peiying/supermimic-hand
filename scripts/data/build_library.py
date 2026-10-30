"""
将筛选后的 NPZ 动作片段打包成一个紧凑的动作库文件，
其中参考状态采用 FP16（16 位浮点数）格式，每个状态包含 71 个维度。
运行命令：
uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed --manifest data/npz/bones_seed/manifest_filtered.csv --out data/library/bones_seed_compact.pt
多个 NPZ 动作文件
       ↓
读取每个文件中的骨盆状态和关节状态
       ↓
拼接为每帧 71 维参考状态
       ↓
转换为 FP16
       ↓
合并所有动作片段
       ↓
保存为 bones_seed_compact.pt
BONES-SEED 中大量独立的 .npz 动作片段，打包成一个 .pt 文件，方便 SuperMimic 在训练时快速读取和随机采样参考动作
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro
from tqdm import tqdm

from supermimic.envs.mdp.motion_library import discover_npz


def main(npz_dir: Path, out: Path, manifest: Path | None = None, workers: int = 16) -> None:
    files = discover_npz(npz_dir, manifest=manifest)
    print(f"{len(files)} clips")
    from concurrent.futures import ThreadPoolExecutor

    def load(f):
        with np.load(f) as d:
            names = [str(n) for n in d["body_names"]]
            r = names.index("pelvis") # 找到骨盆对应的索引
            ref = np.concatenate([d["body_pos_w"][:, r],  # 骨盆在世界坐标系中的位置  3
                                  d["body_quat_w"][:, r], # 骨盆在世界坐标系中的姿态四元数  4
                                  d["joint_pos"],         # 机器人关节位置  N 
                                  d["body_lin_vel_w"][:, r], # 骨盆世界坐标系线速度 3
                                  d["body_ang_vel_w"][:, r],  # 骨盆世界坐标系角速度 3
                                  d["joint_vel"]], axis=1) # 构造每帧参考状态 N
        return f.stem, ref.astype(np.float16)

    with ThreadPoolExecutor(workers) as ex: # 使用 16 个线程并行读取 workers: int = 16，函数默认输入16
        items = list(tqdm(ex.map(load, files), total=len(files)))
    names = [n for n, _ in items]
    lengths = [int(r.shape[0]) for _, r in items]
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ref = torch.from_numpy(np.concatenate([r for _, r in items], 0)) # 合并并保存
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref, "names": names, "starts": starts, "lengths": lengths}, out) # 最终 .pt 文件保存四个字段
    print(f"saved {out}: {ref.shape[0]} frames ({ref.shape[0]/50/3600:.2f} h), {ref.numel()*2/1e9:.2f} GB fp16")


if __name__ == "__main__":
    tyro.cli(main)
