r"""
翻译：
将重定向到 Unitree G1 的 CSV 动作文件批量转换为与 mjlab 兼容的 50 Hz NPZ 动作文件。
运行示例：
```
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 \
    --output-dir data/npz/lafan1 --input-fps 30 --workers 16
```
参数说明：

| 参数               | 含义                        |
| ---------------- | ------------------------- |
| `--input-dir`    | 输入目录，存放 G1 重定向后的 CSV 动作文件 |
| `--output-dir`   | 输出目录，保存转换后的 NPZ 文件        |
| `--input-fps 30` | 输入动作数据的采样帧率为 30 FPS       |
| `--workers 16`   | 使用最多 16 个工作线程并行转换         |
| `--fps-map`      | 可选参数，为不同动作片段指定各自的原始帧率     |

输出文件：
脚本会为每个 CSV 文件生成对应的 NPZ 文件，并在输出目录中创建 `manifest.csv` 清单文件。
```
data/raw/lafan1/g1/
├── walk.csv
├── run.csv
└── jump.csv
         ↓
data/npz/lafan1/
├── walk.npz
├── run.npz
├── jump.npz
└── manifest.csv
```

如果不同动作片段的原始帧率不相同，可以使用 `--fps-map` 指定一个 CSV 映射文件，其中包含 `name,fps` 两列，为每个动作片段单独设置输入帧率。
与前面 `build_library.py` 的关系：
\\[ \boxed{ \text{G1 CSV} \xrightarrow[\text{重采样至 50 Hz}]{\text{csv\\\_to\\\_npz\\\_batch.py}} \text{多个 NPZ} \xrightarrow{\text{build\\\_library.py}} \text{一个 PT 动作库} } \\]

第一个脚本负责格式转换与帧率统一，第二个脚本负责将多个动作片段整理、拼接并保存为一个紧凑的训练动作库。
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import tyro

_MODEL = None


def _init_worker() -> None:
    global _MODEL  # 声明使用模块级全局变量 _MODEL
    # Keep BLAS single-threaded inside workers. 每个工作进程加载一次 G1 机器人模型，供后续动作转换重复使用
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from supermimic.tasks.motion_tracking.data.motion_convert import load_g1_model

    _MODEL = load_g1_model() # 保存在当前工作进程中，避免每转换一个 CSV 都重新加载模型


def _convert_one(args: tuple[str, str, float, float]) -> dict:
    from supermimic.tasks.motion_tracking.data.motion_convert import convert_csv_to_npz

    src, dst, in_fps, out_fps = args
    try:
        info = convert_csv_to_npz(src, dst, in_fps, out_fps, model=_MODEL)
        info["ok"] = True
        info["error"] = ""
    except Exception as e:  # noqa: BLE001
        info = {"source": src, "out": dst, "ok": False, "error": repr(e), "input_frames": 0, "output_frames": 0, "duration_s": 0.0}
    return info


def main(
    input_dir: Path,
    output_dir: Path,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    workers: int = 16,
    pattern: str = "**/*.csv", # 递归查找所有 CSV 文件
    fps_map: Path | None = None,
    overwrite: bool = False, # 默认不覆盖已有 NPZ
    limit: int | None = None,
) -> None:
    files = sorted(input_dir.glob(pattern))
    if limit is not None:
        files = files[:limit]
    if not files:
        print(f"no files matching {pattern} under {input_dir}", file=sys.stderr)
        sys.exit(1)
    fps_by_name: dict[str, float] = {}
    if fps_map is not None:
        with open(fps_map) as f:
            for row in csv.DictReader(f):
                fps_by_name[row["name"]] = float(row["fps"])

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for src in files:
        name = src.stem
        dst = output_dir / f"{name}.npz"
        if dst.exists() and not overwrite:
            continue
        jobs.append((str(src), str(dst), fps_by_name.get(name, input_fps), output_fps))
    print(f"{len(files)} csv files, {len(jobs)} to convert, {workers} workers")

    t0 = time.time()
    results: list[dict] = []
    if jobs:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            for i, info in enumerate(pool.imap_unordered(_convert_one, jobs, chunksize=4), 1):
                results.append(info)
                if i % 200 == 0 or i == len(jobs):
                    dt = time.time() - t0
                    print(f"  {i}/{len(jobs)} done, {dt:.0f}s elapsed, {i / dt:.1f} clips/s")
    failures = [r for r in results if not r["ok"]]
    for r in failures[:20]:
        print("FAILED", r["source"], r["error"])

    # Manifest over everything present in output_dir (so re-runs stay complete).
    import numpy as np

    rows = []
    for npz in sorted(output_dir.glob("*.npz")):
        with np.load(npz) as d:
            n = int(d["joint_pos"].shape[0])
            fps = float(d["fps"][0])
            in_fps = float(d["input_fps"][0]) if "input_fps" in d else float("nan")
        rows.append({"name": npz.stem, "frames": n, "fps": fps, "duration_s": (n - 1) / fps, "input_fps": in_fps})
    with open(output_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "frames", "fps", "duration_s", "input_fps"])
        w.writeheader()
        w.writerows(rows)
    total_h = sum(r["duration_s"] for r in rows) / 3600.0
    print(f"manifest: {len(rows)} clips, {total_h:.3f} h total, {len(failures)} failures, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    tyro.cli(main)
