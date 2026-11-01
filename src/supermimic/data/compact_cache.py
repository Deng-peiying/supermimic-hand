"""流式打包完整动作库；限制在途文件数，避免把所有 FP32 动作同时装入内存。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def build(npz_dir: Path, manifest: Path, out: Path, workers: int = 8):
    if workers < 1:
        raise ValueError("workers must be positive")
    with manifest.open(encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["keep"] == "1"]
    rows.sort(key=lambda r: r["name"])
    names = [r["name"] for r in rows]
    lengths = [int(r["frames"]) for r in rows]
    if not names or len(set(names)) != len(names) or min(lengths) < 1:
        raise ValueError("Invalid manifest")
    files = [npz_dir / (n + ".npz") for n in names]
    # WSL 挂载目录逐文件 stat 开销很大；一次列目录核对名称，读取时再检查内容。
    available = {p.name for p in npz_dir.glob("*.npz")}
    for n, p in zip(names, files):
        if Path(n).name != n or p.name not in available:
            raise ValueError(f"Missing or invalid clip: {n}")
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    total = sum(lengths)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "reference.npy"
    metadata = out / "library.json"
    if target.exists() or metadata.exists():
        raise FileExistsError("Use a new output directory; existing cache will not be overwritten")
    temporary = out / "reference.partial.npy"
    if temporary.exists():
        raise FileExistsError("Incomplete build exists; inspect it before retrying")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(total, 71))

    def load(index):
        with np.load(files[index], allow_pickle=False) as data:
            pelvis = list(data["body_names"].astype(str)).index("pelvis")
            ref = np.concatenate(
                [
                    data["body_pos_w"][:, pelvis],
                    data["body_quat_w"][:, pelvis],
                    data["joint_pos"],
                    data["body_lin_vel_w"][:, pelvis],
                    data["body_ang_vel_w"][:, pelvis],
                    data["joint_vel"],
                ],
                axis=1,
            )
        if ref.shape != (lengths[index], 71) or not np.isfinite(ref).all():
            raise ValueError(f"Invalid reference: {files[index]}")
        ref = ref.astype(np.float16)
        if not np.isfinite(ref).all():
            raise ValueError(f"FP16 overflow: {files[index]}")
        return ref

    print(f"Building {len(names)} clips, {total} frames, {total * 142 / 1e9:.2f} GB", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        # Python 3.12 Executor.map 会立即提交所有任务，因此显式限制每组大小。
        chunk = workers * 4
        for first in range(0, len(names), chunk):
            indices = range(first, min(first + chunk, len(names)))
            for i, ref in zip(indices, pool.map(load, indices)):
                array[starts[i] : starts[i] + lengths[i]] = ref
            if first // 2048 != (first + chunk) // 2048 or first + chunk >= len(names):
                array.flush()
                print(f"packed {min(first + chunk, len(names))}/{len(names)}", flush=True)
    array.flush()
    del array
    temporary.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    info = {
        "format": "supermimic-mmap-v1",
        "reference": target.name,
        "reference_sha256": digest.hexdigest(),
        "reference_bytes": target.stat().st_size,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "names": names,
        "starts": starts,
        "lengths": lengths,
        "frames": total,
        "fps": 50,
        "dtype": "float16",
        "state_dim": 71,
    }
    tmp = out / "library.partial.json"
    tmp.write_text(json.dumps(info), encoding="utf-8")
    tmp.replace(metadata)  # 只有完整构建后才发布可加载的入口。
    print(f"saved {metadata}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args.npz_dir, args.manifest, args.out, args.workers)


if __name__ == "__main__":
    main()
