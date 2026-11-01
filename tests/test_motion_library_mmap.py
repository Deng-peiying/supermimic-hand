"""CPU 映射库与原 GPU compact 库必须对同一参考帧给出一致结果。"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.data.compact_cache import build
from supermimic.tasks.motion_tracking.envs.mdp.motion_library import MotionLibrary, discover_npz
from supermimic.tasks.motion_tracking.envs.tracking_env_cfg import TRACKED_BODIES

NPZ = Path(__file__).resolve().parents[1] / "data/npz/lafan1"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA and local LAFAN1"
)
def test_mapped_library_matches_gpu_and_clip_boundaries(tmp_path):
    files = discover_npz(NPZ, names=["walk1_subject1", "dance1_subject1"])
    assert len(files) == 2
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "frames", "keep", "reason"])
        for file in files:
            with np.load(file) as data:
                writer.writerow([file.stem, len(data["joint_pos"]), 1, ""])
    out = tmp_path / "cache"
    build(NPZ, manifest, out, workers=2)
    descriptor = out / "library.json"
    info = json.loads(descriptor.read_text())
    assert (
        hashlib.sha256((out / "reference.npy").read_bytes()).hexdigest() == info["reference_sha256"]
    )
    q0 = torch.zeros(29)
    gpu = MotionLibrary(files, TRACKED_BODIES, q0, device="cuda", compact=True)
    mapped = MotionLibrary([descriptor], TRACKED_BODIES, q0, device="cuda", compact=True)
    assert mapped._ref.device.type == "cpu" and gpu._ref.device.type == "cuda"
    frames = torch.cat(
        [gpu.clip_start, gpu.clip_last, torch.randint(gpu.num_frames, (32,), device="cuda")]
    )
    assert torch.equal(gpu.ref71(frames), mapped.ref71(frames))
    # TorchScript 首次执行与优化后融合核可能有浮点舍入差；原始参考值须逐位一致。
    torch.testing.assert_close(
        gpu.goal_window(frames, 10), mapped.goal_window(frames, 10), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        gpu.bodies(frames).pos_w, mapped.bodies(frames).pos_w, atol=1e-6, rtol=1e-6
    )
    assert torch.equal(gpu.last_frame_of(frames), mapped.last_frame_of(frames))
    assert torch.equal(gpu.bin_of_frame(frames), mapped.bin_of_frame(frames))
    assert torch.equal(gpu.ref71(frames.reshape(6, 6)), mapped.ref71(frames.reshape(6, 6)))
    with pytest.raises(FileExistsError):
        build(NPZ, manifest, out, workers=2)
    info["reference_bytes"] += 1
    descriptor.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Truncated"):
        MotionLibrary([descriptor], TRACKED_BODIES, q0, device="cuda", compact=True)
