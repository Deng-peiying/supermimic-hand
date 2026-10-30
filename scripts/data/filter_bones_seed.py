"""BONES-SEED 的附录 E 筛选规则（D16 计划）：名称筛选 + 可行性筛选。
读取转换后的 NPZ 动作片段（50 Hz，包含 30 个身体连杆的状态），
并在 <npz_dir>/manifest_filtered.csv 中生成筛选结果文件。

每个动作片段对应 CSV 中的一行，包含以下字段：
- name：动作片段名称
- frames：帧数
- keep：是否保留
- reason：保留或剔除的原因

运行命令：
  uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers 24
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import tyro

# SONIC 的关键词列表（来源：gear_sonic/data_process/filter_and_copy_bones_data.py），
# 并补充了论文中列举的示例关键词。
# 使用下划线作为分隔符，以避免子字符串匹配造成误判（D16 计划）。
# 2026-10-21，为了使名称筛选后的数据量与论文一致，曾移除纯杂技类关键词。
# 但 e8 完整基准测试表明，论文实际上也会过滤这些动作：
# 保留纯杂技类关键词时，筛选后剩余 129,491 条；
# 移除这些关键词时，筛选后剩余 129,514 条。
# 因此，现已恢复这些关键词（参见 EXPERIMENTS e8）。
SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "box_hspu", "50cm_box_", "on_box",
    "fall_from", "on_1m", "form_box", "off_1m", "230m", "jump_over_obstacle_", "lift_crate_come_up_",
    "stair", "box_jump", "box_dips", "walking_on_edge", "push_obstacle",
]
SONIC_ACROBATICS = ["acrobatics_", "cartwheel", "handstand", "handstand_ff_", "safety_roll", "kozak_dance", "jump_to_shoulder_roll", "monkey_jump"]
PAPER_EXTRA = ["_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over"]
KEYWORDS = sorted(set(SONIC_KEYWORDS + SONIC_ACROBATICS + PAPER_EXTRA))

# 物理可行性筛选阈值（参考论文附录 E 及 D16 方案中的具体选择）。
ELEVATED_Z = 0.20  # 最低身体链接高度超过 20 cm
ELEVATED_S = 0.6 # 上述状态连续持续至少 0.6 秒
PENETRATE_Z_HARD = -0.12 # 任意一帧最低链接位置低于地面 12 cm
PENETRATE_Z_SOFT = -0.08 # 较轻的地下高度异常阈值，为 −8 cm
PENETRATE_FRAC = 0.80 # 超过 80% 的帧满足较轻穿透条件
FLOAT_Z = 0.02 # 静止站立时脚底距离地面超过 2 cm
FLOAT_S = 0.5 # 脚底悬空连续持续至少 0.5 秒
UPRIGHT_GZ = 0.9 # 判断骨盆是否基本直立
STAND_Z = (0.60, 0.85)  #骨盆站立高度范围
STILL_V = 0.1 # 骨盆线速度阈值
STILL_QD = 0.5 # 关节速度均方根阈值
FPS = 50


def name_filter(name: str) -> str | None:
    n = name.lower()
    for k in KEYWORDS:
        if k in n:
            return f"name:{k}"
    return None


def feasibility_filter(npz_path: str) -> str | None:
    with np.load(npz_path) as d:
        names = [str(x) for x in d["body_names"]] 
        pos = d["body_pos_w"]  # (T, 30, 3) 各时间帧中 30 个身体链接在世界坐标系下的位置
        quat = d["body_quat_w"]  # (T, 30, 4) wxyz 姿态四元数
        jv = d["joint_vel"]
        lin = d["body_lin_vel_w"]
    T = pos.shape[0]
    # 提取所有帧、所有身体链接的 z 坐标，再沿身体链接维度取最小值
    # 结果形状为 (T,)，表示每一帧中最低身体链接原点的高度
    zmin = pos[:, :, 2].min(1)
    # 获取左脚踝、右脚踝和骨盆在身体链接列表中的索引
    lf, rf, pel = names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link"), names.index("pelvis")
    # elevated support: zmin > 0.20 for >= 0.6 s
    if _longest_run(zmin > ELEVATED_Z) >= ELEVATED_S * FPS:
        return "elevated_support"
    if (zmin < PENETRATE_Z_HARD).any() or (zmin < PENETRATE_Z_SOFT).mean() > PENETRATE_FRAC:
        return "ground_penetration" # 地面穿透
    q = quat[:, pel]
    # gravity z-component in pelvis frame: -(R^T e_z)_z = -(1 - 2(x^2+y^2)) for wxyz quaternion
    gz = -(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    upright = gz < -UPRIGHT_GZ
    standing = (pos[:, pel, 2] > STAND_Z[0]) & (pos[:, pel, 2] < STAND_Z[1])
    still = (np.linalg.norm(lin[:, pel], axis=1) < STILL_V) & (np.sqrt((jv**2).mean(1)) < STILL_QD)
    gate = upright & standing & still
    if _longest_run(gate) >= FLOAT_S * FPS:
        # Exact lowest point of the foot collision capsules (paper: mj_geomDistance to the plane).
        foot_z = _foot_sole_height(npz_path, gate)
        floating = gate.copy()
        floating[gate] = foot_z > FLOAT_Z
        if _longest_run(floating) >= FLOAT_S * FPS:
            return "floating_feet"
    return None


_MODEL = None
_FOOT_GEOMS = None


def _foot_sole_height(npz_path: str, gate: np.ndarray) -> np.ndarray:
    """Lowest point over both feet's collision capsules, for the gated frames (returns len(gate.sum()),)."""
    global _MODEL, _FOOT_GEOMS
    import mujoco
    import re

    if _MODEL is None:
        from supermimic.data.motion_convert import load_g1_model

        _MODEL = load_g1_model()
        _FOOT_GEOMS = [g for g in range(_MODEL.ngeom) if re.match(r"^(left|right)_foot[1-7]_collision$", mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        assert len(_FOOT_GEOMS) == 14 and all(_MODEL.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in _FOOT_GEOMS)
    m = _MODEL
    d = mujoco.MjData(m)
    with np.load(npz_path) as z:
        names = [str(x) for x in z["body_names"]]
        pel = names.index("pelvis")
        root_pos = z["body_pos_w"][:, pel]
        root_quat = z["body_quat_w"][:, pel]
        jp = z["joint_pos"]
    idx = np.nonzero(gate)[0]
    out = np.empty(len(idx))
    radius = m.geom_size[_FOOT_GEOMS, 0]
    half = m.geom_size[_FOOT_GEOMS, 1]
    for k, t in enumerate(idx):
        d.qpos[:3] = root_pos[t]
        d.qpos[3:7] = root_quat[t]
        d.qpos[7:] = jp[t]
        mujoco.mj_kinematics(m, d)
        cz = d.geom_xpos[_FOOT_GEOMS, 2]
        axis_z = d.geom_xmat[_FOOT_GEOMS].reshape(-1, 3, 3)[:, 2, 2]  # world z of the capsule axis
        out[k] = (cz - half * np.abs(axis_z) - radius).min()
    return out


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _one(path: str) -> tuple[str, int, str | None]:
    name = Path(path).stem
    r = name_filter(name)
    frames = 0
    if r is None:
        r = feasibility_filter(path)
    with np.load(path) as d:
        frames = int(d["joint_pos"].shape[0])
    return name, frames, r


def main(npz_dir: Path, workers: int = 24, out: Path | None = None) -> None:
    files = sorted(str(p) for p in npz_dir.glob("*.npz"))
    print(f"{len(files)} clips")
    with mp.Pool(workers) as pool:
        rows = pool.map(_one, files, chunksize=64)
    out = out or npz_dir / "manifest_filtered.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frames", "keep", "reason"])
        for name, frames, r in rows:
            w.writerow([name, frames, int(r is None), r or ""])
    reasons = Counter(r for _, _, r in rows)
    kept = sum(1 for _, _, r in rows if r is None)
    name_removed = sum(1 for _, _, r in rows if r and r.startswith("name:"))
    feas_removed = sum(1 for _, _, r in rows if r and not r.startswith("name:"))
    hours = sum(fr for _, fr, r in rows if r is None) / FPS / 3600
    print(f"raw {len(rows)} -> after name filter {len(rows) - name_removed} (paper 129,667) -> after feasibility {kept} (paper 129,514); {hours:.2f} h kept (paper 264.56)")
    print("by reason:", dict(reasons.most_common(40)))
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
