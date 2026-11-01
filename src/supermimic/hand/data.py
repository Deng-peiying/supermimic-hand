"""锁定人工检查过的候选；资产路径可跨 Windows / WSL，不修改原文件。"""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

CLIP_SHA = "6833ef6510be13abd904096d8ce592bb112411d2c4ae83b2d887009fa2b8b619"
JOINTS = [
    f"{f}{i}"
    for f, n in [("FF", 4), ("MF", 4), ("RF", 4), ("LF", 5), ("TH", 5)]
    for i in range(1, n + 1)
]
DEFAULT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CLIP = DEFAULT_ROOT / "datasets/prima1_left_pen_spin/candidates/round3_kinematic_clip001"
DEFAULT_ASSET = DEFAULT_ROOT / "Assets-repo/xynova/Prima1/left"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"文件校验失败，需重新审查: {path}")


class Clip:
    def __init__(self, folder=DEFAULT_CLIP, asset=DEFAULT_ASSET):
        self.folder, self.asset = Path(folder), Path(asset)
        checked(self.folder / "candidate.npz", CLIP_SHA)
        review = json.loads((self.folder / "review_status.json").read_text(encoding="utf-8"))
        if review.get("candidate_sha256") != CLIP_SHA or not review.get("training_allowed"):
            raise ValueError("此候选没有与文件哈希绑定的训练许可")
        manifest = json.loads((self.folder / "asset_manifest.json").read_text(encoding="utf-8"))
        checked(self.asset / "prima_left_after_1.1.urdf", manifest["asset_sha256"])
        checked(self.folder / "prima_left_review.xml", manifest["model_sha256"])
        for path, digest in manifest["meshes"].items():
            checked(self.asset / "meshes" / Path(path.replace("\\", "/")).name, digest)
        with np.load(self.folder / "candidate.npz", allow_pickle=False) as data:
            self.q = data["q_ref"].astype(np.float32)
            self.qd = data["qd_ref"].astype(np.float32)
            self.pose = data["object_pose"].astype(np.float32)
            times = data["timestamps"]
            if list(data["joint_names"].astype(str)) != JOINTS:
                raise ValueError("关节顺序错误")
        self.dt = float(np.median(np.diff(times)))
        if not np.allclose(np.diff(times), 0.02) or self.q.shape != (369, 22):
            raise ValueError("仅支持审查过的 369 帧、50 Hz 片段")
        if not all(np.isfinite(a).all() for a in (self.q, self.qd, self.pose)):
            raise ValueError("参考数据包含非有限数")
        if not np.allclose(np.linalg.norm(self.pose[:, 3:], axis=-1), 1, atol=1e-4):
            raise ValueError("笔四元数必须归一化，顺序 wxyz")
        # 固定四元数符号，避免世界模型把 q 和 -q 当成不连续跳变。
        for i in range(1, len(self.pose)):
            if self.pose[i, 3:] @ self.pose[i - 1, 3:] < 0:
                self.pose[i, 3:] *= -1
        metadata = json.loads((self.folder / "retarget_metadata.json").read_text(encoding="utf-8"))
        self.gravity = 9.81 * np.array(metadata["target_anatomy"])[:, 2]
        self.manifest = manifest

    def model(self):
        root = ET.parse(self.folder / "prima_left_review.xml").getroot()
        for mesh in root.findall("./asset/mesh"):
            mesh.set("file", str((self.asset / "meshes" / Path(mesh.get("file")).name).resolve()))
        root.find("option").set("gravity", " ".join(map(str, self.gravity)))
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def provenance(self):
        return {
            "clip_sha256": CLIP_SHA,
            "asset_sha256": self.manifest["asset_sha256"],
            "model_sha256": self.manifest["model_sha256"],
            "joint_names": JOINTS,
            "human_geometry_accepted": True,
            "authorization": "用户：我看你重定向候选里没有问题；随后授权单片段 mimic 实现和训练验证。",
            "scope": "仅 round3 单片段几何参考实验，不是物理成功或真机部署验收",
            "physics_validated": False,
            "dynamics_calibrated": False,
            "dt": self.dt,
            "quaternion": "wxyz",
            "side": "left",
        }
