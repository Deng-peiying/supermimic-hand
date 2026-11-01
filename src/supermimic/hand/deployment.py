"""离线推理适配器；没有导入 SDK、没有连接或发送硬件命令。

SDK 关节的顺序、方向、零点必须由实测配置提供，不能猜测 22 个 URDF 关节
与硬件独立自由度一一对应。耦合/欠驱动机构需另写经过标定的适配器。
"""

import json
from pathlib import Path

import numpy as np
import torch

from .data import JOINTS
from .networks import append_history, initial_history


class OfflineController:
    def __init__(self, bundle, mapping):
        bundle = Path(bundle)
        self.schema = json.loads((bundle / "actor_schema.json").read_text(encoding="utf-8"))
        if self.schema["joint_names"] != JOINTS:
            raise ValueError("部署关节顺序不匹配")
        self.actor = torch.jit.load(str(bundle / "actor.ts"), map_location="cpu").eval()
        with np.load(bundle / "joint_reference.npz", allow_pickle=False) as data:
            self.reference = data["q"].copy()
            self.dt = float(data["dt"])
        # q_urdf(rad) = sdk_value * scale_to_rad + offset_rad。
        # mapping 为按 JOINTS 排列的 22 项；每项指定唯一 sdk_name。
        if len(mapping) != 22 or [m["urdf_joint"] for m in mapping] != JOINTS:
            raise ValueError("需要明确的 22 关节硬件映射")
        self.names = [m["sdk_name"] for m in mapping]
        self.scale = np.array([m["scale_to_rad"] for m in mapping])
        self.offset = np.array([m["offset_rad"] for m in mapping])
        if (
            len(set(self.names)) != 22
            or not np.isfinite(self.scale).all()
            or np.any(self.scale == 0)
            or not np.isfinite(self.offset).all()
        ):
            raise ValueError("无效 SDK 映射；耦合关节不允许假装独立")
        self.history = None
        self.frame = 0
        self.last_time = None
        self.stopped = False

    @torch.no_grad()
    def step(self, sdk_positions, timestamp):
        if self.stopped:
            raise RuntimeError("控制器已停止，需要人工复位并重新初始化")
        try:
            values = np.array([sdk_positions[n] for n in self.names])
            q = values * self.scale + self.offset
            if not np.isfinite(q).all() or not np.isfinite(timestamp):
                raise ValueError("反馈不是有限数")
            if (
                self.last_time is not None
                and not 0.5 * self.dt <= timestamp - self.last_time <= 1.5 * self.dt
            ):
                raise TimeoutError("反馈周期异常：停止生成新动作，由外部看门狗处理")
            qt = torch.tensor(q[None], dtype=torch.float32)
            limits = self.actor.limits.numpy()
            if np.any(q < limits[:, 0] - 0.05) or np.any(q > limits[:, 1] + 0.05):
                raise ValueError("关节反馈超出 URDF 限位")
            if self.history is None:
                if np.max(np.abs(q - self.reference[0])) > 0.15:
                    raise ValueError("需先缓慢到达首帧放笔姿态；本接口不执行启动运动")
                self.history = initial_history(qt)
            else:
                self.history = append_history(
                    self.history, qt, self.last_q, self.last_target, timestamp - self.last_time
                )
            length = self.schema["stage_length"]
            if self.frame >= length:
                raise StopIteration("单片段结束，不允许直接首尾跳接循环")
            ref_frame = 0 if self.schema["stage"] == "hold" else self.frame + 1
            phase = 0.0 if self.schema["stage"] == "hold" else self.frame / length
            target = self.actor(
                self.history, torch.tensor(self.reference[None, ref_frame]), torch.tensor([[phase]])
            )
            if not torch.isfinite(target).all():
                raise FloatingPointError("策略输出非有限")
            self.last_q, self.last_target, self.last_time = qt, target, timestamp
            self.frame += 1
            sdk_target = (target[0].numpy() - self.offset) / self.scale
            # 返回值供离线检查；集成时需 SDK 回报实际执行目标并核实限幅一致。
            return dict(zip(self.names, sdk_target.tolist()))
        except Exception:
            self.stopped = True
            raise
