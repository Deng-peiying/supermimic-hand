"""按片段保存边界的参考集合；全局帧号只用于索引，不代表连续运动。"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .clip import Clip


class ClipCollection:
    def __init__(self, folder, asset):
        self.folder = Path(folder)
        manifest = json.loads((self.folder / 'dataset.json').read_text(encoding='utf-8'))
        self.clips = []
        for row in manifest['clips']:
            path = (self.folder / row['path']).resolve()
            if not path.is_relative_to(self.folder.resolve()):
                raise ValueError('片段路径超出数据集目录')
            clip = Clip(path, asset)
            if clip.digest != row['sha256']:
                raise ValueError('数据集清单与片段哈希不匹配')
            self.clips.append(clip)
        if not self.clips:
            raise ValueError('空数据集')
        first = self.clips[0]
        self.dt, self.gravity, self.manifest = first.dt, first.gravity, first.manifest
        for clip in self.clips:
            if (clip.manifest['model_sha256'] != first.manifest['model_sha256']
                    or clip.manifest['asset_sha256'] != first.manifest['asset_sha256']
                    or not np.allclose(clip.gravity, self.gravity) or not np.isclose(clip.dt, self.dt)):
                raise ValueError('多片段必须共用模型、重力和控制周期')
        lengths = np.array([len(c.q) for c in self.clips])
        self.starts = np.r_[0, np.cumsum(lengths)[:-1]]
        self.ends = self.starts + lengths - 1
        self.q = np.concatenate([c.q for c in self.clips])
        self.qd = np.concatenate([c.qd for c in self.clips])
        self.pose = np.concatenate([c.pose for c in self.clips])

    def model(self):
        return self.clips[0].model()

    def provenance(self):
        clips = [c.provenance() for c in self.clips]
        digest = hashlib.sha256(json.dumps(clips, sort_keys=True).encode()).hexdigest()
        return dict(dataset_sha256=digest, clips=clips, dt=self.dt, side='left',
                    physics_validated=False, dynamics_calibrated=False)


def load_reference(folder, asset):
    return ClipCollection(folder, asset) if (Path(folder) / 'dataset.json').exists() else Clip(folder, asset)
