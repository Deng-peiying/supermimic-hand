"""CPU MuJoCo stepping, independent of task rewards and reference motion."""

import mujoco
import numpy as np


class MujocoCPU:
    def __init__(self, model, control_dt):
        self.model = model
        self.data = mujoco.MjData(model)
        self.substeps = round(control_dt / model.opt.timestep)
        if self.substeps < 1 or not np.isclose(self.substeps * model.opt.timestep, control_dt):
            raise ValueError("Control interval must be an integer multiple of physics timestep")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self, control):
        self.data.ctrl[:] = control
        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
