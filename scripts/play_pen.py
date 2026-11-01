"""本地浏览器查看 Prima1 策略及参考；v3/v4 导出接口显式区分。"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
import viser

from supermimic.tasks.pen_spin.data import Clip
from supermimic.tasks.pen_spin.envs.env import HandEnv
from supermimic.tasks.pen_spin.envs.mdp.observations import append_history, initial_history
from supermimic.tasks.pen_spin.policy.reference import future_joint_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    torch.set_num_threads(2)
    schema = json.loads((args.bundle / 'actor_schema.json').read_text())
    actor = torch.jit.load(str(args.bundle / 'actor.ts')).eval()
    clip = Clip()
    if schema['clip_sha256'] != clip.provenance()['clip_sha256']:
        raise ValueError('Actor与参考片段不匹配')
    env = HandEnv(clip, schema['stage'])
    reference_data = mujoco.MjData(env.model)
    reference_q = torch.tensor(clip.q)
    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## Prima1 左手转笔\n左：自由笔策略控制；右：几何参考。\n单片段 clip001，CPU MuJoCo；不代表5090接触后端完全一致。')
    restart = server.gui.add_button('Play selected / Restart')
    paused = server.gui.add_checkbox('Pause', initial_value=False)
    step = server.gui.add_button('Step')
    speed = server.gui.add_slider('Speed', min=0.1, max=1.0, step=0.1, initial_value=0.5)
    loop = server.gui.add_checkbox('Auto replay from first frame', initial_value=True)
    status = server.gui.add_markdown('Loading')
    commands = {'restart': False, 'step': False}
    restart.on_click(lambda _: commands.update(restart=True))
    step.on_click(lambda _: commands.update(step=True))
    offsets = [np.array([-0.14, 0., 0.]), np.array([0.14, 0., 0.])]
    handles = []
    model = env.model
    for side in range(2):
        group = []
        for g in range(model.ngeom):
            kind, size = model.geom_type[g], model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                mesh = trimesh.Trimesh(model.mesh_vert[va:va+vn], model.mesh_face[fa:fa+fn], process=False)
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                mesh = trimesh.creation.cylinder(radius=size[0], height=2*size[1])
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                mesh = trimesh.creation.capsule(radius=size[0], height=2*size[1])
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                mesh = trimesh.creation.box(extents=2*size)
            else:
                continue
            color = tuple((model.geom_rgba[g,:3] * 255).astype(int)) if side == 0 else (90, 170, 230)
            handle = server.scene.add_mesh_simple(f'/view{side}/geom{g}', vertices=mesh.vertices,
                                                  faces=mesh.faces, color=color)
            group.append((g, handle))
        handles.append(group)

    @server.on_client_connect
    def camera(client):
        client.camera.position = (0.4, -0.5, 0.45)
        client.camera.look_at = (0., 0., 0.06)

    def reset():
        state = env.reset()
        return state, initial_history(torch.tensor(state[None, :22]))

    state, history = reset()
    finished = False
    restart_at = None
    print(f'READY http://localhost:{args.port} interface={schema.get("interface_version", 3)}', flush=True)
    while True:
        tick = time.monotonic()
        if commands['restart'] or (restart_at is not None and tick >= restart_at and loop.value and not paused.value):
            state, history = reset()
            finished, restart_at = False, None
            commands['restart'] = False
        if not finished and (not paused.value or commands['step']):
            frame = env.frame
            if schema.get('interface_version', 3) == 4:
                goal = future_joint_reference(reference_q, torch.tensor([frame]), env.length, env.stage)
            else:
                goal = reference_q[None, frame if env.stage == 'hold' else frame + 1]
            with torch.no_grad():
                action = actor(history, goal, torch.tensor([[frame / env.length]]))[0].numpy()
            next_state, finished, info = env.step(action)
            history = append_history(history, torch.tensor(next_state[None,:22]),
                                     torch.tensor(state[None,:22]), torch.tensor(env.previous[None]), clip.dt)
            state = next_state
            status.content = (f'Frame **{env.frame}/{env.length}** · 笔位置误差 **{info["position_error_m"]*1000:.1f} mm**'
                              + (' · **距离失败，2秒后重播**' if info['dropped'] else ' · 参考末帧' if finished else ''))
            if finished:
                restart_at = tick + 2
        commands['step'] = False
        reference_data.qpos[env.qa] = clip.q[env.frame]
        reference_data.qpos[env.pq:env.pq+7] = clip.pose[env.frame]
        mujoco.mj_forward(model, reference_data)
        mujoco.mj_forward(model, env.data)
        with server.atomic():
            for side, data in enumerate((env.data, reference_data)):
                for g, handle in handles[side]:
                    quat = np.empty(4)
                    mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
                    handle.position = data.geom_xpos[g] + offsets[side]
                    handle.wxyz = quat
        time.sleep(max(0.001, clip.dt / speed.value - (time.monotonic() - tick)))


if __name__ == '__main__':
    main()
