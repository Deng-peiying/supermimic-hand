"""基于 mjlab 的 SuperMimic 运动跟踪环境（计划阶段 2.3）。

与 mjlab 的 BeyondMimic 复现实现（mjlab.tasks.tracking）的区别：
  * 观测：遵循论文表 A.1，仅使用本体感知信息（不包含锚点位姿和基座线速度）；
    使用 10 帧历史观测（按特征项排列），另设独立的 goal 观测组，包含未来 K 帧参考状态；
  * 动作：采用 NoisyJointPositionAction（观测记录干净动作，仅执行时添加噪声）；
  * 指令：支持多条动作片段，重置时按回报采样，并在片段结束时触发超时；
  * 奖励：移除自碰撞奖励项，关节限位惩罚权重设为 -2；
  * 终止：根据骨盆高度及相对参考姿态的倾斜判断，不使用末端执行器终止条件；
    动作片段结束时触发超时；
  * 事件：不施加外部推力；每回合重新采样摩擦系数和编码器偏差；
    质心偏移在启动时采样；
  * 回放：ReplayRecorder 将状态转移数据持续写入 GPU 经验回放缓冲区。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as track_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from supermimic.data.recorder import ReplayRecorderCfg
from supermimic.envs.actions import NoisyJointPositionActionCfg
from supermimic.envs.mdp import observations as sm_obs
from supermimic.envs.mdp import terminations as sm_term
from supermimic.envs.mdp.commands import MultiClipMotionCommandCfg

class QuietSimulationCfg(SimulationCfg):
    """关闭 MuJoCo Warp 每步的溢出警告打印。该选项是内核编译时常量，必须在捕获 CUDA Graph 前设置。
    """

    def apply_wp_opt(self, wp_opt) -> None:  # 配置 Warp 选项
        super().apply_wp_opt(wp_opt)  # 先执行父类的配置方法
        wp_opt.warn_overflow = 0  # 关闭溢出警告打印


TRACKED_BODIES = (  # 运动跟踪的身体部位（14个）
    "pelvis",  # 骨盆
    "left_hip_roll_link",  # 左髋侧摆连杆
    "left_knee_link",  # 左膝连杆
    "left_ankle_roll_link",  # 左踝侧摆连杆
    "right_hip_roll_link",  # 右髋侧摆连杆
    "right_knee_link",  # 右膝连杆
    "right_ankle_roll_link",  # 右踝侧摆连杆
    "torso_link",  # 躯干连杆
    "left_shoulder_roll_link",  # 左肩侧摆连杆
    "left_elbow_link",  # 左肘连杆
    "left_wrist_yaw_link",  # 左腕偏航连杆
    "right_shoulder_roll_link",  # 右肩侧摆连杆
    "right_elbow_link",  # 右肘连杆
    "right_wrist_yaw_link",  # 右腕偏航连杆
)
ANCHOR_BODY = "torso_link"  # 锚定参考部位：躯干
ROOT_BODY = "pelvis"  # 根部参考部位：骨盆
FOOT_GEOM_REGEX = r"^(left|right)_foot[1-7]_collision$"  # 匹配左右脚的碰撞几何体
# Root 用于描述机器人整体的根部运动状态，Anchor 用于对齐参考动作与当前机器人运动

@dataclass
class SuperMimicEnvCfgArgs:
    npz_files: list[str]  # NPZ 动作数据文件列表
    num_envs: int = 4096  # 并行环境数量
    future_frames: int = 10  # 未来参考帧数
    history_length: int = 10  # 历史观测长度
    episode_length_s: float = 10.0  # 单回合最大时长（秒）
    action_noise_std: float = 0.1  # 动作噪声标准差
    obs_noise: bool = True  # 启用观测噪声
    reset_noise: bool = True  # 启用重置扰动
    domain_randomization: bool = True  # 启用域随机化
    sampling_mode: str = "return"  # 采样模式：按回报 | 均匀 | 起始位置
    replay_capacity: int = 1000  # 经验回放容量
    record_transitions: bool = True  # 记录状态转移数据
    seed: int | None = 0  # 随机种子
    root_height_threshold: float = 0.25  # 根部高度终止阈值
    root_tilt_threshold: float = 0.8  # 根部倾斜终止阈值
    joint_limit_weight: float = -2.0  # 关节限位惩罚权重
    bin_frames: int = 50  # 每个采样区间的帧数
    compact_library: bool = False  # 启用紧凑动作库
    anchor_obs: bool = False  # 将 BeyondMimic 的锚点相对位姿（9维）加入目标观测组（D5备选方案）
    debug_vis: bool = False  # 在可视化窗口绘制参考机器人（scripts/play.py）
    extra_command_kwargs: dict = field(default_factory=dict)  # 额外指令参数


def make_env_cfg(a: SuperMimicEnvCfgArgs) -> ManagerBasedRlEnvCfg: # 生成仿真环境配置
    # ---------------------------------------------------------------- observations
    noise = (lambda lo, hi: Unoise(n_min=lo, n_max=hi)) if a.obs_noise else (lambda lo, hi: None)
    policy_terms = { # 93
        "joint_pos": ObservationTermCfg(func=base_mdp.joint_pos_rel, params={"biased": True}, noise=noise(-0.01, 0.01)),  # 相对关节位置 29
        "joint_vel": ObservationTermCfg(func=base_mdp.joint_vel_rel, noise=noise(-0.5, 0.5)),  # 相对关节速度  29
        "base_ang_vel": ObservationTermCfg(func=base_mdp.base_ang_vel, noise=noise(-0.2, 0.2)),  # 基座角速度 3
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity, noise=noise(-0.05, 0.05)),  # 投影重力方向 3
        "actions": ObservationTermCfg(func=base_mdp.last_action),  # 上一步动作 29 
    }
    observations = {
        "policy": ObservationGroupCfg( # 策略观测组 93维
            # 关节位置、关节速度、基座角速度、投影重力、历史动作
            terms=policy_terms, # 关节、IMU、历史动作
            concatenate_terms=True, # 拼接观测特征
            enable_corruption=a.obs_noise, # 启用观测噪声
            history_length=a.history_length,  # 10帧历史，10×93=930维
            flatten_history_dim=True, # 展平时间维度
        ),
        "goal": ObservationGroupCfg( # 目标观测组 68维
            # 关节位置29维、关节速度29维、重力3维、机体系角速度3维、水平线速度3维、高度1维
            terms={
                "future_frames": ObservationTermCfg(func=sm_obs.goal_future_frames, params={"command_name": "motion"}), # 获取未来参考运动
                **( # 展开可选锚点观测
                    {
                        "anchor_pos": ObservationTermCfg(func=track_mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=noise(-0.25, 0.25)),# 锚点位置噪声
                        "anchor_ori": ObservationTermCfg(func=track_mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=noise(-0.05, 0.05)),# 锚点姿态噪声
                    }
                    if a.anchor_obs # 启用时添加9维锚点观测
                    else {}
                ),
            },
            concatenate_terms=True, # 拼接目标特征
            enable_corruption=a.obs_noise, # 启用观测噪声
        ),
    }

    # ---------------------------------------------------------------- actions
    actions = {
        "joint_pos": NoisyJointPositionActionCfg( # 带噪声的关节位置控制 
            entity_name="robot", # 控制对象
            actuator_names=(".*",), # 匹配所有执行器
            scale=G1_ACTION_SCALE, # 动作缩放系数
            use_default_offset=True, # 叠加默认关节位置
            noise_std=a.action_noise_std, # 执行动作噪声标准差
        )
    }

    # ---------------------------------------------------------------- commands
    pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01), "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)} # 根部重置位姿扰动范围
    vel_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)} # 根部重置速度扰动范围
    commands = {
        "motion": MultiClipMotionCommandCfg( # 多片段参考运动配置
            npz_files=[str(f) for f in a.npz_files], # 参考动作文件
            anchor_body_name=ANCHOR_BODY, # 锚点身体：躯干
            root_body_name=ROOT_BODY, # 根部身体：骨盆
            body_names=TRACKED_BODIES, # 14个跟踪身体部位
            entity_name="robot", # 目标机器人
            future_frames=a.future_frames, # 未来参考帧数
            bin_frames=a.bin_frames, # 每个采样区间的帧数
            pose_range=pose_range if a.reset_noise else {}, # 启用位姿重置扰动
            velocity_range=vel_range if a.reset_noise else {}, # 启用速度重置扰动
            joint_position_range=(-0.1, 0.1) if a.reset_noise else (0.0, 0.0), # 关节位置重置扰动
            sampling_mode=a.sampling_mode,  # type: ignore[arg-type]
            compact_library=a.compact_library, # 紧凑动作库开关
            debug_vis=a.debug_vis, # 参考动作可视化
            **a.extra_command_kwargs, # 展开额外指令参数
        )
    }

    # ---------------------------------------------------------------- events (plan D8)
    events: dict[str, EventTermCfg] = {}
    if a.domain_randomization: # 启用域随机化
        events["base_com"] = EventTermCfg(
            mode="startup", # 仿真启动时执行
            func=dr.body_com_offset, # 质心偏移
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), # 躯干刚体
                "operation": "add", # 在原始值上叠加偏移
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},  # 三轴质心偏移范围（米）
            },
        )
        events["encoder_bias"] = EventTermCfg(
            mode="reset", # 每次重置时执行
            func=dr.encoder_bias, # 关节编码器偏差
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)}, # 偏差范围（弧度）
        )
        events["foot_friction"] = EventTermCfg(
            mode="reset", # 每次重置时执行
            func=dr.geom_friction, # 足部摩擦随机化
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX), # 足部碰撞体
                "operation": "abs",  # 直接设置摩擦参数
                "ranges": (0.3, 1.2), # 摩擦参数采样范围
                "shared_random": True, # 共享随机采样值
            },
        )
    else:
        # 评估协议：足部摩擦固定为1.0，不进行其他随机化
        events["foot_friction"] = EventTermCfg(
            mode="startup", 
            func=dr.geom_friction, # 设置足部摩擦
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_REGEX),
                "operation": "abs", # 直接设置摩擦参数
                "ranges": (1.0, 1.0), # 固定摩擦参数
                "shared_random": True,
            },
        )

    # ---------------------------------------------------------------- rewards (Table A.2)
    rewards = {
        "motion_global_root_pos": RewardTermCfg(func=track_mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}),  # 全局锚点位置跟踪
        "motion_global_root_ori": RewardTermCfg(func=track_mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}),  # 全局锚点姿态跟踪
        "motion_body_pos": RewardTermCfg(func=track_mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}),  # 身体相对位置跟踪
        "motion_body_ori": RewardTermCfg(func=track_mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}),  # 身体相对姿态跟踪
        "motion_body_lin_vel": RewardTermCfg(func=track_mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}),  # 身体线速度跟踪
        "motion_body_ang_vel": RewardTermCfg(func=track_mdp.motion_global_body_angular_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 3.14}),  # 身体角速度跟踪
        "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),  # 动作变化惩罚
        "joint_limit": RewardTermCfg(func=base_mdp.joint_pos_limits, weight=a.joint_limit_weight, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),  # 关节限位惩罚
    }

    # ---------------------------------------------------------------- terminations
    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),  # 达到回合最大时长
        "clip_end": TerminationTermCfg(func=sm_term.clip_end, params={"command_name": "motion"}, time_out=True),  # 参考动作片段结束
        "root_height": TerminationTermCfg(func=sm_term.bad_root_height, params={"command_name": "motion", "threshold": a.root_height_threshold}),  # 根部高度误差超限
        "root_tilt": TerminationTermCfg(func=sm_term.bad_root_tilt, params={"command_name": "motion", "threshold": a.root_tilt_threshold}),  # 根部倾斜误差超限
    }

    recorders = {}  # 初始化记录器
    if a.record_transitions:  # 启用状态转移记录
        recorders["replay"] = ReplayRecorderCfg(params={"capacity": a.replay_capacity, "foot_geom_regex": FOOT_GEOM_REGEX, "com_body_name": "torso_link"})  # 配置经验回放记录器

    cfg = ManagerBasedRlEnvCfg(  # 组装完整环境配置
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=a.num_envs, entities={"robot": get_g1_robot_cfg()}),  # 平地、并行环境、G1机器人
        observations=observations,  # 观测配置
        actions=actions,  # 动作配置
        commands=commands,  # 参考运动指令
        events=events,  # 域随机化事件
        rewards=rewards,  # 奖励配置
        terminations=terminations,  # 终止条件
        recorders=recorders,  # 经验记录器
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link", distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),  # 以躯干为中心的可视化相机
        sim=QuietSimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),  # MuJoCo物理仿真配置
        decimation=4,  # 每4个物理步更新一次控制
        episode_length_s=a.episode_length_s,  # 最大回合时长
        seed=a.seed,  # 环境随机种子
    )
    return cfg


def make_env(a: SuperMimicEnvCfgArgs, device: str = "cuda:0") -> ManagerBasedRlEnv:  # 创建仿真环境
    from mjlab.utils.torch import configure_torch_backends  # 导入PyTorch后端配置函数

    configure_torch_backends()  # 配置PyTorch计算后端
    return ManagerBasedRlEnv(make_env_cfg(a), device=device)  # 创建并返回强化学习环境
