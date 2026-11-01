"""已锁定 MuJoCo Warp 版本的凸接触容量适配。

该版本未公开 EPA horizon 容量配置，只在 collision_convex 分配临时数组时读取
模块常量。扩大实际工作区，不屏蔽 overflow，也不改碰撞几何/摩擦/求解公式。
必须在本进程创建任何 Warp 碰撞图之前调用；不修改已安装依赖文件。
"""


def configure_epa_horizon(edges=128):
    from mujoco_warp._src import collision_convex

    if not hasattr(collision_convex, "MJ_MAX_EPAHORIZON"):
        raise RuntimeError("MuJoCo Warp API 已变更，请重新验证 EPA 容量适配")
    collision_convex.MJ_MAX_EPAHORIZON = max(collision_convex.MJ_MAX_EPAHORIZON, edges)
    return collision_convex.MJ_MAX_EPAHORIZON
