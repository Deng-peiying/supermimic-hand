"""G1 free-base state ordering shared by analytic heads and model integration."""

from supermimic.core.state import StateLayout

NUM_JOINTS = 29
STATE_LAYOUT = StateLayout((
    ("root_pos", 3), ("root_quat", 4), ("joint_pos", NUM_JOINTS),
    ("root_lin_vel", 3), ("root_ang_vel", 3), ("joint_vel", NUM_JOINTS),
))


def split_state(state):
    return tuple(state[..., STATE_LAYOUT.index(name)] for name, _ in STATE_LAYOUT.fields)
