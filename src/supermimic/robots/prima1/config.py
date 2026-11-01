"""Prima1 left-hand embodiment; task rewards do not encode joint counts."""

from supermimic.core.state import StateLayout

JOINTS = [
    f"{finger}{i}"
    for finger, count in (("FF", 4), ("MF", 4), ("RF", 4), ("LF", 5), ("TH", 5))
    for i in range(1, count + 1)
]
NUM_JOINTS = len(JOINTS)
MAX_TARGET_DELTA = 0.12
RESIDUAL_LIMIT = 0.15
STATE_LAYOUT = StateLayout(
    (
        ("joint_pos", NUM_JOINTS),
        ("joint_vel", NUM_JOINTS),
        ("object_pos", 3),
        ("object_quat", 4),
        ("object_lin_vel", 3),
        ("object_ang_vel", 3),
    )
)
