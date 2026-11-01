"""Lazy task entry points: importing pen spin never initializes mjlab/Warp."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class TaskSpec:
    name: str
    robot: str
    backend: str
    entrypoint: str

    def load(self):
        module, name = self.entrypoint.split(":")
        return getattr(import_module(module), name)


_TASKS: dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec):
    if spec.name in _TASKS:
        raise ValueError(f"Task already registered: {spec.name}")
    _TASKS[spec.name] = spec


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]
    except KeyError:
        raise ValueError(f"Unknown task {name!r}; available: {', '.join(_TASKS)}") from None


def list_tasks() -> tuple[TaskSpec, ...]:
    return tuple(_TASKS.values())


register_task(
    TaskSpec(
        "g1_tracking",
        "g1",
        "mjlab",
        "supermimic.tasks.motion_tracking.train:main",
    )
)
register_task(
    TaskSpec(
        "prima1_pen_spin",
        "prima1_left",
        "mujoco_warp (train) / mujoco_cpu (evaluate)",
        "supermimic.tasks.pen_spin.train:main",
    )
)
