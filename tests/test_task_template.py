"""A renamed task must resolve its own modules, not the original template."""

import ast
import importlib
import shutil
import sys
from pathlib import Path

import pytest


def test_motion_tracking_can_be_imported_under_another_name(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src/supermimic/tasks/motion_tracking"
    package = "copied_tracking_task"
    shutil.copytree(source, tmp_path / package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        config = importlib.import_module(f"{package}.config")
        cfg = config.TrainConfig()
        assert type(cfg.policy).__module__ == f"{package}.policy.nets"
        assert type(cfg.world_model).__module__ == f"{package}.world_model.world_model"
        assert type(cfg.policy_trainer).__module__ == f"{package}.imagination.policy_trainer"
        assert type(cfg.rewards).__module__ == f"{package}.reward_cfg"
        for path in source.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("supermimic.tasks.motion_tracking")
                elif isinstance(node, ast.Import):
                    assert all(
                        not alias.name.startswith("supermimic.tasks.motion_tracking")
                        for alias in node.names
                    )
    finally:
        for name in list(sys.modules):
            if name == package or name.startswith(package + "."):
                del sys.modules[name]


def test_pen_spin_can_be_imported_under_another_name(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src/supermimic/tasks/pen_spin"
    package = "copied_pen_task"
    shutil.copytree(source, tmp_path / package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        config = importlib.import_module(f"{package}.config")
        args = config.TrainConfig()
        assert args == config.TrainConfig(**vars(config.make_parser().parse_args(["train"])))
        policy = importlib.import_module(f"{package}.policy.nets")
        model = importlib.import_module(f"{package}.world_model.world_model")
        loop = importlib.import_module(f"{package}.training.loop")
        entry = importlib.import_module(f"{package}.train")
        assert loop.Actor is policy.Actor
        assert loop.WorldModel is model.WorldModel
        assert entry.train is loop.train
        cfg = importlib.import_module(f"{package}.envs.pen_spin_env_cfg").make_env_cfg()
        assert all(
            term.func.__module__.startswith(package + ".envs.mdp.") for term in cfg.rewards.values()
        )
    finally:
        for name in list(sys.modules):
            if name == package or name.startswith(package + "."):
                del sys.modules[name]


@pytest.mark.parametrize("task", ["motion_tracking", "pen_spin"])
def test_task_dependency_direction(task):
    """复制任务不会回指原任务；基础职责不依赖 CLI 或训练调度。"""
    root = Path(__file__).resolve().parents[1] / "src/supermimic/tasks" / task
    prefix = f"supermimic.tasks.{task}"
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        package = ".".join([prefix, *relative.parts[:-1]])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    modules = [
                        importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    ]
                else:
                    # 任务内部用相对导入，复制任务不需要逐个改绝对路径。
                    assert not (node.module or "").startswith(prefix), path
                    modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                assert not any(m.startswith(prefix) for m in modules), path
            for module in modules:
                if module.startswith("supermimic.tasks."):
                    assert module == prefix or module.startswith(prefix + "."), (path, module)
                if relative.parts[0] in {
                    "envs",
                    "data",
                    "policy",
                    "world_model",
                    "imagination",
                    "eval",
                    "deployment",
                }:
                    assert module != prefix + ".train", (path, module)
                    assert not module.startswith(prefix + ".training"), (path, module)
