"""YAML loading with portable project-root expansion."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _expand_project_root(value: Any, project_root: str) -> Any:
    if isinstance(value, dict):
        return {key: _expand_project_root(item, project_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_project_root(item, project_root) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value.replace("PROJECT_ROOT", project_root))
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    default_root = config_path.parents[2]
    project_root = str(Path(os.environ.get("PROJECT_ROOT", str(default_root))).resolve())
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _expand_project_root(data, project_root)
