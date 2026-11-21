"""Load configs/config.yaml and resolve paths against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/scorecard/config.py -> parents[2] is the repo root. This assumes the
# package is installed in editable mode (pip install -e .), as in the README.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML config into a dict."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)  # safe_load: never executes arbitrary YAML tags
    if not isinstance(cfg, dict):
        raise TypeError(f"Config at {path} is empty or not a mapping")
    return cfg


def resolve_path(relative: Path | str) -> Path:
    """Turn a repo-relative path from the config into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
