"""Scope resolution and configuration loading for memra-local."""

from __future__ import annotations

from pathlib import Path

import yaml


_DEFAULT_CONFIG = {
    "port": 8765,
    "host": "127.0.0.1",
    "scope": "auto",
}


def resolve_storage_dir(scope: str, cwd: Path | None = None) -> Path:
    """Resolve the storage directory for a given scope.

    Args:
        scope: "global" or "project" (use detect_scope for "auto")
        cwd: Current working directory (used for "project" scope)

    Returns:
        Path to the storage directory

    Raises:
        ValueError: If scope is not "global" or "project"
    """
    if scope == "global":
        return Path.home() / ".memra" / "global"
    elif scope == "project":
        base = cwd or Path.cwd()
        return base / ".memra"
    else:
        raise ValueError(f"Unknown scope: {scope!r}. Use 'global' or 'project'.")


def detect_scope(cwd: Path | None = None) -> str:
    """Auto-detect scope based on whether .memra/ exists in cwd.

    Args:
        cwd: Directory to check (defaults to current working directory)

    Returns:
        "project" if .memra/ exists in cwd, otherwise "global"
    """
    base = cwd or Path.cwd()
    if (base / ".memra").is_dir():
        return "project"
    return "global"


def resolve_scope(scope: str, cwd: Path | None = None) -> str:
    """Resolve scope string, handling "auto" by calling detect_scope.

    Args:
        scope: "global", "project", or "auto"
        cwd: Directory context for auto-detection

    Returns:
        Resolved scope: "global" or "project"
    """
    if scope == "auto":
        return detect_scope(cwd=cwd)
    return scope


def load_config(config_path: Path | None = None) -> dict:
    """Load config from YAML file, falling back to defaults.

    Args:
        config_path: Path to config.yaml file. If None, uses ~/.memra/config.yaml

    Returns:
        Configuration dictionary with all required keys
    """
    defaults = dict(_DEFAULT_CONFIG)

    path = config_path or (Path.home() / ".memra" / "config.yaml")

    if path.is_file():
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        defaults.update(user_config)

    return defaults
