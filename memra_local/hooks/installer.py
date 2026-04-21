"""Install/uninstall Memra auto-capture hook in Claude Code settings.json."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
MEMRA_HOOK_ID = "memra-auto-capture"


def install_session_end_hook(settings_path: Path | None = None) -> bool:
    """Add Memra capture hook to Claude Code SessionEnd.

    Args:
        settings_path: Override path for testing. Defaults to ~/.claude/settings.json.

    Returns:
        True if hook was installed, False if already present.

    Raises:
        FileNotFoundError: If ~/.claude/ directory does not exist (Claude Code not installed).
    """
    if settings_path is None:
        settings_path = CLAUDE_SETTINGS

    if not settings_path.parent.exists():
        raise FileNotFoundError(
            f"Claude Code not installed ({settings_path.parent} not found)"
        )

    # Read existing settings or start fresh
    settings: dict = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

    hooks = settings.setdefault("hooks", {})
    session_end: list = hooks.setdefault("SessionEnd", [])

    # Check if already installed (idempotent)
    for group in session_end:
        for h in group.get("hooks", []):
            if MEMRA_HOOK_ID in h.get("command", ""):
                return False

    # Resolve full path to memra binary
    memra_bin = shutil.which("memra")
    memra_cmd = memra_bin if memra_bin else "memra"

    session_end.append(
        {
            "hooks": [
                {
                    "type": "command",
                    "command": f"{memra_cmd} capture --hook-id {MEMRA_HOOK_ID}",
                    "timeout": 30,
                }
            ]
        }
    )

    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    return True


def uninstall_session_end_hook(settings_path: Path | None = None) -> bool:
    """Remove Memra capture hook from Claude Code SessionEnd.

    Args:
        settings_path: Override path for testing. Defaults to ~/.claude/settings.json.

    Returns:
        True if hook was removed, False if not found.
    """
    if settings_path is None:
        settings_path = CLAUDE_SETTINGS

    if not settings_path.exists():
        return False

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    session_end: list = settings.get("hooks", {}).get("SessionEnd", [])

    original_len = len(session_end)
    session_end[:] = [
        group
        for group in session_end
        if not any(
            MEMRA_HOOK_ID in h.get("command", "") for h in group.get("hooks", [])
        )
    ]

    if len(session_end) == original_len:
        return False

    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    return True
