"""YAML flat-file storage with atomic writes."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml


class FlatFileStore:
    """Flat-file store that writes YAML files atomically using tempfile + rename."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()

    def _validate_namespace(self, namespace: str) -> None:
        if not namespace or any(part in namespace for part in ("..", "/", "\\", "\x00")):
            raise ValueError("invalid namespace")

    def _resolve_path(self, storage_path: str) -> Path:
        full_path = (self._base_dir / storage_path).resolve()
        if self._base_dir != full_path and self._base_dir not in full_path.parents:
            raise ValueError("path escapes storage root")
        return full_path

    def write(self, namespace: str, memory_id: str, data: dict) -> str:
        """Write memory data to a YAML file atomically.

        Args:
            namespace: Subdirectory namespace (e.g., "default")
            memory_id: Unique memory identifier
            data: Dictionary to serialize as YAML

        Returns:
            Relative storage path: "{namespace}/{memory_id}.yaml"
        """
        self._validate_namespace(namespace)
        dir_path = (self._base_dir / namespace).resolve()
        if self._base_dir != dir_path and self._base_dir not in dir_path.parents:
            raise ValueError("path escapes storage root")
        dir_path.mkdir(parents=True, exist_ok=True)

        target = dir_path / f"{memory_id}.yaml"

        # Atomic write: tempfile in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
            os.replace(tmp_path, str(target))
        except BaseException:
            # Clean up temp file on any error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return f"{namespace}/{memory_id}.yaml"

    def read(self, storage_path: str) -> dict:
        """Read memory data from a YAML file.

        Args:
            storage_path: Relative path like "namespace/memory_id.yaml"

        Returns:
            Parsed YAML data as dictionary

        Raises:
            FileNotFoundError: If file does not exist
        """
        full_path = self._resolve_path(storage_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Memory file not found: {storage_path}")

        with open(full_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def delete(self, storage_path: str) -> bool:
        """Delete a memory file.

        Args:
            storage_path: Relative path like "namespace/memory_id.yaml"

        Returns:
            True if file was deleted, False if it did not exist
        """
        full_path = self._resolve_path(storage_path)
        try:
            full_path.unlink()
            return True
        except FileNotFoundError:
            return False

    def exists(self, storage_path: str) -> bool:
        """Check if a memory file exists.

        Args:
            storage_path: Relative path like "namespace/memory_id.yaml"

        Returns:
            True if file exists
        """
        return self._resolve_path(storage_path).exists()
