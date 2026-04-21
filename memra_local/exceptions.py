"""Custom exceptions for memra-local."""

from __future__ import annotations


class ConcurrentModificationError(Exception):
    """Raised when optimistic locking detects a version conflict."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__(
            f"Memory '{memory_id}' was modified by another process. "
            "Re-read the memory and retry."
        )
