"""Heuristic extraction of decisions, patterns, and facts from transcript text."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Each tuple: (regex_pattern, memory_type)
# Group 1 is the captured content.
EXTRACTION_PATTERNS: list[tuple[str, str]] = [
    # Decisions
    (r"(?:decided|chose|chosen|selected|went with|picked)\s+(?:to\s+)?(.{10,500}?)(?:\.|$)", "decision"),
    (r"(?:Key decisions?|Decision):?\s*\n((?:\s*[-*]\s+.+\n?)+)", "decision"),
    # Patterns
    (r"(?:pattern|approach|strategy|technique):?\s+(.{10,500}?)(?:\.|$)", "pattern"),
    (r"(?:always|never|prefer|avoid)\s+(.{10,500}?)(?:\.|$)", "pattern"),
    # Facts / Learnings
    (r"(?:learned|discovered|found out|realized|important):?\s+(.{10,500}?)(?:\.|$)", "fact"),
    (r"(?:TIL|Note|Gotcha|Pitfall):?\s+(.{10,500}?)(?:\.|$)", "fact"),
    (r"(?:root cause|fix|fixed|bug|issue)\s+(?:was|is|:)\s+(.{10,500}?)(?:\.|$)", "fact"),
]


def parse_transcript(transcript_path: Path) -> list[str]:
    """Read a JSONL transcript file and return assistant message text strings.

    Filters to ``type == "assistant"`` entries only. Skips entries where
    ``isMeta`` is ``True``. Handles ``content`` as either a plain string
    or a list of content blocks (``{type: "text", text: "..."}``).
    """
    messages: list[str] = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if entry.get("type") != "assistant":
                continue
            if entry.get("isMeta") is True:
                continue

            msg = entry.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            if content:
                messages.append(content)

    return messages


def extract_from_text(text: str) -> list[dict]:
    """Apply heuristic patterns to a single text and return extracted items.

    Each returned dict has keys ``content`` (str) and ``type`` (str).
    Matches shorter than 10 characters or longer than 500 characters are
    discarded.
    """
    items: list[dict] = []
    for pattern, memory_type in EXTRACTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            content = match.group(1).strip()
            if 10 <= len(content) <= 500:
                items.append({"content": content, "type": memory_type})
    return items


def extract_from_messages(messages: list[str]) -> list[dict]:
    """Extract from multiple messages and deduplicate by first 100 chars."""
    all_items: list[dict] = []
    for msg in messages:
        all_items.extend(extract_from_text(msg))

    # Dedup by first 100 chars lowercase
    seen: set[str] = set()
    unique: list[dict] = []
    for item in all_items:
        key = item["content"][:100].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique
