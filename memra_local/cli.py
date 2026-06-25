"""Click CLI entry point for memra-local server."""

from __future__ import annotations

import os
from pathlib import Path

import click
import uvicorn
import yaml

from memra_local import __version__
from memra_local.services.factory import create_service


@click.group()
@click.version_option(version=__version__, prog_name="memra")
def cli():
    """Memra local memory server."""
    pass


@cli.command()
@click.option(
    "--port",
    default=8765,
    type=int,
    show_default=True,
    help="Port to listen on",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind to",
)
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
    show_default=True,
    help="Storage scope",
)
def serve(port: int, host: str, scope: str):
    """Start the local Memra memory server."""
    from memra_local.app import create_app
    from memra_local.config import resolve_scope, resolve_storage_dir

    actual_scope = resolve_scope(scope)
    storage_dir = resolve_storage_dir(actual_scope)

    click.echo(f"Memra local server v{__version__}")
    click.echo(f"Scope: {actual_scope}")
    click.echo(f"Storage: {storage_dir}")
    click.echo(f"Listening on http://{host}:{port}")

    app = create_app(scope=actual_scope)
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


@cli.command()
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
    show_default=True,
    help="Storage scope",
)
def mcp(scope: str):
    """Start MCP stdio server for IDE integration (Claude Code, Cursor, Zed)."""
    from memra_local.mcp_server import run_mcp

    run_mcp(scope=scope)


# -------------------------------------------------------------------
# add
# -------------------------------------------------------------------

@cli.command()
@click.argument("content")
@click.option("--namespace", "-n", default="default", help="Namespace")
@click.option(
    "--type",
    "-t",
    "type_",
    default="fact",
    type=click.Choice(["fact", "decision", "pattern", "preference", "skill"]),
    help="Memory type",
)
@click.option(
    "--importance",
    "-i",
    default=5,
    type=click.IntRange(1, 10),
    help="Importance 1-10",
)
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
)
def add(content: str, namespace: str, type_: str, importance: int, scope: str):
    """Add a memory from the command line."""
    svc = create_service(scope=scope)
    data, is_duplicate = svc.add(
        {
            "content": content,
            "project_id": namespace,
            "type": type_,
            "importance": importance,
        }
    )
    memory_id = data.get("id", "unknown")
    if is_duplicate:
        click.echo(f"Duplicate: {memory_id}")
    else:
        click.echo(f"Added: {memory_id}")


# -------------------------------------------------------------------
# supersede
# -------------------------------------------------------------------

@cli.command()
@click.argument("old_id")
@click.option("--content", "-c", required=True, help="New memory content")
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
)
def supersede(old_id: str, content: str, scope: str):
    """Supersede an existing memory with new content."""
    from memra_local.exceptions import ConcurrentModificationError

    svc = create_service(scope=scope)
    try:
        new_mem, _old_mem = svc.supersede(old_id, content)
        click.echo(f"Superseded: {old_id} -> {new_mem['id']}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except ConcurrentModificationError:
        click.echo("Error: Memory was modified concurrently. Please retry.", err=True)
        raise SystemExit(1)


# -------------------------------------------------------------------
# history
# -------------------------------------------------------------------

@cli.command()
@click.argument("memory_id")
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
)
def history(memory_id: str, scope: str):
    """Show supersession history for a memory."""
    svc = create_service(scope=scope)
    chain = svc.get_chain(memory_id)

    if not chain:
        click.echo("Memory not found.")
        return

    for i, mem in enumerate(chain):
        status = mem.get("status", "active")
        marker = " <- current" if status != "superseded" else ""
        click.echo(f"  {i + 1}. {mem['id']} ({status}){marker}")


# -------------------------------------------------------------------
# search
# -------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--namespace", "-n", default="default", help="Namespace")
@click.option("--type", "-t", "type_", default=None, help="Filter by type")
@click.option("--limit", "-l", default=10, type=int, help="Max results")
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
)
def search(query: str, namespace: str, type_: str | None, limit: int, scope: str):
    """Search memories by query."""
    svc = create_service(scope=scope)
    results, meta = svc.recall(
        query=query,
        namespace=namespace,
        tenant_id="local",
        type_=type_,
        limit=limit,
    )

    if not results:
        click.echo("No memories found.")
        return

    for r in results:
        score = r.get("score", 0)
        mem_type = r.get("type", "fact")
        content = r.get("content", "")
        preview = content[:80] + ("..." if len(content) > 80 else "")
        click.echo(f"  [{score:.3f}] ({mem_type}) {preview}")

    click.echo(f"\n{len(results)} result(s) found.")


# -------------------------------------------------------------------
# status
# -------------------------------------------------------------------

@cli.command()
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
)
def status(scope: str):
    """Show memory store status."""
    svc = create_service(scope=scope)

    # Memory count
    _, total = svc.list_(limit=0)

    # Disk usage
    storage_dir = svc.store._base_dir
    disk_bytes = 0
    for dirpath, _dirnames, filenames in os.walk(storage_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                disk_bytes += os.path.getsize(fp)
            except OSError:
                pass

    # Format disk usage
    if disk_bytes < 1024:
        disk_str = f"{disk_bytes} B"
    elif disk_bytes < 1024 * 1024:
        disk_str = f"{disk_bytes / 1024:.1f} KB"
    else:
        disk_str = f"{disk_bytes / (1024 * 1024):.1f} MB"

    # Embedding model
    has_model = svc.embedding_service is not None and svc.embedding_service._model is not None

    from memra_local.config import resolve_scope as _rs

    actual_scope = _rs(scope)

    click.echo(f"Memra v{__version__}")
    click.echo(f"Scope: {actual_scope}")
    click.echo(f"Storage: {storage_dir}")
    click.echo(f"Memories: {total}")
    click.echo(f"Disk usage: {disk_str}")
    # Sync status
    if svc.sync_service:
        sync_rows = svc.sync_service._conn.execute(
            "SELECT COUNT(*) FROM sync_cursors"
        ).fetchone()[0]
        if sync_rows > 0:
            click.echo(f"Synced namespaces: {sync_rows}")

    click.echo(f"Embedding model: {'loaded' if has_model else 'not loaded'}")


# -------------------------------------------------------------------
# reindex
# -------------------------------------------------------------------

@cli.command()
@click.option("--embeddings", is_flag=True, help="Also regenerate embeddings (slow)")
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
)
def reindex(embeddings: bool, scope: str):
    """Rebuild the SQLite index from flat files."""
    import hashlib
    from datetime import datetime, timezone

    svc = create_service(scope=scope)
    storage_dir = svc.store._base_dir

    # Walk all .yaml files
    yaml_files: list[tuple[str, Path]] = []
    for dirpath, _dirnames, filenames in os.walk(storage_dir):
        for fname in filenames:
            if fname.endswith(".yaml") and not fname.startswith("."):
                full = Path(dirpath) / fname
                rel = str(full.relative_to(storage_dir))
                yaml_files.append((rel, full))

    count = 0
    embed_count = 0

    for rel_path, full_path in yaml_files:
        try:
            with open(full_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            click.echo(f"  SKIP: {rel_path} (parse error)")
            continue

        if not isinstance(data, dict) or "id" not in data or "content" not in data:
            click.echo(f"  SKIP: {rel_path} (missing id/content)")
            continue

        memory_id = data["id"]
        content = data["content"]
        namespace = data.get("project_id", "default")
        tenant_id = data.get("tenant_id", "local")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Remove existing entry if present
        svc.index.delete_by_id(memory_id)

        # Generate embedding if requested
        embedding_blob = None
        if embeddings and svc.embedding_service is not None:
            emb = svc.embedding_service.encode(content)
            embedding_blob = svc.embedding_service.serialize(emb)
            embed_count += 1

        try:
            svc.index.insert_with_embedding(
                memory_id=memory_id,
                namespace=namespace,
                tenant_id=tenant_id,
                type_=data.get("type", "fact"),
                importance=data.get("importance", 5),
                tags=data.get("tags") or [],
                content_hash=content_hash,
                storage_path=rel_path,
                content=content,
                source=data.get("source"),
                metadata=data.get("metadata"),
                created_at=data.get("created_at", now),
                updated_at=data.get("updated_at", now),
                embedding=embedding_blob,
                # Preserve lifecycle state from the flat file -- without this,
                # reindex resurrects superseded memories and severs chains.
                status=data.get("status", "active"),
                superseded_by=data.get("superseded_by"),
            )
            count += 1
        except Exception as e:
            click.echo(f"  SKIP: {rel_path} ({e})")

    if embeddings and embed_count > 0:
        click.echo(f"Reindexed {count} memories ({embed_count} embeddings regenerated)")
    else:
        click.echo(f"Reindexed {count} memories")


# -------------------------------------------------------------------
# doctor
# -------------------------------------------------------------------

@cli.command()
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
)
def doctor(scope: str):
    """Check store health: orphaned files, orphaned index entries, corrupt YAML."""
    svc = create_service(scope=scope)
    storage_dir = svc.store._base_dir
    issues: list[str] = []

    # 1. SQLite integrity check
    try:
        result = svc.index._c.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            issues.append(f"WARN: SQLite integrity check failed: {result[0]}")
    except Exception as e:
        issues.append(f"WARN: SQLite integrity check error: {e}")

    # 2. Orphaned index entries (index row but no file)
    rows, _ = svc.index.list_memories(limit=10000)
    for row in rows:
        file_path = storage_dir / row["storage_path"]
        if not file_path.exists():
            issues.append(f"WARN: Orphaned index entry (no file): {row['id']} -> {row['storage_path']}")

    # 3. Orphaned flat files (file exists but no index entry)
    for dirpath, _dirnames, filenames in os.walk(storage_dir):
        for fname in filenames:
            if not fname.endswith(".yaml") or fname.startswith("."):
                continue
            full = Path(dirpath) / fname
            rel = str(full.relative_to(storage_dir))
            entry = svc.index.find_by_path(rel)
            if entry is None:
                issues.append(f"WARN: Orphaned file (no index entry): {rel}")

    # 4. Corrupt YAML files
    for dirpath, _dirnames, filenames in os.walk(storage_dir):
        for fname in filenames:
            if not fname.endswith(".yaml") or fname.startswith("."):
                continue
            full = Path(dirpath) / fname
            try:
                with open(full, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    rel = str(full.relative_to(storage_dir))
                    issues.append(f"WARN: Corrupt/invalid YAML (not a mapping): {rel}")
            except yaml.YAMLError:
                rel = str(full.relative_to(storage_dir))
                issues.append(f"WARN: Corrupt YAML (parse error): {rel}")

    if issues:
        for issue in issues:
            click.echo(issue)
        click.echo(f"\n{len(issues)} issue(s) found.")
    else:
        click.echo("All checks passed. Store is healthy.")


# -------------------------------------------------------------------
# sync group
# -------------------------------------------------------------------

@cli.group()
def sync():
    """Manage namespace synchronization with cloud."""
    pass


@sync.command("enable")
@click.argument("namespace")
@click.option("--api-key", prompt=True, hide_input=True, help="Cloud API key (memra_live_...)")
@click.option("--api-url", default="https://usememra.com/api/v1", show_default=True, help="Cloud API URL")
@click.option("--mode", type=click.Choice(["local_private", "shared_masked", "shared_raw"]), default="local_private", show_default=True, help="PII sharing mode")
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def sync_enable(namespace, api_key, api_url, mode, scope):
    """Enable cloud sync for a namespace."""
    svc = create_service(scope=scope)
    svc.sync_service.enable(namespace, api_key, api_url, pii_mode=mode)
    click.echo(f"Sync enabled for namespace: {namespace}")
    click.echo(f"Cloud API: {api_url}")
    click.echo(f"PII mode: {mode}")


@sync.command("push")
@click.option("--namespace", "-n", required=True, help="Namespace to push")
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def sync_push(namespace, scope):
    """Push local changes to cloud."""
    svc = create_service(scope=scope)
    result = svc.sync_service.push(namespace)
    if result.error:
        click.echo(f"Error: {result.error}", err=True)
        raise SystemExit(1)
    click.echo(f"Pushed {result.pushed} event(s)")
    if result.conflicts:
        click.echo(f"Conflicts: {len(result.conflicts)} (run 'memra sync conflicts' to view)")


@sync.command("pull")
@click.option("--namespace", "-n", required=True, help="Namespace to pull")
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def sync_pull(namespace, scope):
    """Pull remote changes from cloud."""
    svc = create_service(scope=scope)
    result = svc.sync_service.pull(namespace, svc)
    if result.error:
        click.echo(f"Error: {result.error}", err=True)
        raise SystemExit(1)
    click.echo(f"Applied {result.applied} event(s)")
    if result.has_more:
        click.echo("More events available -- run again to continue")


@sync.command("status")
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def sync_status(scope):
    """Show sync status for all enabled namespaces."""
    svc = create_service(scope=scope)
    rows = svc.sync_service._conn.execute(
        "SELECT * FROM sync_cursors ORDER BY namespace"
    ).fetchall()
    if not rows:
        click.echo("No namespaces have sync enabled.")
        return
    for row in rows:
        ns = row["namespace"]
        cursor = row["remote_cursor"]
        last = row["last_synced_at"] or "never"
        pii_mode = row["pii_mode"] if "pii_mode" in row.keys() else "shared_masked"
        unpushed = svc.sync_service._conn.execute(
            "SELECT COUNT(*) FROM sync_events WHERE namespace = ? AND pushed_at IS NULL",
            (ns,),
        ).fetchone()[0]
        click.echo(f"  {ns}: mode={pii_mode}, cursor={cursor}, unpushed={unpushed}, last_sync={last}")


@sync.command("conflicts")
@click.option("--namespace", "-n", default=None, help="Filter by namespace")
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def sync_conflicts(namespace, scope):
    """List unresolved sync conflicts."""
    svc = create_service(scope=scope)
    conflicts = svc.sync_service.list_conflicts(namespace)
    if not conflicts:
        click.echo("No unresolved conflicts.")
        return
    for c in conflicts:
        click.echo(f"  {c['id']} ({c.get('type', 'fact')}) -- created {c.get('created_at', '?')}")
    click.echo(f"\n{len(conflicts)} conflict(s). Resolve with: memra sync resolve <id> --keep local|remote")


@sync.command("resolve")
@click.argument("conflict_id")
@click.option("--keep", type=click.Choice(["local", "remote"]), required=True)
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def sync_resolve(conflict_id, keep, scope):
    """Resolve a sync conflict by keeping local or remote version."""
    svc = create_service(scope=scope)
    ok = svc.sync_service.resolve_conflict(conflict_id, keep, svc)
    if ok:
        click.echo(f"Conflict resolved: kept {keep} version")
    else:
        click.echo(f"Conflict not found: {conflict_id}", err=True)
        raise SystemExit(1)


@sync.command("set-mode")
@click.argument("namespace")
@click.argument("mode", type=click.Choice(["local_private", "shared_masked", "shared_raw"]))
@click.option("--allow-raw-pii", is_flag=True, help="Required for shared_raw mode")
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def sync_set_mode(namespace, mode, allow_raw_pii, scope):
    """Set PII sharing mode for a synced namespace."""
    svc = create_service(scope=scope)

    if mode == "shared_raw" and not allow_raw_pii:
        click.echo("Error: shared_raw requires --allow-raw-pii flag", err=True)
        raise SystemExit(1)

    if mode == "shared_raw":
        tier = svc.sync_service.check_account_tier(namespace)
        if tier not in ("team", "admin"):
            click.echo(f"Error: shared_raw requires Team tier or above (current: {tier})", err=True)
            raise SystemExit(1)

    ok = svc.sync_service.set_mode(namespace, mode)
    if not ok:
        click.echo(f"Error: namespace '{namespace}' is not sync-enabled", err=True)
        raise SystemExit(1)
    click.echo(f"PII mode set to '{mode}' for namespace: {namespace}")


# -------------------------------------------------------------------
# migrate
# -------------------------------------------------------------------

@cli.command("migrate")
@click.argument("direction", type=click.Choice(["local->cloud"]))
@click.option("--api-key", envvar="MEMRA_API_KEY", help="Cloud API key (or set MEMRA_API_KEY)")
@click.option("--api-url", default="https://usememra.com/api/v1", show_default=True, help="Cloud API URL")
@click.option("--project-id", required=True, help="Target cloud project ID (proj_...) to migrate into")
@click.option("--dry-run", is_flag=True, help="Show what would be migrated without uploading")
@click.option("--scope", type=click.Choice(["global", "project", "auto"]), default="auto")
def migrate(direction, api_key, api_url, project_id, dry_run, scope):
    """Migrate memories between local and cloud."""
    if not api_key:
        api_key = click.prompt("Cloud API key", hide_input=True)

    svc = create_service(scope=scope)

    from memra_local.services.pii_client import PiiClient
    from memra_local.services.migration_service import MigrationService

    pii_client = PiiClient(api_url=api_url, api_key=api_key)
    migration_svc = MigrationService(api_url=api_url, api_key=api_key, pii_client=pii_client)

    if dry_run:
        click.echo("Dry run -- no data will be uploaded")

    def progress(migrated, total):
        click.echo(f"  Progress: {migrated}/{total} memories", nl=False)
        click.echo("\r", nl=False)

    result = migration_svc.migrate(
        index=svc.index,
        store=svc.store,
        project_id=project_id,
        dry_run=dry_run,
        progress_callback=progress,
    )

    click.echo("")  # newline after progress
    if result.dry_run:
        click.echo(f"Would migrate {result.total} memories")
    else:
        click.echo(
            f"Migration complete: {result.migrated} migrated, "
            f"{result.skipped} skipped (duplicates), {result.failed} failed"
        )

    if result.errors:
        for err in result.errors:
            click.echo(f"  Error: {err}", err=True)

    if result.failed > 0:
        raise SystemExit(1)


# -------------------------------------------------------------------
# hooks group
# -------------------------------------------------------------------

@cli.group()
def hooks():
    """Manage auto-capture hooks for IDE integration."""
    pass


@hooks.command("install")
def hooks_install():
    """Install SessionEnd hook in Claude Code settings.json."""
    from memra_local.hooks.installer import install_session_end_hook

    try:
        result = install_session_end_hook()
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        click.echo("Is Claude Code installed? (~/.claude/ directory not found)", err=True)
        raise SystemExit(1)

    if result:
        click.echo("Memra auto-capture hook installed in Claude Code.")
        click.echo("Sessions will now auto-capture learnings on exit.")
    else:
        click.echo("Memra auto-capture hook already installed.")


@hooks.command("uninstall")
def hooks_uninstall():
    """Remove SessionEnd hook from Claude Code settings.json."""
    from memra_local.hooks.installer import uninstall_session_end_hook

    result = uninstall_session_end_hook()
    if result:
        click.echo("Memra auto-capture hook removed.")
    else:
        click.echo("Memra auto-capture hook was not installed.")


# -------------------------------------------------------------------
# capture (top-level -- invoked by the hook itself)
# -------------------------------------------------------------------

@cli.command()
@click.option("--hook-id", default="memra-auto-capture", hidden=True)
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
    show_default=True,
)
def capture(hook_id: str, scope: str):
    """Capture learnings from a Claude Code session transcript.

    This command is invoked automatically by the SessionEnd hook.
    It reads JSON from stdin containing session_id and transcript_path.
    """
    import json as _json
    import sys

    try:
        stdin_data = sys.stdin.read()
    except Exception:
        stdin_data = ""

    if not stdin_data.strip():
        # No input -- exit silently (don't spam Claude Code)
        _output_hook_result(0)
        return

    try:
        hook_input = _json.loads(stdin_data)
    except _json.JSONDecodeError:
        click.echo("Warning: invalid JSON on stdin", err=True)
        _output_hook_result(0)
        return

    transcript_path_str = hook_input.get("transcript_path", "")
    session_id = hook_input.get("session_id", "unknown")

    if not transcript_path_str:
        _output_hook_result(0)
        return

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        click.echo(f"Warning: transcript not found: {transcript_path}", err=True)
        _output_hook_result(0)
        return

    try:
        from memra_local.hooks.extractor import extract_from_messages, parse_transcript

        messages = parse_transcript(transcript_path)
        # Process only last 100 assistant messages (avoid timeout on huge transcripts)
        messages = messages[-100:]

        items = extract_from_messages(messages)

        if not items:
            _output_hook_result(0)
            return

        svc = create_service(scope=scope)
        saved = 0
        for item in items:
            _data, is_dup = svc.add(
                {
                    "content": item["content"],
                    "project_id": "auto-capture",
                    "type": item["type"],
                    "importance": 5,
                    "source": "claude-code-session",
                    "metadata": {
                        "trust_state": "proposed",
                        "session_id": session_id,
                    },
                }
            )
            if not is_dup:
                saved += 1

        _output_hook_result(saved)

    except Exception as exc:
        # Never crash the hook runner
        click.echo(f"Warning: capture failed: {exc}", err=True)
        _output_hook_result(0)


# -------------------------------------------------------------------
# review
# -------------------------------------------------------------------

@cli.command()
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "auto"]),
    default="auto",
    show_default=True,
    help="Storage scope",
)
@click.option(
    "--batch",
    "batch_action",
    type=click.Choice(["verify", "reject"]),
    default=None,
    help="Non-interactive: verify or reject all proposed at once",
)
def review(scope: str, batch_action: str | None):
    """Review auto-captured memories (trust_state=proposed).

    Interactively verify, reject, or skip each proposed memory.
    Use --batch verify/reject for non-interactive bulk action.
    """
    import json as _json

    svc = create_service(scope=scope)

    # Query for proposed memories via direct SQL on the index
    rows = svc.index._c.execute(
        """SELECT * FROM memories_index
           WHERE json_extract(metadata, '$.trust_state') = 'proposed'
           ORDER BY created_at DESC""",
    ).fetchall()
    proposed = [dict(r) for r in rows]

    if not proposed:
        click.echo("No proposed memories to review.")
        return

    verified_count = 0
    rejected_count = 0
    skipped_count = 0

    for row in proposed:
        memory_id = row["id"]
        mem_type = row.get("type", "fact")

        # Read content from flat file
        try:
            file_data = svc.store.read(row["storage_path"])
            content = file_data.get("content", "")
        except FileNotFoundError:
            content = "(file missing)"

        preview = content[:120] + ("..." if len(content) > 120 else "")

        # Parse metadata for session_id
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = _json.loads(metadata)
        session_id = (metadata or {}).get("session_id", "unknown")

        if batch_action == "verify":
            # Auto-verify
            metadata["trust_state"] = "verified"
            svc.update(memory_id, {"metadata": metadata})
            verified_count += 1
            continue
        elif batch_action == "reject":
            # Auto-reject
            metadata["trust_state"] = "rejected"
            svc.update(memory_id, {"metadata": metadata})
            rejected_count += 1
            continue

        # Interactive mode
        click.echo(f"\n  [{mem_type}] {preview}")
        click.echo(f"  ID: {memory_id}  session: {session_id}")
        action = click.prompt(
            "  Action", type=click.Choice(["v", "r", "s"]), default="s"
        )

        if action == "v":
            metadata["trust_state"] = "verified"
            svc.update(memory_id, {"metadata": metadata})
            verified_count += 1
        elif action == "r":
            metadata["trust_state"] = "rejected"
            svc.update(memory_id, {"metadata": metadata})
            rejected_count += 1
        else:
            skipped_count += 1

    total = verified_count + rejected_count + skipped_count
    click.echo(
        f"\nReviewed {total} memories: "
        f"{verified_count} verified, {rejected_count} rejected, {skipped_count} skipped"
    )


def _output_hook_result(saved: int) -> None:
    """Print JSON output for the Claude Code hook system."""
    import json as _json

    result = {
        "hookSpecificOutput": {
            "hookEventName": "SessionEnd",
            "memoriesCaptured": saved,
        }
    }
    click.echo(_json.dumps(result))


if __name__ == "__main__":
    cli()
