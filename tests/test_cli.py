"""Tests for CLI commands using Click CliRunner."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from memra_local.cli import cli
from memra_local.services.factory import create_service


@pytest.fixture
def tmp_storage():
    """Create a temporary storage directory, clean up after test."""
    d = Path(tempfile.mkdtemp(prefix="memra_cli_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def runner():
    return CliRunner()


def _patch_factory(tmp_dir: Path):
    """Return a patch that makes create_service use a tmp dir."""
    return patch(
        "memra_local.cli.create_service",
        side_effect=lambda scope="auto", storage_dir=None: create_service(
            scope="global", storage_dir=tmp_dir
        ),
    )


# -------------------------------------------------------------------
# find_by_path tests
# -------------------------------------------------------------------

class TestFindByPath:
    def test_returns_row_for_existing_path(self, tmp_storage):
        svc = create_service(scope="global", storage_dir=tmp_storage)
        data, _ = svc.add({"content": "test content", "project_id": "default"})
        storage_path = f"default/{data['id']}.yaml"
        result = svc.index.find_by_path(storage_path)
        assert result is not None
        assert result["id"] == data["id"]

    def test_returns_none_for_missing_path(self, tmp_storage):
        svc = create_service(scope="global", storage_dir=tmp_storage)
        result = svc.index.find_by_path("nonexistent/path.yaml")
        assert result is None


# -------------------------------------------------------------------
# memra add
# -------------------------------------------------------------------

class TestAddCommand:
    def test_add_outputs_id(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["add", "test content", "-n", "default"])
        assert result.exit_code == 0
        assert "Added: mem_" in result.output

    def test_add_duplicate_outputs_duplicate(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(cli, ["add", "duplicate content", "-n", "default"])
            result = runner.invoke(cli, ["add", "duplicate content", "-n", "default"])
        assert result.exit_code == 0
        assert "Duplicate: mem_" in result.output

    def test_add_with_type_and_importance(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(
                cli, ["add", "decision content", "-n", "default", "-t", "decision", "-i", "8"]
            )
        assert result.exit_code == 0
        assert "Added: mem_" in result.output


# -------------------------------------------------------------------
# memra search
# -------------------------------------------------------------------

class TestSearchCommand:
    def test_search_finds_memories(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(cli, ["add", "python is great for data science", "-n", "default"])
            result = runner.invoke(cli, ["search", "python", "-n", "default"])
        assert result.exit_code == 0
        assert "python" in result.output.lower()

    def test_search_no_results(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["search", "nonexistent query xyz", "-n", "default"])
        assert result.exit_code == 0
        assert "No memories found" in result.output


# -------------------------------------------------------------------
# memra status
# -------------------------------------------------------------------

class TestStatusCommand:
    def test_status_shows_info(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Memories:" in result.output
        assert "Disk usage:" in result.output

    def test_status_shows_count_after_add(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(cli, ["add", "memory one", "-n", "default"])
            runner.invoke(cli, ["add", "memory two", "-n", "default"])
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "2" in result.output


# -------------------------------------------------------------------
# memra reindex
# -------------------------------------------------------------------

class TestReindexCommand:
    def test_reindex_rebuilds_index(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            # Add some memories
            runner.invoke(cli, ["add", "memory alpha", "-n", "default"])
            runner.invoke(cli, ["add", "memory beta", "-n", "default"])

            # Clear the index manually (simulate corruption)
            svc = create_service(scope="global", storage_dir=tmp_storage)
            svc.index._c.execute("DELETE FROM memories_index")
            svc.index._c.execute("DELETE FROM memories_fts")
            svc.index._c.commit()

            # Verify index is empty
            _, count = svc.index.list_memories()
            assert count == 0

            # Run reindex
            result = runner.invoke(cli, ["reindex"])

        assert result.exit_code == 0
        assert "Reindexed" in result.output
        assert "2" in result.output

    def test_reindex_preserves_superseded_status_and_chain(self, runner, tmp_storage):
        """Reindex must not resurrect superseded memories or sever chains."""
        with _patch_factory(tmp_storage):
            svc = create_service(scope="global", storage_dir=tmp_storage)
            old, _ = svc.add({"content": "version one", "project_id": "default"})
            new_data, _ = svc.supersede(old["id"], "version two")

            result = runner.invoke(cli, ["reindex"])
            assert result.exit_code == 0
            assert "Reindexed 2" in result.output

            svc = create_service(scope="global", storage_dir=tmp_storage)
            old_row = svc.index.get_by_id(old["id"])
            assert old_row["status"] == "superseded"
            assert old_row["superseded_by"] == new_data["id"]

            # Superseded memory must stay out of active recall
            results = svc.search("version", "default", "local")
            ids = [r["id"] for r in results]
            assert old["id"] not in ids
            assert new_data["id"] in ids

            # Chain intact after reindex
            chain = svc.get_chain(new_data["id"])
            assert [m["id"] for m in chain] == [old["id"], new_data["id"]]


# -------------------------------------------------------------------
# memra doctor
# -------------------------------------------------------------------

class TestDoctorCommand:
    def test_doctor_healthy_store(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(cli, ["add", "healthy memory", "-n", "default"])
            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "All checks passed" in result.output

    def test_doctor_detects_orphaned_index(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(cli, ["add", "will be orphaned", "-n", "default"])

            # Delete the flat file but keep index entry
            svc = create_service(scope="global", storage_dir=tmp_storage)
            rows, _ = svc.index.list_memories()
            for row in rows:
                file_path = tmp_storage / row["storage_path"]
                if file_path.exists():
                    file_path.unlink()

            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "WARN" in result.output
        assert "orphaned index" in result.output.lower() or "no file" in result.output.lower()

    def test_doctor_detects_orphaned_files(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            # Create a YAML file that has no index entry
            ns_dir = tmp_storage / "default"
            ns_dir.mkdir(parents=True, exist_ok=True)
            orphan_file = ns_dir / "mem_orphan123.yaml"
            yaml.safe_dump(
                {"id": "mem_orphan123", "content": "orphaned"},
                orphan_file.open("w"),
            )

            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "WARN" in result.output
        assert "orphan" in result.output.lower()

    def test_doctor_detects_corrupt_yaml(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            # Create a corrupt YAML file
            ns_dir = tmp_storage / "default"
            ns_dir.mkdir(parents=True, exist_ok=True)
            corrupt_file = ns_dir / "mem_corrupt123.yaml"
            corrupt_file.write_text("{{{{invalid yaml: [[[", encoding="utf-8")

            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "WARN" in result.output
        assert "corrupt" in result.output.lower() or "invalid" in result.output.lower()


# -------------------------------------------------------------------
# memra sync
# -------------------------------------------------------------------

class TestSyncCommands:
    def test_sync_enable_prompts_api_key(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(
                cli,
                ["sync", "enable", "my-ns", "--api-url", "https://example.com/api/v1"],
                input="memra_live_test123\n",
            )
        assert result.exit_code == 0
        assert "Sync enabled for namespace: my-ns" in result.output

    def test_sync_status_empty(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["sync", "status"])
        assert result.exit_code == 0
        assert "No namespaces have sync enabled" in result.output

    def test_sync_conflicts_empty(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["sync", "conflicts"])
        assert result.exit_code == 0
        assert "No unresolved conflicts" in result.output

    def test_status_shows_synced_namespaces(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(
                cli,
                ["sync", "enable", "my-ns"],
                input="memra_live_test123\n",
            )
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Synced namespaces: 1" in result.output

    def test_sync_set_mode_valid(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(
                cli,
                ["sync", "enable", "my-ns"],
                input="memra_live_test123\n",
            )
            result = runner.invoke(cli, ["sync", "set-mode", "my-ns", "shared_masked"])
        assert result.exit_code == 0
        assert "PII mode set to 'shared_masked'" in result.output

    def test_sync_set_mode_shared_raw_without_flag(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(
                cli,
                ["sync", "enable", "my-ns"],
                input="memra_live_test123\n",
            )
            result = runner.invoke(cli, ["sync", "set-mode", "my-ns", "shared_raw"])
        assert result.exit_code != 0
        assert "allow-raw-pii" in result.output or "allow-raw-pii" in (result.output + str(result.exception or ""))

    def test_sync_set_mode_unknown_namespace(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["sync", "set-mode", "nonexistent", "shared_masked"])
        assert result.exit_code != 0
        assert "not sync-enabled" in result.output or "not sync-enabled" in str(result.exception or "")

    def test_sync_status_shows_pii_mode(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            runner.invoke(
                cli,
                ["sync", "enable", "my-ns", "--mode", "shared_masked"],
                input="memra_live_test123\n",
            )
            result = runner.invoke(cli, ["sync", "status"])
        assert result.exit_code == 0
        assert "mode=shared_masked" in result.output

    def test_sync_enable_with_mode(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            result = runner.invoke(
                cli,
                ["sync", "enable", "my-ns", "--mode", "shared_masked"],
                input="memra_live_test123\n",
            )
        assert result.exit_code == 0
        assert "PII mode: shared_masked" in result.output


# -------------------------------------------------------------------
# memra migrate
# -------------------------------------------------------------------

class TestMigrateCommand:
    def test_migrate_dry_run(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            # Add some memories first
            runner.invoke(cli, ["add", "migrate test one", "-n", "default"])
            runner.invoke(cli, ["add", "migrate test two", "-n", "default"])

            result = runner.invoke(
                cli,
                ["migrate", "local->cloud", "--api-key", "memra_live_test",
                 "--project-id", "proj_test", "--dry-run"],
            )
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "Would migrate 2 memories" in result.output

    def test_migrate_success(self, runner, tmp_storage):
        from unittest.mock import MagicMock

        mock_batch_resp = MagicMock()
        mock_batch_resp.status_code = 200
        mock_batch_resp.json.return_value = {"created": 2, "duplicates": 0, "memories": []}
        mock_batch_resp.raise_for_status = MagicMock()

        mock_pii_resp = MagicMock()
        mock_pii_resp.status_code = 200
        mock_pii_resp.json.return_value = {
            "results": [
                {"masked_content": "migrate success one"},
                {"masked_content": "migrate success two"},
            ]
        }
        mock_pii_resp.raise_for_status = MagicMock()

        def route_post(url, **kwargs):
            if "pii/mask" in url:
                return mock_pii_resp
            return mock_batch_resp

        with _patch_factory(tmp_storage):
            runner.invoke(cli, ["add", "migrate success one", "-n", "default"])
            runner.invoke(cli, ["add", "migrate success two", "-n", "default"])

            with patch("httpx.post", side_effect=route_post):
                result = runner.invoke(
                    cli,
                    ["migrate", "local->cloud", "--api-key", "memra_live_test",
                     "--project-id", "proj_test"],
                )
        assert result.exit_code == 0
        assert "Migration complete" in result.output
        assert "2 migrated" in result.output

# -------------------------------------------------------------------
# memra supersede
# -------------------------------------------------------------------

class TestSupersedeCommand:
    def test_cli_supersede(self, runner, tmp_storage):
        """CLI supersede outputs 'Superseded: <old_id> -> <new_id>'."""
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["add", "old fact", "-n", "default"])
            old_id = result.output.strip().split("Added: ")[1]

            result = runner.invoke(cli, ["supersede", old_id, "-c", "new fact"])
        assert result.exit_code == 0
        assert f"Superseded: {old_id}" in result.output
        assert " -> mem_" in result.output

    def test_cli_supersede_not_found(self, runner, tmp_storage):
        """CLI supersede with nonexistent id outputs error and exits 1."""
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["supersede", "mem_nonexistent", "-c", "new"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_cli_history(self, runner, tmp_storage):
        """CLI history outputs numbered chain with current marker."""
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["add", "original fact", "-n", "default"])
            old_id = result.output.strip().split("Added: ")[1]

            result = runner.invoke(cli, ["supersede", old_id, "-c", "updated fact"])
            new_id = result.output.strip().split(" -> ")[1]

            result = runner.invoke(cli, ["history", old_id])
        assert result.exit_code == 0
        assert "1." in result.output
        assert "2." in result.output
        assert "current" in result.output.lower()

    def test_cli_history_not_found(self, runner, tmp_storage):
        """CLI history with nonexistent id outputs 'Memory not found.'"""
        with _patch_factory(tmp_storage):
            result = runner.invoke(cli, ["history", "mem_nonexistent"])
        assert result.exit_code == 0
        assert "Memory not found" in result.output


# -------------------------------------------------------------------
# memra migrate
# -------------------------------------------------------------------

class TestMigrateCommand:
    def test_migrate_dry_run(self, runner, tmp_storage):
        with _patch_factory(tmp_storage):
            # Add some memories first
            runner.invoke(cli, ["add", "migrate test one", "-n", "default"])
            runner.invoke(cli, ["add", "migrate test two", "-n", "default"])

            result = runner.invoke(
                cli,
                ["migrate", "local->cloud", "--api-key", "memra_live_test",
                 "--project-id", "proj_test", "--dry-run"],
            )
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "Would migrate 2 memories" in result.output

    def test_migrate_uses_env_var_api_key(self, runner, tmp_storage):
        from unittest.mock import MagicMock

        mock_batch_resp = MagicMock()
        mock_batch_resp.status_code = 200
        mock_batch_resp.json.return_value = {"created": 1, "duplicates": 0, "memories": []}
        mock_batch_resp.raise_for_status = MagicMock()

        mock_pii_resp = MagicMock()
        mock_pii_resp.status_code = 200
        mock_pii_resp.json.return_value = {
            "results": [{"masked_content": "env key test"}]
        }
        mock_pii_resp.raise_for_status = MagicMock()

        def route_post(url, **kwargs):
            if "pii/mask" in url:
                return mock_pii_resp
            return mock_batch_resp

        with _patch_factory(tmp_storage):
            runner.invoke(cli, ["add", "env key test", "-n", "default"])

            with patch("httpx.post", side_effect=route_post):
                result = runner.invoke(
                    cli,
                    ["migrate", "local->cloud", "--project-id", "proj_test"],
                    env={"MEMRA_API_KEY": "memra_live_from_env"},
                )
        assert result.exit_code == 0
        assert "Migration complete" in result.output
