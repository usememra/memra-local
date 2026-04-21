"""API endpoint tests for memra-local — validates cloud-compatible JSON responses."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memra_local.app import create_app


# ---------------------------------------------------------------------------
# All fields the SDK Memory model expects
# ---------------------------------------------------------------------------
SDK_MEMORY_FIELDS = {
    "id", "content", "tenant_id", "project_id", "type", "importance",
    "tags", "source", "metadata", "embedding_status", "expires_at",
    "confidence", "staleness_score", "last_accessed_at", "status",
    "superseded_by", "ttl_days", "soft_ttl_days", "created_at", "updated_at",
}


@pytest.fixture
def client():
    """Create a FastAPI TestClient backed by a temporary storage directory."""
    tmp = Path(tempfile.mkdtemp(prefix="memra_api_test_"))
    app = create_app(scope="project", storage_dir=tmp)
    with TestClient(app) as c:
        yield c
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "version" in body


# ---------------------------------------------------------------------------
# POST /v1/memories
# ---------------------------------------------------------------------------
class TestAddMemory:
    def test_add_memory(self, client: TestClient):
        r = client.post("/v1/memories", json={
            "content": "Python was created by Guido.",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        assert r.status_code == 201
        body = r.json()
        assert body["content"] == "Python was created by Guido."
        assert body["tenant_id"] == "t1"
        assert body["project_id"] == "p1"
        assert body["id"].startswith("mem_")

    def test_add_memory_defaults(self, client: TestClient):
        r = client.post("/v1/memories", json={"content": "Hello world"})
        assert r.status_code == 201
        body = r.json()
        assert body["tenant_id"] == "local"
        assert body["project_id"] == "default"
        assert body["type"] == "fact"
        assert body["importance"] == 5

    def test_add_duplicate(self, client: TestClient):
        payload = {"content": "Duplicate test", "tenant_id": "t1", "project_id": "p1"}
        r1 = client.post("/v1/memories", json=payload)
        assert r1.status_code == 201
        first_id = r1.json()["id"]

        r2 = client.post("/v1/memories", json=payload)
        assert r2.status_code == 200  # Not 409 or 201
        assert r2.json()["id"] == first_id

    def test_add_response_matches_sdk_memory(self, client: TestClient):
        """Ensure response has all fields the Python SDK Memory model expects."""
        r = client.post("/v1/memories", json={"content": "SDK compat test"})
        assert r.status_code == 201
        body = r.json()
        missing = SDK_MEMORY_FIELDS - set(body.keys())
        assert not missing, f"Missing SDK fields: {missing}"

    def test_add_rejects_project_path_traversal(self, client: TestClient):
        r = client.post("/v1/memories", json={
            "content": "blocked",
            "project_id": "../escape",
            "tenant_id": "t1",
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /v1/memories/{id}
# ---------------------------------------------------------------------------
class TestGetMemory:
    def test_get_memory(self, client: TestClient):
        r = client.post("/v1/memories", json={"content": "Retrievable fact"})
        mid = r.json()["id"]

        r2 = client.get(f"/v1/memories/{mid}")
        assert r2.status_code == 200
        assert r2.json()["content"] == "Retrievable fact"
        assert r2.json()["id"] == mid

    def test_get_memory_not_found(self, client: TestClient):
        r = client.get("/v1/memories/mem_nonexistent")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/memories
# ---------------------------------------------------------------------------
class TestListMemories:
    def test_list_memories(self, client: TestClient):
        for i in range(3):
            client.post("/v1/memories", json={"content": f"List item {i}"})

        r = client.get("/v1/memories")
        assert r.status_code == 200
        body = r.json()
        assert "memories" in body
        assert body["total"] == 3
        assert "limit" in body
        assert "offset" in body
        assert "has_more" in body

    def test_list_memories_filter_type(self, client: TestClient):
        client.post("/v1/memories", json={"content": "A fact", "type": "fact"})
        client.post("/v1/memories", json={"content": "An event", "type": "event"})

        r = client.get("/v1/memories", params={"type": "fact"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["memories"][0]["type"] == "fact"

    def test_list_response_matches_sdk(self, client: TestClient):
        """Ensure list response validates against SDK MemoryList model."""
        client.post("/v1/memories", json={"content": "List compat test"})
        r = client.get("/v1/memories")
        body = r.json()
        assert "memories" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body
        assert "has_more" in body


# ---------------------------------------------------------------------------
# PATCH /v1/memories/{id}
# ---------------------------------------------------------------------------
class TestUpdateMemory:
    def test_update_memory(self, client: TestClient):
        r = client.post("/v1/memories", json={
            "content": "Original content",
            "importance": 3,
        })
        mid = r.json()["id"]

        r2 = client.patch(f"/v1/memories/{mid}", json={
            "content": "Updated content",
            "importance": 8,
        })
        assert r2.status_code == 200
        body = r2.json()
        assert body["content"] == "Updated content"
        assert body["importance"] == 8

    def test_update_not_found(self, client: TestClient):
        r = client.patch("/v1/memories/mem_nonexistent", json={"importance": 5})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /v1/memories/{id}
# ---------------------------------------------------------------------------
class TestDeleteMemory:
    def test_delete_memory(self, client: TestClient):
        r = client.post("/v1/memories", json={"content": "To be deleted"})
        mid = r.json()["id"]

        r2 = client.delete(f"/v1/memories/{mid}")
        assert r2.status_code == 204

        # Confirm gone
        r3 = client.get(f"/v1/memories/{mid}")
        assert r3.status_code == 404

    def test_delete_not_found(self, client: TestClient):
        r = client.delete("/v1/memories/mem_nonexistent")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /v1/memories/search
# ---------------------------------------------------------------------------
class TestSearch:
    def test_search(self, client: TestClient):
        client.post("/v1/memories", json={
            "content": "Python is a programming language",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        client.post("/v1/memories", json={
            "content": "Rust is a systems language",
            "tenant_id": "t1",
            "project_id": "p1",
        })

        r = client.post("/v1/memories/search", json={
            "query": "Python",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        assert "Python" in body[0]["content"]


# ---------------------------------------------------------------------------
# POST /v1/memories/recall
# ---------------------------------------------------------------------------
class TestRecall:
    def test_recall(self, client: TestClient):
        client.post("/v1/memories", json={
            "content": "The capital of France is Paris",
            "tenant_id": "t1",
            "project_id": "p1",
        })

        r = client.post("/v1/memories/recall", json={
            "query": "capital France",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        assert r.status_code == 200
        body = r.json()
        assert "data" in body
        assert "meta" in body
        # With embeddings wired, scoring should be cosine_similarity
        assert body["meta"]["scoring"] == "cosine_similarity"
        assert body["meta"]["query_cached"] is False
        assert body["meta"]["response_cached"] is False
        assert isinstance(body["meta"]["total_candidates"], int)
        assert isinstance(body["meta"]["returned"], int)

    def test_recall_response_matches_sdk(self, client: TestClient):
        """Ensure recall response validates against SDK RecallResult model."""
        client.post("/v1/memories", json={
            "content": "SDK recall compat",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        r = client.post("/v1/memories/recall", json={
            "query": "SDK recall",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        body = r.json()
        assert "data" in body
        assert "meta" in body
        if body["data"]:
            item = body["data"][0]
            for key in ("id", "content", "score", "type", "importance", "tags", "created_at"):
                assert key in item, f"Missing RecallMemory field: {key}"


# ---------------------------------------------------------------------------
# POST /v1/bootstrap
# ---------------------------------------------------------------------------
class TestBootstrap:
    def test_bootstrap(self, client: TestClient):
        # Add memories with different importances
        client.post("/v1/memories", json={
            "content": "Low importance",
            "importance": 2,
            "tenant_id": "t1",
            "project_id": "p1",
        })
        client.post("/v1/memories", json={
            "content": "High importance",
            "importance": 9,
            "tenant_id": "t1",
            "project_id": "p1",
        })
        client.post("/v1/memories", json={
            "content": "Medium importance",
            "importance": 5,
            "tenant_id": "t1",
            "project_id": "p1",
        })

        r = client.post("/v1/bootstrap", json={
            "tenant_id": "t1",
            "project_id": "p1",
        })
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 3
        # Should be sorted by importance DESC
        importances = [m["importance"] for m in body]
        assert importances == sorted(importances, reverse=True)


# ---------------------------------------------------------------------------
# Semantic Recall (embedding-aware)
# ---------------------------------------------------------------------------
class TestSemanticRecall:
    def test_add_sets_embedding_status(self, client: TestClient):
        """Adding a memory should set embedding_status to 'complete'."""
        r = client.post("/v1/memories", json={
            "content": "Dogs are loyal pets",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        assert r.status_code == 201
        assert r.json()["embedding_status"] == "complete"

    def test_recall_meta_reports_cosine(self, client: TestClient):
        """Recall with embedded memories reports scoring='cosine_similarity'."""
        client.post("/v1/memories", json={
            "content": "Dogs are loyal pets",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        r = client.post("/v1/memories/recall", json={
            "query": "loyal animals",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        assert r.status_code == 200
        assert r.json()["meta"]["scoring"] == "cosine_similarity"

    def test_recall_semantic_scoring(self, client: TestClient):
        """Semantic recall ranks related concepts higher than unrelated."""
        client.post("/v1/memories", json={
            "content": "dogs are loyal pets",
            "tenant_id": "t1",
            "project_id": "p1",
            "importance": 5,
        })
        client.post("/v1/memories", json={
            "content": "cats are independent animals",
            "tenant_id": "t1",
            "project_id": "p1",
            "importance": 5,
        })
        client.post("/v1/memories", json={
            "content": "quantum physics is complex",
            "tenant_id": "t1",
            "project_id": "p1",
            "importance": 5,
        })

        r = client.post("/v1/memories/recall", json={
            "query": "canine companions",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        assert r.status_code == 200
        data = r.json()["data"]
        assert len(data) >= 2
        # Dog memory should score higher than quantum physics
        contents = [d["content"] for d in data]
        dog_idx = next(i for i, c in enumerate(contents) if "dogs" in c)
        quantum_idx = next(i for i, c in enumerate(contents) if "quantum" in c)
        assert dog_idx < quantum_idx, "Dog memory should rank above quantum physics"

    def test_recall_importance_in_score(self, client: TestClient):
        """Higher importance increases score via 0.3*(importance/10) factor."""
        client.post("/v1/memories", json={
            "content": "the sky is blue during day",
            "tenant_id": "t1",
            "project_id": "p1",
            "importance": 3,
        })
        client.post("/v1/memories", json={
            "content": "the sky appears blue in daytime",
            "tenant_id": "t1",
            "project_id": "p1",
            "importance": 9,
        })

        r = client.post("/v1/memories/recall", json={
            "query": "sky color",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        data = r.json()["data"]
        assert len(data) == 2
        # Higher importance (9) should score higher than lower (3)
        assert data[0]["importance"] == 9
        assert data[0]["score"] > data[1]["score"]

    def test_update_content_re_embeds(self, client: TestClient):
        """Updating content re-generates embedding so new query finds it."""
        r = client.post("/v1/memories", json={
            "content": "original unrelated topic about rocks",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        mid = r.json()["id"]

        # Update content to be about dogs
        client.patch(f"/v1/memories/{mid}", json={
            "content": "dogs are wonderful loyal companions",
        })

        r = client.post("/v1/memories/recall", json={
            "query": "loyal pet animals",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        data = r.json()["data"]
        assert len(data) >= 1
        assert data[0]["id"] == mid

    def test_recall_score_formula(self, client: TestClient):
        """Score follows 0.7*similarity + 0.3*(importance/10)."""
        client.post("/v1/memories", json={
            "content": "machine learning algorithms",
            "tenant_id": "t1",
            "project_id": "p1",
            "importance": 10,
        })

        r = client.post("/v1/memories/recall", json={
            "query": "machine learning algorithms",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        data = r.json()["data"]
        assert len(data) == 1
        score = data[0]["score"]
        # With identical query/content, similarity ~ 1.0
        # Score should be ~ 0.7*1.0 + 0.3*(10/10) = 1.0
        assert 0.9 <= score <= 1.0, f"Expected score ~1.0 for identical text, got {score}"

class TestSupersede:
    def test_supersede_memory(self, client: TestClient):
        """POST /v1/memories/{id}/supersede returns 201 with new memory data."""
        r = client.post("/v1/memories", json={"content": "Original fact"})
        old_id = r.json()["id"]

        r2 = client.post(f"/v1/memories/{old_id}/supersede", json={"content": "Updated fact"})
        assert r2.status_code == 201
        body = r2.json()
        assert "data" in body
        assert body["data"]["id"].startswith("mem_")
        assert body["data"]["content"] == "Updated fact"

    def test_supersede_not_found(self, client: TestClient):
        """POST /v1/memories/nonexistent/supersede returns 404."""
        r = client.post("/v1/memories/mem_nonexistent/supersede", json={"content": "New"})
        assert r.status_code == 404

    def test_supersede_already_superseded(self, client: TestClient):
        """Superseding an already-superseded memory returns 409 Conflict."""
        r = client.post("/v1/memories", json={"content": "Will be superseded"})
        old_id = r.json()["id"]

        # Supersede once
        client.post(f"/v1/memories/{old_id}/supersede", json={"content": "First replacement"})

        # Supersede again should fail
        r2 = client.post(f"/v1/memories/{old_id}/supersede", json={"content": "Second replacement"})
        assert r2.status_code == 409


class TestChain:
    def test_get_chain(self, client: TestClient):
        """GET /v1/memories/{id}/chain returns ordered chain."""
        r = client.post("/v1/memories", json={"content": "Chain root"})
        root_id = r.json()["id"]

        r2 = client.post(f"/v1/memories/{root_id}/supersede", json={"content": "Chain v2"})
        new_id = r2.json()["data"]["id"]

        r3 = client.get(f"/v1/memories/{root_id}/chain")
        assert r3.status_code == 200
        body = r3.json()
        assert "data" in body
        assert "length" in body
        assert body["length"] == 2
        assert len(body["data"]) == 2

    def test_get_chain_not_found(self, client: TestClient):
        """GET /v1/memories/nonexistent/chain returns 404."""
        r = client.get("/v1/memories/mem_nonexistent/chain")
        assert r.status_code == 404

    def test_list_excludes_superseded(self, client: TestClient):
        """GET /v1/memories after supersession does not return superseded memory."""
        r = client.post("/v1/memories", json={"content": "Will be superseded", "project_id": "p1"})
        old_id = r.json()["id"]

        client.post(f"/v1/memories/{old_id}/supersede", json={"content": "Replacement"})

        r2 = client.get("/v1/memories", params={"project_id": "p1"})
        body = r2.json()
        ids = [m["id"] for m in body["memories"]]
        assert old_id not in ids

    def test_recall_excludes_superseded(self, client: TestClient):
        """POST /v1/recall after supersession does not return superseded memory."""
        r = client.post("/v1/memories", json={
            "content": "Python is great for AI",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        old_id = r.json()["id"]

        client.post(f"/v1/memories/{old_id}/supersede", json={"content": "Python is excellent for AI"})

        r2 = client.post("/v1/memories/recall", json={
            "query": "Python AI",
            "tenant_id": "t1",
            "project_id": "p1",
        })
        data = r2.json()["data"]
        ids = [d["id"] for d in data]
        assert old_id not in ids


class TestSemanticRecallExtra:
    def test_recall_no_network_calls(self, client: TestClient):
        """All embedding and recall operations are fully local — zero network calls."""
        from unittest.mock import patch

        # Add a memory first (this triggers model loading + encoding)
        client.post("/v1/memories", json={
            "content": "Local embedding test content",
            "tenant_id": "t1",
            "project_id": "p1",
        })

        # Now block all network sockets and verify recall still works
        original_connect = __import__("socket").socket.connect

        def blocked_connect(self, *args, **kwargs):
            raise ConnectionError("Network calls are blocked in this test")

        with patch("socket.socket.connect", blocked_connect):
            r = client.post("/v1/memories/recall", json={
                "query": "local embedding",
                "tenant_id": "t1",
                "project_id": "p1",
            })
            assert r.status_code == 200
            data = r.json()["data"]
            assert len(data) >= 1
            assert r.json()["meta"]["scoring"] == "cosine_similarity"
