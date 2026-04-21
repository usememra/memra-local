"""Tests for EmbeddingService and SQLite embedding BLOB storage."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from memra_local.services.embedding_service import EmbeddingService
from memra_local.storage.sqlite_index import SQLiteIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def embedding_service() -> EmbeddingService:
    """Create a fresh EmbeddingService instance (model NOT loaded yet)."""
    return EmbeddingService()


@pytest.fixture
def sqlite_index(tmp_storage: Path) -> SQLiteIndex:
    """Create and initialize a SQLiteIndex with embedding column."""
    idx = SQLiteIndex(tmp_storage / "test.db")
    idx.initialize()
    return idx


# ---------------------------------------------------------------------------
# EmbeddingService: Lazy Loading
# ---------------------------------------------------------------------------

class TestLazyLoading:
    def test_constructor_does_not_load_model(self) -> None:
        """EmbeddingService() constructor should NOT import sentence_transformers."""
        # Remove sentence_transformers from sys.modules if present so we can detect fresh import
        was_loaded = "sentence_transformers" in sys.modules
        svc = EmbeddingService()
        assert svc._model is None
        # If it wasn't loaded before construction, it shouldn't be loaded after
        if not was_loaded:
            assert "sentence_transformers" not in sys.modules

    def test_model_loads_on_first_encode(self, embedding_service: EmbeddingService) -> None:
        """Model should load lazily on first encode() call."""
        assert embedding_service._model is None
        embedding_service.encode("trigger model load")
        assert embedding_service._model is not None


# ---------------------------------------------------------------------------
# EmbeddingService: Encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encode_returns_384_dim_float32(self, embedding_service: EmbeddingService) -> None:
        """encode() should return ndarray of shape (384,) with dtype float32."""
        vec = embedding_service.encode("hello world")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_encode_batch_returns_correct_shape(self, embedding_service: EmbeddingService) -> None:
        """encode_batch() should return ndarray of shape (N, 384)."""
        vecs = embedding_service.encode_batch(["alpha", "beta"])
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (2, 384)
        assert vecs.dtype == np.float32

    def test_normalized_embeddings_unit_length(self, embedding_service: EmbeddingService) -> None:
        """Normalized embeddings should have L2 norm ~1.0."""
        vec = embedding_service.encode("test normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# EmbeddingService: Cosine Similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_cosine_similarity_ranking(self, embedding_service: EmbeddingService) -> None:
        """'dog pet' should be more similar to 'canine animal' than to 'quantum physics'."""
        query = embedding_service.encode("dog pet")
        candidates = embedding_service.encode_batch(["canine animal", "quantum physics"])

        scores = EmbeddingService.cosine_similarity(query, candidates)

        assert scores.shape == (2,)
        # "canine animal" (index 0) should score higher than "quantum physics" (index 1)
        assert scores[0] > scores[1], (
            f"Expected 'canine animal' ({scores[0]:.4f}) > 'quantum physics' ({scores[1]:.4f})"
        )

    def test_cosine_similarity_range(self, embedding_service: EmbeddingService) -> None:
        """Cosine similarity scores should be between -1 and 1 for normalized vectors."""
        query = embedding_service.encode("hello")
        candidates = embedding_service.encode_batch(["world", "goodbye", "unrelated topic"])
        scores = EmbeddingService.cosine_similarity(query, candidates)
        assert all(-1.0 <= s <= 1.0 + 1e-5 for s in scores)


# ---------------------------------------------------------------------------
# EmbeddingService: Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_serialize_produces_correct_byte_length(self, embedding_service: EmbeddingService) -> None:
        """serialize() should produce 384 * 4 = 1536 bytes."""
        vec = embedding_service.encode("test")
        blob = EmbeddingService.serialize(vec)
        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4  # float32 = 4 bytes

    def test_serialize_deserialize_roundtrip(self, embedding_service: EmbeddingService) -> None:
        """deserialize(serialize(vec)) should produce exactly the same values."""
        vec = embedding_service.encode("roundtrip test")
        blob = EmbeddingService.serialize(vec)
        restored = EmbeddingService.deserialize(blob)
        assert np.array_equal(vec, restored)
        assert restored.dtype == np.float32


# ---------------------------------------------------------------------------
# SQLite: Embedding Column Migration
# ---------------------------------------------------------------------------

class TestSQLiteMigration:
    def test_initialize_adds_embedding_column(self, tmp_storage: Path) -> None:
        """initialize() should add embedding BLOB column to memories_index."""
        idx = SQLiteIndex(tmp_storage / "migrate.db")
        idx.initialize()

        cursor = idx._c.execute("PRAGMA table_info(memories_index)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "embedding" in columns

    def test_initialize_idempotent(self, tmp_storage: Path) -> None:
        """Calling initialize() twice should not error (column already exists)."""
        idx = SQLiteIndex(tmp_storage / "idem.db")
        idx.initialize()
        idx.close()
        # Re-open and re-initialize
        idx2 = SQLiteIndex(tmp_storage / "idem.db")
        idx2.initialize()  # Should not raise
        cursor = idx2._c.execute("PRAGMA table_info(memories_index)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "embedding" in columns


# ---------------------------------------------------------------------------
# SQLite: BLOB Storage Round-Trip
# ---------------------------------------------------------------------------

class TestBlobRoundTrip:
    def test_insert_and_retrieve_embedding(self, sqlite_index: SQLiteIndex, embedding_service: EmbeddingService) -> None:
        """Insert with embedding BLOB, retrieve, and verify exact values match."""
        vec = embedding_service.encode("stored embedding test")
        blob = EmbeddingService.serialize(vec)

        sqlite_index.insert_with_embedding(
            memory_id="mem_test1",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=["test"],
            content_hash="hash1",
            storage_path="/tmp/test1.json",
            content="stored embedding test",
            source=None,
            metadata=None,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            embedding=blob,
        )

        row = sqlite_index.get_by_id("mem_test1")
        assert row is not None
        assert row["embedding"] is not None
        restored = EmbeddingService.deserialize(row["embedding"])
        assert np.array_equal(vec, restored)

    def test_insert_without_embedding_stores_null(self, sqlite_index: SQLiteIndex) -> None:
        """insert() without embedding should store NULL in embedding column."""
        sqlite_index.insert(
            memory_id="mem_no_embed",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash="hash_none",
            storage_path="/tmp/none.json",
            content="no embedding",
            source=None,
            metadata=None,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )

        row = sqlite_index.get_by_id("mem_no_embed")
        assert row is not None
        assert row["embedding"] is None


# ---------------------------------------------------------------------------
# SQLite: get_candidates_with_embeddings
# ---------------------------------------------------------------------------

class TestGetCandidatesWithEmbeddings:
    def _insert_memory_with_embedding(
        self, idx: SQLiteIndex, memory_id: str, namespace: str, tenant_id: str,
        type_: str, importance: int, embedding: bytes | None,
    ) -> None:
        """Helper to insert a memory with optional embedding."""
        idx.insert_with_embedding(
            memory_id=memory_id,
            namespace=namespace,
            tenant_id=tenant_id,
            type_=type_,
            importance=importance,
            tags=[],
            content_hash=f"hash_{memory_id}",
            storage_path=f"/tmp/{memory_id}.json",
            content=f"content for {memory_id}",
            source=None,
            metadata=None,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
            embedding=embedding,
        )

    def test_returns_only_rows_with_embeddings(self, sqlite_index: SQLiteIndex) -> None:
        """get_candidates_with_embeddings should exclude rows with NULL embedding."""
        fake_blob = np.zeros(384, dtype=np.float32).tobytes()

        self._insert_memory_with_embedding(sqlite_index, "mem_with", "ns", "t1", "fact", 5, sqlite3.Binary(fake_blob))
        self._insert_memory_with_embedding(sqlite_index, "mem_without", "ns", "t1", "fact", 5, None)

        candidates = sqlite_index.get_candidates_with_embeddings("ns", "t1")
        ids = [c["id"] for c in candidates]
        assert "mem_with" in ids
        assert "mem_without" not in ids

    def test_filters_by_namespace_and_tenant(self, sqlite_index: SQLiteIndex) -> None:
        """get_candidates_with_embeddings should filter by namespace and tenant_id."""
        fake_blob = np.zeros(384, dtype=np.float32).tobytes()

        self._insert_memory_with_embedding(sqlite_index, "mem_a", "ns1", "t1", "fact", 5, sqlite3.Binary(fake_blob))
        self._insert_memory_with_embedding(sqlite_index, "mem_b", "ns2", "t1", "fact", 5, sqlite3.Binary(fake_blob))
        self._insert_memory_with_embedding(sqlite_index, "mem_c", "ns1", "t2", "fact", 5, sqlite3.Binary(fake_blob))

        candidates = sqlite_index.get_candidates_with_embeddings("ns1", "t1")
        ids = [c["id"] for c in candidates]
        assert ids == ["mem_a"]

    def test_filters_by_type(self, sqlite_index: SQLiteIndex) -> None:
        """get_candidates_with_embeddings should filter by type when provided."""
        fake_blob = np.zeros(384, dtype=np.float32).tobytes()

        self._insert_memory_with_embedding(sqlite_index, "mem_fact", "ns", "t1", "fact", 5, sqlite3.Binary(fake_blob))
        self._insert_memory_with_embedding(sqlite_index, "mem_event", "ns", "t1", "event", 5, sqlite3.Binary(fake_blob))

        candidates = sqlite_index.get_candidates_with_embeddings("ns", "t1", type_="fact")
        ids = [c["id"] for c in candidates]
        assert "mem_fact" in ids
        assert "mem_event" not in ids

    def test_filters_by_importance_min(self, sqlite_index: SQLiteIndex) -> None:
        """get_candidates_with_embeddings should filter by minimum importance."""
        fake_blob = np.zeros(384, dtype=np.float32).tobytes()

        self._insert_memory_with_embedding(sqlite_index, "mem_low", "ns", "t1", "fact", 3, sqlite3.Binary(fake_blob))
        self._insert_memory_with_embedding(sqlite_index, "mem_high", "ns", "t1", "fact", 8, sqlite3.Binary(fake_blob))

        candidates = sqlite_index.get_candidates_with_embeddings("ns", "t1", importance_min=5)
        ids = [c["id"] for c in candidates]
        assert "mem_high" in ids
        assert "mem_low" not in ids


# ---------------------------------------------------------------------------
# SQLite: update_embedding
# ---------------------------------------------------------------------------

class TestUpdateEmbedding:
    def test_update_embedding_backfill(self, sqlite_index: SQLiteIndex) -> None:
        """update_embedding() should set embedding BLOB on existing row."""
        # Insert without embedding
        sqlite_index.insert(
            memory_id="mem_backfill",
            namespace="default",
            tenant_id="local",
            type_="fact",
            importance=5,
            tags=[],
            content_hash="hash_bf",
            storage_path="/tmp/bf.json",
            content="backfill me",
            source=None,
            metadata=None,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )

        # Verify no embedding
        row = sqlite_index.get_by_id("mem_backfill")
        assert row["embedding"] is None

        # Update with embedding
        fake_blob = np.ones(384, dtype=np.float32).tobytes()
        sqlite_index.update_embedding("mem_backfill", sqlite3.Binary(fake_blob))

        # Verify embedding now set
        row = sqlite_index.get_by_id("mem_backfill")
        assert row["embedding"] is not None
        restored = np.frombuffer(row["embedding"], dtype=np.float32)
        assert np.allclose(restored, np.ones(384, dtype=np.float32))
