"""Lazy-loading local embedding service using sentence-transformers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class EmbeddingService:
    """Generate and compare embeddings using all-MiniLM-L6-v2.

    Model loads lazily on first encode call -- NOT at import or construction time.
    This keeps server startup fast (~6s model load deferred until needed).
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSIONS = 384

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self) -> None:
        """Load model on first use. Downloads if not cached (~80MB)."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode a single text to a 384-dim normalized float32 vector."""
        self._ensure_model()
        return self._model.encode(text, normalize_embeddings=True)

    def encode_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode multiple texts. More efficient than calling encode() in a loop."""
        self._ensure_model()
        return self._model.encode(texts, normalize_embeddings=True)

    @staticmethod
    def cosine_similarity(
        query: NDArray[np.float32], candidates: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between query and candidate vectors.

        Pre-normalized vectors (normalize_embeddings=True) make this a simple dot product.
        """
        return candidates @ query

    @staticmethod
    def serialize(embedding: NDArray[np.float32]) -> bytes:
        """Serialize embedding to bytes for SQLite BLOB storage.

        Produces 384 * 4 = 1536 bytes (float32).
        """
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def deserialize(blob: bytes) -> NDArray[np.float32]:
        """Deserialize embedding from SQLite BLOB."""
        return np.frombuffer(blob, dtype=np.float32)
