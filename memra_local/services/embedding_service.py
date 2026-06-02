"""Lazy-loading local embedding service using fastembed (ONNX runtime).

Runs the same all-MiniLM-L6-v2 model as sentence-transformers, but on the
~50MB onnxruntime instead of the ~1.2GB torch stack. Embeddings are numerically
identical (verified) and L2-normalized, so cosine similarity stays a dot product.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class EmbeddingService:
    """Generate and compare embeddings using all-MiniLM-L6-v2.

    Model loads lazily on first encode call -- NOT at import or construction time.
    This keeps server startup fast (model load deferred until needed).
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSIONS = 384

    def __init__(self) -> None:
        self._model = None

    def _ensure_model(self) -> None:
        """Load model on first use. Downloads if not cached (~90MB).

        Cache the ONNX model in a persistent dir. fastembed otherwise defaults
        to a tempfile dir (/tmp/fastembed_cache) that is wiped on reboot, which
        forces a full re-download before any embedding works and silently
        degrades recall to FTS-only until it completes. Default to
        ~/.cache/fastembed; honor FASTEMBED_CACHE_PATH for shared/custom caches.
        """
        if self._model is not None:
            return
        from fastembed import TextEmbedding

        cache_dir = os.environ.get(
            "FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed")
        )
        self._model = TextEmbedding(model_name=self.MODEL_NAME, cache_dir=cache_dir)

    def encode(self, text: str) -> NDArray[np.float32]:
        """Encode a single text to a 384-dim normalized float32 vector."""
        self._ensure_model()
        return next(iter(self._model.embed([text]))).astype(np.float32)

    def encode_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode multiple texts. More efficient than calling encode() in a loop."""
        self._ensure_model()
        return np.array(list(self._model.embed(texts)), dtype=np.float32)

    @staticmethod
    def cosine_similarity(
        query: NDArray[np.float32], candidates: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Compute cosine similarity between query and candidate vectors.

        Pre-normalized vectors make this a simple dot product.
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
