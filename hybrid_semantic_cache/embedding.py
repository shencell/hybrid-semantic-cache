"""Lazy SentenceTransformer wrapper.

The model is loaded on first use rather than at import time, so importing this
module (or the package) costs nothing until you actually encode text. This
matters for consumers who inject their own embeddings and never need the
default model.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """Remember the model name; the heavy load happens on first encode."""
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        """Download/load the SentenceTransformer exactly once."""
        if self._model is None:
            logger.info("Loading embedding model '%s'...", self.model_name)
            start = time.perf_counter()

            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "Embedding model loaded in %.2fs (dim=%s).",
                time.perf_counter() - start,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimensionality (triggers the model load)."""
        return int(self._ensure_model().get_sentence_embedding_dimension())

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one sentence to a float32 vector (FAISS-compatible)."""
        vector = self._ensure_model().encode(text)
        return np.asarray(vector, dtype="float32")

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode many sentences at once (efficient for pre-warming)."""
        vectors = self._ensure_model().encode(texts)
        return np.asarray(vectors, dtype="float32")


# --- SINGLETON INSTANCE (kept for backwards compatibility with v0.1.x) ---
# Constructing TextEmbedder is now free; the model loads on first encode call.
embedder = TextEmbedder()
