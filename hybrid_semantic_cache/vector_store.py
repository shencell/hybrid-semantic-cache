"""Low-level FAISS vector store with positional metadata.

This is the original building block of the library: a flat inner-product index
(cosine similarity over L2-normalised vectors) paired with a Python list that
maps each FAISS position to an arbitrary metadata dict.

For most applications prefer :class:`hybrid_semantic_cache.HybridSemanticCache`,
which adds ids, LRU eviction, and automatic persistence on top.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension: int = 384):
        """Initialise an empty FAISS ``IndexFlatIP`` of the given dimension.

        The default (384) matches MiniLM-class sentence-transformers models.
        """
        # Inner Product over L2-normalised vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)

        # FAISS stores only vectors; this positional list carries the original
        # texts / answers / arbitrary metadata for each vector.
        self.metadata: list[dict] = []

    def add_to_index(self, question_vectors: np.ndarray, qa_pairs: list) -> None:
        """Add a batch of vectors with their metadata dicts.

        ``question_vectors`` is normalised in place so inner-product search
        behaves as cosine similarity. ``qa_pairs`` must have one dict per row.
        """
        if len(qa_pairs) != question_vectors.shape[0]:
            raise ValueError(
                f"vector/metadata count mismatch: {question_vectors.shape[0]} != {len(qa_pairs)}"
            )

        faiss.normalize_L2(question_vectors)
        self.index.add(question_vectors)
        self.metadata.extend(qa_pairs)
        logger.debug("Added %s vectors (total=%s).", len(qa_pairs), self.index.ntotal)

    def search(self, query_vector: np.ndarray, top_k: int = 1):
        """Return ``(score, metadata)`` of the best match, or ``(0.0, None)``."""
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query)

        if self.index.ntotal == 0:
            return 0.0, None

        distances, indices = self.index.search(query, top_k)
        score = float(distances[0][0])
        idx = int(indices[0][0])

        if idx != -1:
            return score, self.metadata[idx]
        return 0.0, None

    # ------------------------------------------------------------ persistence
    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Write the FAISS index and the metadata list to disk."""
        index_path, metadata_path = Path(index_path), Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug("Saved %s vectors to %s.", self.index.ntotal, index_path)

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> "VectorStore":
        """Restore a store previously written by :meth:`save`."""
        index = faiss.read_index(str(index_path))
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        if index.ntotal != len(metadata):
            raise ValueError(
                f"index has {index.ntotal} vectors but metadata has {len(metadata)} items"
            )
        store = cls(dimension=index.d)
        store.index = index
        store.metadata = metadata
        logger.info("Loaded vector store (%s vectors).", index.ntotal)
        return store


# --- SINGLETON INSTANCE (kept for backwards compatibility with v0.1.x) ---
vector_store = VectorStore()
