"""High-level semantic cache with LRU eviction and disk persistence.

:class:`HybridSemanticCache` is the recommended entry point of this library.
It pairs a FAISS ``IndexIDMap2`` (so entries can be removed individually and
reconstructed for persistence) with an in-memory metadata store, and exposes a
two-method API: :meth:`search` and :meth:`add`.

Example
-------
>>> from hybrid_semantic_cache import HybridSemanticCache
>>> cache = HybridSemanticCache(threshold=0.80, max_records=1000)
>>> if (hit := cache.search("gmn cara reset kata sandi?")) is not None:
...     print(hit.response, hit.score)
... else:
...     answer = call_your_llm(...)          # cache miss -> ask the LLM
...     cache.add("gmn cara reset kata sandi?", answer)

Custom embeddings
-----------------
By default the cache lazily loads a SentenceTransformer model. To integrate
with an existing embedding pipeline (different model, prefixing rules, GPU
batching, ...), inject ``encode_fn`` returning a 1-D ``float32`` numpy array
and pass the matching ``dimension``::

    cache = HybridSemanticCache(encode_fn=my_encoder, dimension=768)

Persistence
-----------
Pass ``persist_dir`` to make the cache durable: state is restored on
construction and saved automatically after every mutation. Saved artifacts are
a FAISS index file plus a JSON metadata file.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_INDEX_FILE = "cache.index"
_STORE_FILE = "cache_store.json"


@dataclass
class CacheHit:
    """A successful semantic-cache lookup."""

    id: int
    prompt: str
    response: str
    score: float
    hits: int
    metadata: dict = field(default_factory=dict)


class HybridSemanticCache:
    """Semantic prompt/response cache: FAISS vectors + metadata + LRU eviction.

    Args:
        model_name: SentenceTransformer model used when ``encode_fn`` is not
            given. Loaded lazily on first use.
        threshold: Minimum cosine similarity (0..1) for :meth:`search` to
            report a hit.
        dimension: Embedding dimensionality. Required with ``encode_fn``;
            otherwise inferred from the model on first use.
        max_records: Hard capacity. Inserting beyond it evicts the
            least-recently-used entry first.
        encode_fn: Optional ``str -> np.ndarray`` (1-D float32) embedding
            override. When provided, no SentenceTransformer is loaded.
        persist_dir: Optional directory for automatic save/restore.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        threshold: float = 0.75,
        dimension: Optional[int] = None,
        max_records: int = 1000,
        encode_fn: Optional[Callable[[str], np.ndarray]] = None,
        persist_dir: Optional[str | Path] = None,
    ) -> None:
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        if encode_fn is not None and dimension is None:
            raise ValueError("dimension is required when encode_fn is provided")

        self.threshold = threshold
        self.max_records = max_records
        self._model_name = model_name
        self._encode_fn = encode_fn
        self._dimension = dimension
        self._persist_dir = Path(persist_dir) if persist_dir else None

        self._index = None  # faiss.IndexIDMap2, created lazily (needs dimension)
        self._store: dict[int, dict[str, Any]] = {}
        self._next_id = 0
        self._lock = threading.RLock()

        if self._persist_dir is not None:
            self._restore()

    # ------------------------------------------------------------- embeddings
    def _encode(self, text: str) -> np.ndarray:
        if self._encode_fn is not None:
            vector = np.asarray(self._encode_fn(text), dtype="float32")
        else:
            from .embedding import TextEmbedder

            if not hasattr(self, "_embedder"):
                self._embedder = TextEmbedder(self._model_name)
            vector = self._embedder.encode_text(text)
            if self._dimension is None:
                self._dimension = int(vector.shape[-1])

        vector = vector.reshape(-1)
        norm = np.linalg.norm(vector)
        if norm > 0:  # normalise so inner product == cosine similarity
            vector = vector / norm
        return vector.astype("float32")

    def _ensure_index(self):
        if self._index is None:
            import faiss

            if self._dimension is None:
                # Trigger a model load to discover the dimension.
                self._encode("dimension probe")
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
        return self._index

    # ------------------------------------------------------------------ API
    def search(self, query: str, threshold: Optional[float] = None) -> Optional[CacheHit]:
        """Return the best :class:`CacheHit` for ``query``, or ``None``.

        A hit refreshes the entry's ``last_accessed`` timestamp and increments
        its hit counter (this drives the LRU eviction order).
        """
        with self._lock:
            index = self._ensure_index()
            if index.ntotal == 0:
                return None

            vector = self._encode(query).reshape(1, -1)
            scores, ids = index.search(vector, 1)
            score = float(scores[0][0])
            match_id = int(ids[0][0])

            min_score = self.threshold if threshold is None else threshold
            if match_id == -1 or score < min_score:
                logger.debug("Cache miss (best=%.4f < threshold=%.2f).", score, min_score)
                return None

            entry = self._store[match_id]
            entry["hits"] += 1
            entry["last_accessed"] = time.time()
            if self._persist_dir is not None:
                self._save()

            logger.debug("Cache hit id=%s score=%.4f.", match_id, score)
            return CacheHit(
                id=match_id,
                prompt=entry["prompt"],
                response=entry["response"],
                score=score,
                hits=entry["hits"],
                metadata=entry.get("metadata", {}),
            )

    def add(self, prompt: str, response: str, metadata: Optional[dict] = None) -> int:
        """Store a prompt/response pair and return its id.

        When the cache is at ``max_records`` capacity the least-recently-used
        entry is evicted (from both FAISS and the metadata store) first.
        """
        with self._lock:
            index = self._ensure_index()

            while len(self._store) >= self.max_records:
                self._evict_lru()

            entry_id = self._next_id
            self._next_id += 1

            vector = self._encode(prompt).reshape(1, -1)
            index.add_with_ids(vector, np.asarray([entry_id], dtype="int64"))

            now = time.time()
            self._store[entry_id] = {
                "prompt": prompt,
                "response": response,
                "metadata": metadata or {},
                "hits": 0,
                "created_at": now,
                "last_accessed": now,
            }
            if self._persist_dir is not None:
                self._save()

            logger.debug("Cache store id=%s (total=%s).", entry_id, len(self._store))
            return entry_id

    def remove(self, entry_id: int) -> bool:
        """Delete one entry by id. Returns ``True`` when it existed."""
        with self._lock:
            if entry_id not in self._store:
                return False
            self._ensure_index().remove_ids(np.asarray([entry_id], dtype="int64"))
            del self._store[entry_id]
            if self._persist_dir is not None:
                self._save()
            return True

    def clear(self) -> None:
        """Remove every entry."""
        with self._lock:
            self._index = None
            self._store.clear()
            if self._persist_dir is not None:
                self._save()

    def stats(self) -> dict:
        """Size/usage statistics."""
        with self._lock:
            return {
                "records": len(self._store),
                "capacity": self.max_records,
                "threshold": self.threshold,
                "total_hits": sum(e["hits"] for e in self._store.values()),
            }

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------- internals
    def _evict_lru(self) -> None:
        victim_id = min(self._store, key=lambda i: self._store[i]["last_accessed"])
        self._ensure_index().remove_ids(np.asarray([victim_id], dtype="int64"))
        del self._store[victim_id]
        logger.debug("LRU eviction: removed id=%s.", victim_id)

    # ------------------------------------------------------------ persistence
    def save(self, directory: Optional[str | Path] = None) -> None:
        """Write the FAISS index and metadata store to ``directory``."""
        with self._lock:
            target = Path(directory) if directory else self._persist_dir
            if target is None:
                raise ValueError("No directory given and persist_dir is not set.")
            self._persist_to(target)

    def _save(self) -> None:
        self._persist_to(self._persist_dir)

    def _persist_to(self, target: Path) -> None:
        import faiss

        target.mkdir(parents=True, exist_ok=True)
        index_file = target / _INDEX_FILE
        if self._index is not None:
            faiss.write_index(self._index, str(index_file))
        elif index_file.exists():
            index_file.unlink()  # cleared cache: drop the stale index file
        payload = {
            "next_id": self._next_id,
            "dimension": self._dimension,
            "store": {str(k): v for k, v in self._store.items()},
        }
        (target / _STORE_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _restore(self) -> None:
        """Best-effort restore from ``persist_dir``; corrupt state starts fresh."""
        import faiss

        index_path = self._persist_dir / _INDEX_FILE
        store_path = self._persist_dir / _STORE_FILE
        if not store_path.exists():
            return
        try:
            payload = json.loads(store_path.read_text(encoding="utf-8"))
            store = {int(k): v for k, v in payload["store"].items()}
            index = faiss.read_index(str(index_path)) if index_path.exists() else None
            if index is not None and index.ntotal != len(store):
                raise ValueError(
                    f"index has {index.ntotal} vectors but store has {len(store)} entries"
                )
            self._store = store
            self._next_id = int(payload["next_id"])
            self._dimension = payload.get("dimension") or self._dimension
            self._index = index
            logger.info("Restored semantic cache (%s entries).", len(store))
        except Exception as exc:
            logger.warning("Could not restore cache state (%s); starting fresh.", exc)
            self._store, self._next_id, self._index = {}, 0, None
