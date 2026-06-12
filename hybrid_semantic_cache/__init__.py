"""Hybrid Semantic Cache - FAISS-backed semantic caching for LLM applications.

Public API (lazily imported so ``import hybrid_semantic_cache`` stays fast and
never pulls heavy ML dependencies until you actually use them):

* :class:`HybridSemanticCache` - high-level cache with LRU eviction & persistence.
* :class:`CacheHit`            - dataclass returned on a successful lookup.
* :class:`VectorStore`         - low-level FAISS index + positional metadata.
* :class:`TextEmbedder`        - lazy SentenceTransformer wrapper.
* :class:`TextNormalizer` / :func:`normalize_text` - Indonesian slang/typo normalizer.
"""

from __future__ import annotations

__version__ = "0.2.0"

_EXPORTS = {
    "HybridSemanticCache": "hybrid_semantic_cache.cache",
    "CacheHit": "hybrid_semantic_cache.cache",
    "VectorStore": "hybrid_semantic_cache.vector_store",
    "TextEmbedder": "hybrid_semantic_cache.embedding",
    "TextNormalizer": "hybrid_semantic_cache.normalizer",
    "normalize_text": "hybrid_semantic_cache.normalizer",
}

__all__ = list(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    """PEP 562 lazy attribute access for the public API."""
    if name in _EXPORTS:
        import importlib

        module = importlib.import_module(_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
