# Changelog

## 0.2.0 (2026-06-10)

### Added
- **`HybridSemanticCache`** — new high-level API: semantic `search`/`add` with
  configurable similarity threshold, **LRU eviction** at `max_records`
  capacity, per-entry `hits`/`last_accessed` tracking, `remove`/`clear`/`stats`,
  and optional **disk persistence** via `persist_dir` (auto save/restore).
- **`encode_fn` injection** — plug in custom embedding pipelines (e.g.
  multilingual-e5 with `"query: "` prefixing) without loading the default model.
- **`VectorStore.save()` / `VectorStore.load()`** — persistence for the
  low-level store (FAISS index + JSON metadata).
- **`normalize_text()`** — module-level function as documented in the README
  (previously only the `TextNormalizer` class existed).
- Lazy public API exports in `hybrid_semantic_cache/__init__.py` (PEP 562)
  with `__version__`.
- Test suite (pytest) covering the normalizer, vector store, and cache
  (LRU eviction, persistence round-trip, custom encoders).
- GitHub Actions: CI tests on push/PR and automated PyPI publishing on
  `v*` tags.

### Changed
- `TextEmbedder` now loads its SentenceTransformer **lazily** on first encode
  instead of at import time.
- All `print()` calls replaced with standard `logging`.
- `main.py` demo rewritten to be self-contained (its previous imports
  `api.routes` / `core.embedding` did not exist inside the package and crashed
  on import). Gemini fallback is now optional.
- Dependencies slimmed down: core = `faiss-cpu`, `numpy`,
  `sentence-transformers`. FastAPI/uvicorn/google-generativeai moved to the
  `[demo]` extra; unused `matplotlib`/`pandas` removed.
- `VectorStore.add_to_index` validates that vector and metadata counts match;
  `VectorStore.search` handles an empty index without calling FAISS.

### Fixed
- README cleaned up (stray draft/conversation text removed; examples now match
  the real API).
- `dist/` and `*.egg-info/` removed from version control.

## 0.1.1

- Initial public release: `VectorStore`, `TextEmbedder`, `TextNormalizer`,
  FastAPI demo.
