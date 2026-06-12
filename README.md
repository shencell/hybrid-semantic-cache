# Hybrid Semantic Cache

FAISS-backed **semantic caching middleware** for LLM applications: normalize
noisy user queries, retrieve semantically similar cached answers locally, and
only call the cloud LLM on a real miss — cutting latency and API cost.

```
user query ──▶ normalize ──▶ semantic lookup (FAISS) ──┬─▶ HIT  → cached answer (ms)
                                                       └─▶ MISS → your LLM → cache.add()
```

## Features

- **Semantic lookup** — cosine similarity over sentence-transformer embeddings
  (FAISS `IndexFlatIP` / `IndexIDMap2`), with a configurable threshold.
- **LRU eviction** — hard `max_records` capacity; least-recently-used entries
  are evicted from both the vector index and the metadata store.
- **Disk persistence** — pass `persist_dir` and the cache survives restarts.
- **Pluggable embeddings** — inject your own `encode_fn` (custom models,
  prefixing rules such as e5's `"query: "`, GPU batching) or let the library
  lazily load a SentenceTransformer for you.
- **Indonesian text normalization** — `normalize_text()` rewrites slang/typos
  ("gmn cr ganti pw?") into standard text before embedding, raising hit rates.
- **Hit tracking** — per-entry `hits` counter and `last_accessed` timestamps.

## Installation

```bash
pip install hybrid-semantic-cache
# with the FastAPI + Gemini demo app:
pip install "hybrid-semantic-cache[demo]"
```

Requires Python 3.9+.

## Quick Start

```python
from hybrid_semantic_cache import HybridSemanticCache, normalize_text

cache = HybridSemanticCache(
    threshold=0.80,        # min cosine similarity for a hit
    max_records=1000,      # LRU eviction beyond this
    persist_dir="./cache", # optional: survive restarts
)

query = normalize_text("gmn cr ganti pw email yak?")

hit = cache.search(query)
if hit is not None:
    print(f"cache hit ({hit.score:.2f}):", hit.response)
else:
    answer = call_your_llm(query)          # any provider
    cache.add(query, answer)
```

### Custom embeddings (e.g. multilingual-e5)

E5-family models need a `"query: "` prefix and benefit from normalization —
inject your own encoder and the library never loads a second model:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")

def encode(text: str) -> np.ndarray:
    return model.encode(f"query: {text}", normalize_embeddings=True)

cache = HybridSemanticCache(encode_fn=encode, dimension=768, threshold=0.92)
```

### Metadata, stats, and management

```python
entry_id = cache.add(prompt, response, metadata={"sources": [...], "agent": "rag"})
hit = cache.search(prompt)     # hit.metadata, hit.hits, hit.score
cache.stats()                  # {'records': ..., 'capacity': ..., 'total_hits': ...}
cache.remove(entry_id)
cache.clear()
len(cache)
```

### Low-level building blocks

```python
from hybrid_semantic_cache import VectorStore, TextEmbedder

store = VectorStore(dimension=384)
store.add_to_index(vectors, [{"question": q, "answer": a}, ...])
score, meta = store.search(query_vector)
store.save("cache.index", "metadata.json")
store = VectorStore.load("cache.index", "metadata.json")
```

## Demo app

A self-contained FastAPI service showing the full normalize → cache → LLM
fallback flow (uses Gemini when `GEMINI_API_KEY` is set, a stub otherwise):

```bash
pip install "hybrid-semantic-cache[demo]"
uvicorn hybrid_semantic_cache.main:app
# POST {"message": "..."} to http://127.0.0.1:8000/chat
```

## Development

```bash
git clone https://github.com/shencell/hybrid-semantic-cache
cd hybrid-semantic-cache
pip install -e ".[dev]"
pytest
```

## License

MIT
