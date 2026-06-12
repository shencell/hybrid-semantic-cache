"""Tests for HybridSemanticCache using an injected deterministic encoder.

No SentenceTransformer download happens here: every test passes ``encode_fn``,
exactly the way an application with its own embedding pipeline would.
"""

import time

import numpy as np
import pytest

from hybrid_semantic_cache import CacheHit, HybridSemanticCache

DIM = 16

# A tiny deterministic "embedding model": every distinct text gets a stable
# pseudo-random unit vector, and identical texts get identical vectors.
def fake_encode(text: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(DIM).astype("float32")
    return v / np.linalg.norm(v)


def make_cache(**kwargs) -> HybridSemanticCache:
    kwargs.setdefault("encode_fn", fake_encode)
    kwargs.setdefault("dimension", DIM)
    kwargs.setdefault("threshold", 0.95)
    return HybridSemanticCache(**kwargs)


def test_miss_on_empty_cache():
    cache = make_cache()
    assert cache.search("anything") is None


def test_add_then_exact_hit():
    cache = make_cache()
    entry_id = cache.add("how to reset password", "click forgot password")

    hit = cache.search("how to reset password")
    assert isinstance(hit, CacheHit)
    assert hit.id == entry_id
    assert hit.response == "click forgot password"
    assert hit.score == pytest.approx(1.0, abs=1e-5)
    assert hit.hits == 1


def test_different_text_misses():
    cache = make_cache()
    cache.add("how to reset password", "click forgot password")
    # fake_encode gives unrelated texts ~orthogonal vectors -> below threshold.
    assert cache.search("completely unrelated query") is None


def test_metadata_roundtrip():
    cache = make_cache()
    cache.add("q", "a", metadata={"agent": "rag", "sources": [1, 2]})
    hit = cache.search("q")
    assert hit.metadata == {"agent": "rag", "sources": [1, 2]}


def test_hit_count_accumulates():
    cache = make_cache()
    cache.add("q", "a")
    for expected in (1, 2, 3):
        assert cache.search("q").hits == expected
    assert cache.stats()["total_hits"] == 3


def test_lru_eviction_at_capacity():
    cache = make_cache(max_records=2)
    cache.add("first", "1")
    time.sleep(0.01)
    cache.add("second", "2")
    time.sleep(0.01)

    # Touch "first" so "second" becomes the LRU entry.
    assert cache.search("first") is not None
    time.sleep(0.01)

    cache.add("third", "3")  # evicts "second"

    assert len(cache) == 2
    assert cache.search("second") is None
    assert cache.search("first").response == "1"
    assert cache.search("third").response == "3"


def test_remove_and_clear():
    cache = make_cache()
    entry_id = cache.add("q", "a")
    assert cache.remove(entry_id) is True
    assert cache.remove(entry_id) is False
    assert cache.search("q") is None

    cache.add("x", "1")
    cache.add("y", "2")
    cache.clear()
    assert len(cache) == 0
    assert cache.search("x") is None


def test_persistence_roundtrip(tmp_path):
    cache = make_cache(persist_dir=tmp_path)
    cache.add("how to reset password", "click forgot password")
    cache.search("how to reset password")  # bump hit counter

    reloaded = make_cache(persist_dir=tmp_path)
    assert len(reloaded) == 1
    hit = reloaded.search("how to reset password")
    assert hit.response == "click forgot password"
    assert hit.hits == 2  # 1 from before restart + 1 now


def test_persistence_clear_drops_state(tmp_path):
    cache = make_cache(persist_dir=tmp_path)
    cache.add("q", "a")
    cache.clear()

    reloaded = make_cache(persist_dir=tmp_path)
    assert len(reloaded) == 0
    assert reloaded.search("q") is None


def test_encode_fn_requires_dimension():
    with pytest.raises(ValueError):
        HybridSemanticCache(encode_fn=fake_encode)


def test_invalid_capacity_rejected():
    with pytest.raises(ValueError):
        make_cache(max_records=0)
