"""Tests for the low-level VectorStore (no embedding model required)."""

import numpy as np
import pytest

from hybrid_semantic_cache import VectorStore

DIM = 8


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype("float32")
    return v / np.linalg.norm(v)


def test_add_and_search_roundtrip():
    store = VectorStore(dimension=DIM)
    vectors = np.stack([_vec(1), _vec(2)])
    store.add_to_index(vectors, [{"answer": "a"}, {"answer": "b"}])

    score, meta = store.search(_vec(2))
    assert meta == {"answer": "b"}
    assert score == pytest.approx(1.0, abs=1e-5)


def test_search_empty_store_returns_none():
    store = VectorStore(dimension=DIM)
    score, meta = store.search(_vec(1))
    assert meta is None
    assert score == 0.0


def test_mismatched_metadata_count_raises():
    store = VectorStore(dimension=DIM)
    with pytest.raises(ValueError):
        store.add_to_index(np.stack([_vec(1), _vec(2)]), [{"only": "one"}])


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore(dimension=DIM)
    store.add_to_index(np.stack([_vec(1), _vec(2)]), [{"id": 1}, {"id": 2}])
    store.save(tmp_path / "v.index", tmp_path / "v.json")

    loaded = VectorStore.load(tmp_path / "v.index", tmp_path / "v.json")
    assert loaded.index.ntotal == 2
    score, meta = loaded.search(_vec(1))
    assert meta == {"id": 1}
    assert score == pytest.approx(1.0, abs=1e-5)
