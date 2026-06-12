"""Tests for the Indonesian slang/typo normalizer."""

from hybrid_semantic_cache import normalize_text
from hybrid_semantic_cache.normalizer import TextNormalizer, normalizer


def test_module_level_function_matches_readme_api():
    assert normalize_text("Tlong bkinin srt resign dong") == normalizer.normalize(
        "Tlong bkinin srt resign dong"
    )


def test_slang_is_translated():
    out = normalize_text("gmn cr ganti pw?")
    assert "bagaimana" in out
    assert "cara" in out
    assert "kata sandi" in out
    assert "gmn" not in out


def test_stopwords_are_removed():
    out = normalize_text("min gimana dong nih")
    assert "min" not in out.split()
    assert "dong" not in out.split()
    assert "bagaimana" in out


def test_punctuation_and_case_folding():
    out = normalize_text("KLO Error, GIMANA?!")
    assert out == out.lower()
    assert "," not in out and "?" not in out and "!" not in out
    assert "kalau" in out


def test_fresh_instance_behaves_like_singleton():
    assert TextNormalizer().normalize("klo gak bs") == normalizer.normalize("klo gak bs")
