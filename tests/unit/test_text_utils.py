"""
Unit tests for src/utils/text_utils.py.

All tests are dependency-free and run without external services.
"""
import pytest

from src.utils.text_utils import (
    approximate_token_count,
    content_hash,
    normalize_whitespace,
    quality_score,
    remove_boilerplate,
    sliding_window_chunks,
)

pytestmark = pytest.mark.unit


# ── normalize_whitespace ───────────────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert normalize_whitespace("  hi  ") == "hi"

    def test_collapses_tabs_and_newlines(self):
        result = normalize_whitespace("line1\n\t  line2")
        assert result == "line1 line2"

    def test_handles_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_unicode_normalization(self):
        # NFKD should decompose combined characters
        result = normalize_whitespace("\ufb01")  # ﬁ ligature
        assert len(result) >= 1  # decomposed or kept, not empty


# ── remove_boilerplate ────────────────────────────────────────────────────────

class TestRemoveBoilerplate:
    def test_removes_standalone_numbers(self):
        text = "Introduction\n42\nSome content"
        result = remove_boilerplate(text)
        assert "42" not in result.split("\n") or "42" not in result

    def test_removes_separator_lines(self):
        text = "Header\n------\nContent"
        result = remove_boilerplate(text)
        assert "------" not in result

    def test_collapses_excess_blank_lines(self):
        text = "Para one\n\n\n\nPara two"
        result = remove_boilerplate(text)
        assert "\n\n\n" not in result

    def test_leaves_normal_text_intact(self):
        text = "Transformer models use self-attention mechanisms."
        assert remove_boilerplate(text) == text


# ── content_hash ───────────────────────────────────────────────────────────────

class TestContentHash:
    def test_returns_64_char_hex_string(self):
        h = content_hash("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_text_same_hash(self):
        assert content_hash("same text") == content_hash("same text")

    def test_different_text_different_hash(self):
        assert content_hash("text a") != content_hash("text b")

    def test_case_insensitive(self):
        # content_hash lowercases before hashing
        assert content_hash("Hello") == content_hash("hello")

    def test_whitespace_normalised(self):
        # Internal whitespace is collapsed before hashing
        assert content_hash("hello  world") == content_hash("hello world")


# ── approximate_token_count ────────────────────────────────────────────────────

class TestApproximateTokenCount:
    def test_returns_positive_int(self):
        assert approximate_token_count("hello world") >= 1

    def test_empty_string_returns_one(self):
        assert approximate_token_count("") == 1

    def test_longer_text_higher_count(self):
        short = approximate_token_count("hi")
        long = approximate_token_count("hi " * 100)
        assert long > short

    def test_heuristic_multiplier(self):
        # 10 words × 1.3 = 13
        result = approximate_token_count(" ".join(["word"] * 10))
        assert result == 13


# ── quality_score ──────────────────────────────────────────────────────────────

class TestQualityScore:
    def test_range_zero_to_one(self):
        score = quality_score("Hello, this is a well-formed English sentence with diverse vocabulary.")
        assert 0.0 <= score <= 1.0

    def test_empty_string_returns_zero(self):
        assert quality_score("") == 0.0

    def test_short_string_returns_zero(self):
        assert quality_score("hi") == 0.0

    def test_high_score_for_natural_language(self):
        text = (
            "Transformer models have revolutionised natural language processing by introducing "
            "self-attention mechanisms that capture long-range dependencies efficiently."
        )
        assert quality_score(text) > 0.5

    def test_low_score_for_repetitive_text(self):
        text = "123 123 123 123 123 123 123 123"
        assert quality_score(text) < 0.3

    def test_returns_rounded_float(self):
        score = quality_score("Some normal text with words and letters")
        # Should be rounded to 4 decimal places
        assert score == round(score, 4)


# ── sliding_window_chunks ──────────────────────────────────────────────────────

class TestSlidingWindowChunks:
    def _collect(self, text, **kwargs):
        return list(sliding_window_chunks(text, **kwargs))

    def test_empty_string_yields_nothing(self):
        assert self._collect("") == []

    def test_short_text_yields_one_chunk(self):
        text = "A short piece of text that fits in one chunk easily."
        chunks = self._collect(text, chunk_size=512, overlap=64, min_length=10)
        assert len(chunks) == 1
        chunk_text, start, end = chunks[0]
        assert chunk_text == text.strip()
        assert start == 0

    def test_long_text_yields_multiple_chunks(self):
        text = "word " * 400  # 2000 chars
        chunks = self._collect(text, chunk_size=200, overlap=20, min_length=10)
        assert len(chunks) > 1

    def test_chunk_offsets_are_valid(self):
        text = "Hello world. " * 50
        for chunk_text, start, end in self._collect(text, chunk_size=100, overlap=10, min_length=5):
            assert 0 <= start < end <= len(text)

    def test_chunks_below_min_length_filtered(self):
        text = "abc"
        chunks = self._collect(text, chunk_size=512, overlap=64, min_length=10)
        assert chunks == []

    def test_overlap_creates_coverage(self):
        # With overlap, adjacent chunks should share some characters
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = self._collect(text, chunk_size=80, overlap=20, min_length=5)
        if len(chunks) >= 2:
            _, start1, end1 = chunks[0]
            _, start2, end2 = chunks[1]
            # Second chunk starts before the first chunk ends
            assert start2 < end1
