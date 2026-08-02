"""
Text processing utilities: normalization, chunking, hashing, quality scoring.

All functions are pure (no I/O, no side effects) to keep them easily testable.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterator


def normalize_whitespace(text: str) -> str:
    """
    Normalize unicode, collapse multiple whitespace characters into a single space.
    """
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """
    Remove common boilerplate patterns: standalone page numbers,
    decorative separator lines, and excessive blank lines.
    """
    # Remove lines that are purely numeric (page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove lines with only repeated punctuation characters (e.g. -----, =====)
    text = re.sub(r"^[\-=_*#~]{5,}\s*$", "", text, flags=re.MULTILINE)
    # Collapse 3+ consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(text: str) -> str:
    """
    Deterministic SHA-256 hash of normalized, lowercased text.
    Used for deduplication across pipeline runs.
    """
    normalized = normalize_whitespace(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def approximate_token_count(text: str) -> int:
    """
    Estimate token count using a word-count heuristic.
    Assumes ~1.3 tokens per word (conservative estimate for English text).
    """
    words = len(text.split())
    return max(1, int(words * 1.3))


def sliding_window_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_length: int = 50,
) -> Iterator[tuple[str, int, int]]:
    """
    Yield (chunk_text, char_start, char_end) tuples using a character-level
    sliding window with sentence-boundary awareness.

    Args:
        text:        Input text to chunk.
        chunk_size:  Maximum characters per chunk.
        overlap:     Character overlap between consecutive chunks.
        min_length:  Minimum characters for a chunk to be emitted.

    Yields:
        Tuples of (chunk_text, start_offset, end_offset).
    """
    if not text:
        return

    if len(text) <= chunk_size:
        stripped = text.strip()
        if len(stripped) >= min_length:
            yield stripped, 0, len(text)
        return

    step = max(1, chunk_size - overlap)
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at a sentence boundary in the last 20% of the window
        if end < len(text):
            search_from = start + int(chunk_size * 0.80)
            boundary = _find_sentence_boundary(text, search_from, end)
            if boundary:
                end = boundary

        chunk = text[start:end].strip()
        if len(chunk) >= min_length:
            yield chunk, start, end

        start += step


def _find_sentence_boundary(text: str, search_from: int, search_to: int) -> int | None:
    """
    Scan backward from search_to to search_from for sentence-ending punctuation.
    Returns the character position after the punctuation, or None.
    """
    for i in range(min(search_to, len(text) - 1), search_from, -1):
        if text[i] in ".!?\n":
            return i + 1
    return None


def quality_score(text: str) -> float:
    """
    Heuristic quality score in [0.0, 1.0] based on:
      - Ratio of alphabetic characters (weighted 60%)
      - Vocabulary diversity: unique words / total words (weighted 40%)

    A score near 1.0 indicates clean, diverse natural-language text.
    A score near 0.0 indicates gibberish, boilerplate, or code-heavy content.
    """
    if not text or len(text) < 10:
        return 0.0

    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0.0

    score = (alpha_ratio * 0.6) + (unique_ratio * 0.4)
    return round(min(max(score, 0.0), 1.0), 4)
