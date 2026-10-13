"""Önceden Gensim ile eğitilmiş Türkçe sözcük vektörleri için çıkarım katmanı."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
VECTORS_PATH = BASE_DIR / "data" / "word_vectors.json"
TOKEN_RE = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Metni Türkçe karakterleri koruyarak küçük harfli sözcüklere ayırır."""
    return TOKEN_RE.findall(text.lower().replace("İ", "i"))


@lru_cache(maxsize=1)
def get_vectors() -> dict[str, list[float]]:
    """Gensim eğitim betiğinin ürettiği vektörleri süreç başına bir kez yükler."""
    payload = json.loads(VECTORS_PATH.read_text(encoding="utf-8"))
    return payload["vectors"]


def vocabulary_size() -> int:
    return len(get_vectors())


def cosine(first: list[float], second: list[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second))
    norm_a = math.sqrt(sum(value * value for value in first))
    norm_b = math.sqrt(sum(value * value for value in second))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def nearest(word: str, limit: int = 8) -> list[dict[str, float | str]]:
    vectors = get_vectors()
    token = normalize_word(word)
    ensure_known(vectors, token)
    return ranked(vectors[token], vectors, excluded={token}, limit=limit)


def word_similarity(first: str, second: str) -> float:
    vectors = get_vectors()
    a, b = normalize_word(first), normalize_word(second)
    ensure_known(vectors, a, b)
    return round(cosine(vectors[a], vectors[b]), 4)


def analogy(positive_a: str, positive_b: str, negative: str, limit: int = 6):
    vectors = get_vectors()
    a, b, n = map(normalize_word, (positive_a, positive_b, negative))
    ensure_known(vectors, a, b, n)
    target = [x + y - z for x, y, z in zip(vectors[a], vectors[b], vectors[n])]
    return ranked(target, vectors, excluded={a, b, n}, limit=limit)


def sentence_similarity(first: str, second: str) -> dict[str, object]:
    vectors = get_vectors()
    tokens_a = [token for token in tokenize(first) if token in vectors]
    tokens_b = [token for token in tokenize(second) if token in vectors]
    if not tokens_a or not tokens_b:
        raise ValueError("Her iki cümlede de sözlükte bulunan en az bir sözcük olmalı.")
    vector_a = average([vectors[token] for token in tokens_a])
    vector_b = average([vectors[token] for token in tokens_b])
    return {
        "score": round(cosine(vector_a, vector_b), 4),
        "tokens": {"first": tokens_a, "second": tokens_b},
    }


def average(items: list[list[float]]) -> list[float]:
    return [sum(values) / len(items) for values in zip(*items)]


def ranked(target: list[float], vectors: dict[str, list[float]], excluded: set[str], limit: int):
    scores = (
        (word, cosine(target, vector))
        for word, vector in vectors.items()
        if word not in excluded
    )
    best = sorted(scores, key=lambda item: item[1], reverse=True)[:limit]
    return [{"word": word, "score": round(score, 4)} for word, score in best]


def normalize_word(value: str) -> str:
    tokens = tokenize(value)
    if len(tokens) != 1:
        raise ValueError("Lütfen tek bir sözcük girin.")
    return tokens[0]


def ensure_known(vectors: dict[str, list[float]], *words: str) -> None:
    missing = [word for word in words if word not in vectors]
    if missing:
        suggestions = sorted(
            (known for known in vectors if known.startswith(missing[0][:2])),
            key=len,
        )[:5]
        hint = f" Öneriler: {', '.join(suggestions)}." if suggestions else ""
        raise ValueError(f"Sözlükte bulunamadı: {', '.join(missing)}.{hint}")
