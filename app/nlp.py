"""Gensim tabanlı, küçük ve açıklanabilir Türkçe Word2Vec servisi."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_PATH = BASE_DIR / "data" / "turkish_corpus.txt"
TOKEN_RE = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Metni Türkçe karakterleri koruyarak küçük harfli sözcüklere ayırır."""
    return TOKEN_RE.findall(text.lower().replace("İ", "i"))


def load_corpus() -> list[list[str]]:
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    return [tokens for line in lines if (tokens := tokenize(line))]


@lru_cache(maxsize=1)
def get_model() -> Word2Vec:
    """Modeli süreç başına bir kez, tekrarlanabilir ayarlarla eğitir."""
    sentences = load_corpus()
    return Word2Vec(
        sentences=sentences,
        vector_size=80,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        negative=10,
        epochs=220,
        seed=42,
    )


def vocabulary_size() -> int:
    return len(get_model().wv)


def nearest(word: str, limit: int = 8) -> list[dict[str, float | str]]:
    model = get_model().wv
    token = normalize_word(word)
    ensure_known(model, token)
    return [
        {"word": candidate, "score": round(float(score), 4)}
        for candidate, score in model.most_similar(token, topn=limit)
    ]


def word_similarity(first: str, second: str) -> float:
    model = get_model().wv
    a, b = normalize_word(first), normalize_word(second)
    ensure_known(model, a, b)
    return round(float(model.similarity(a, b)), 4)


def analogy(positive_a: str, positive_b: str, negative: str, limit: int = 6):
    model = get_model().wv
    a, b, n = map(normalize_word, (positive_a, positive_b, negative))
    ensure_known(model, a, b, n)
    return [
        {"word": candidate, "score": round(float(score), 4)}
        for candidate, score in model.most_similar(positive=[a, b], negative=[n], topn=limit)
    ]


def sentence_similarity(first: str, second: str) -> dict[str, object]:
    model = get_model().wv
    tokens_a = [token for token in tokenize(first) if token in model]
    tokens_b = [token for token in tokenize(second) if token in model]
    if not tokens_a or not tokens_b:
        raise ValueError("Her iki cümlede de sözlükte bulunan en az bir sözcük olmalı.")
    vector_a = np.mean([model[token] for token in tokens_a], axis=0)
    vector_b = np.mean([model[token] for token in tokens_b], axis=0)
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    score = float(np.dot(vector_a, vector_b) / denominator) if denominator else 0.0
    return {
        "score": round(score, 4),
        "tokens": {"first": tokens_a, "second": tokens_b},
    }


def normalize_word(value: str) -> str:
    tokens = tokenize(value)
    if len(tokens) != 1:
        raise ValueError("Lütfen tek bir sözcük girin.")
    return tokens[0]


def ensure_known(model, *words: str) -> None:
    missing = [word for word in words if word not in model]
    if missing:
        suggestions = sorted(
            (known for known in model.index_to_key if known.startswith(missing[0][:2])),
            key=len,
        )[:5]
        hint = f" Öneriler: {', '.join(suggestions)}." if suggestions else ""
        raise ValueError(f"Sözlükte bulunamadı: {', '.join(missing)}.{hint}")

