"""Türkçe derlemden Gensim Word2Vec modelini eğitip dağıtım vektörlerini üretir."""

from __future__ import annotations

import json
import re
from pathlib import Path

from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "turkish_corpus.txt"
OUTPUT_PATH = ROOT / "data" / "word_vectors.json"
TOKEN_RE = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower().replace("İ", "i"))


def main() -> None:
    sentences = [
        tokens
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if (tokens := tokenize(line))
    ]
    model = Word2Vec(
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
    payload = {
        "metadata": {
            "library": "gensim",
            "algorithm": "Word2Vec Skip-gram",
            "vector_size": model.wv.vector_size,
            "vocabulary_size": len(model.wv),
            "seed": 42,
        },
        "vectors": {
            word: [round(float(value), 8) for value in model.wv[word]]
            for word in model.wv.index_to_key
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"{len(model.wv)} sözcük, {model.wv.vector_size} boyut -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

