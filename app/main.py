"""FastAPI giriş noktası ve HTTP API'si."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.nlp import analogy, nearest, sentence_similarity, vocabulary_size, word_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"

app = FastAPI(
    title="Anlam Atölyesi API",
    description="Gensim Word2Vec ile Türkçe anlamsal ilişkileri keşfedin.",
    version="1.0.0",
)


class PairRequest(BaseModel):
    first: str = Field(min_length=1, max_length=240)
    second: str = Field(min_length=1, max_length=240)


class AnalogyRequest(BaseModel):
    positive_a: str = Field(min_length=1, max_length=40)
    positive_b: str = Field(min_length=1, max_length=40)
    negative: str = Field(min_length=1, max_length=40)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": "Word2Vec", "vocabulary_size": vocabulary_size()}


@app.get("/api/nearest")
def find_nearest(word: str = Query(min_length=1, max_length=40), limit: int = Query(8, ge=1, le=12)):
    try:
        return {"query": word, "results": nearest(word, limit)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/similarity")
def compare_words(payload: PairRequest):
    try:
        return {"first": payload.first, "second": payload.second, "score": word_similarity(payload.first, payload.second)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/analogy")
def solve_analogy(payload: AnalogyRequest):
    try:
        return {"formula": f"{payload.positive_a} + {payload.positive_b} − {payload.negative}", "results": analogy(payload.positive_a, payload.positive_b, payload.negative)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/sentence-similarity")
def compare_sentences(payload: PairRequest):
    try:
        return sentence_similarity(payload.first, payload.second)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(PUBLIC_DIR / "index.html")


app.mount("/assets", StaticFiles(directory=PUBLIC_DIR / "assets"), name="assets")
