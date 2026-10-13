"""FastAPI giriş noktası ve HTTP API'si."""

from collections import deque
from pathlib import Path
from time import monotonic

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from app.nlp import analogy, nearest, sentence_similarity, vocabulary_size, word_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
PUBLIC_DIR = BASE_DIR / "public"
MAX_API_BODY_BYTES = 16 * 1024
RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_CLIENTS = 2048

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'; "
        "object-src 'none'; script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}

_request_windows: dict[str, deque[float]] = {}

app = FastAPI(
    title="Anlam Atölyesi API",
    description="Gensim Word2Vec ile Türkçe anlamsal ilişkileri keşfedin.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.vercel.app", "localhost", "127.0.0.1", "testserver"],
)


def _client_key(request: Request) -> str:
    """Vercel'in doğruladığı istemci adresini, yerelde bağlantı adresini kullanır."""
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


def _is_rate_limited(client: str, now: float) -> bool:
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    if client not in _request_windows and len(_request_windows) >= RATE_LIMIT_MAX_CLIENTS:
        oldest_client = min(
            _request_windows,
            key=lambda key: _request_windows[key][-1] if _request_windows[key] else 0,
        )
        _request_windows.pop(oldest_client, None)

    window = _request_windows.setdefault(client, deque())
    while window and window[0] <= cutoff:
        window.popleft()

    if len(window) >= RATE_LIMIT_REQUESTS:
        return True

    window.append(now)
    return False


@app.middleware("http")
async def secure_api(request: Request, call_next):
    """Küçük API'yi aşırı isteklerden korur ve tüm yanıtlara güvenlik başlıkları ekler."""
    if request.url.path.startswith("/api/"):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_API_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "İstek gövdesi çok büyük."},
                        headers=SECURITY_HEADERS,
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Geçersiz Content-Length başlığı."},
                    headers=SECURITY_HEADERS,
                )

        if _is_rate_limited(_client_key(request), monotonic()):
            return JSONResponse(
                status_code=429,
                content={"detail": "Çok fazla istek gönderildi. Lütfen biraz bekleyin."},
                headers={**SECURITY_HEADERS, "Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )

    response = await call_next(request)
    for name, value in SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


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


@app.get("/docs", include_in_schema=False)
def api_docs():
    return RedirectResponse(url="/docs.html", status_code=307)


app.mount(
    "/assets",
    StaticFiles(directory=PUBLIC_DIR / "assets", check_dir=False),
    name="assets",
)
