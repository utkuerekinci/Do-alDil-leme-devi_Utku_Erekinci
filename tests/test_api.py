from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Anlam Atölyesi" in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_health_reports_model():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "Word2Vec"
    assert body["vocabulary_size"] > 100


def test_static_api_docs_are_available_without_schema_exposure():
    response = client.get("/docs", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs.html"
    assert client.get("/openapi.json").status_code == 404


def test_untrusted_host_is_rejected():
    response = client.get("/api/health", headers={"host": "attacker.example"})
    assert response.status_code == 400


def test_nearest_words():
    response = client.get("/api/nearest", params={"word": "bilim", "limit": 5})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 5


def test_similarity_is_bounded():
    response = client.post("/api/similarity", json={"first": "kitap", "second": "bilgi"})
    assert response.status_code == 200
    assert -1 <= response.json()["score"] <= 1


def test_unknown_word_returns_helpful_error():
    response = client.get("/api/nearest", params={"word": "xyzabc"})
    assert response.status_code == 422
    assert "Sözlükte bulunamadı" in response.json()["detail"]


def test_sentence_similarity_returns_tokens():
    response = client.post(
        "/api/sentence-similarity",
        json={"first": "öğrenci kitap okur", "second": "okul bilgi eğitim"},
    )
    assert response.status_code == 200
    assert response.json()["tokens"]["first"] == ["öğrenci", "kitap"]


def test_oversized_api_body_is_rejected():
    response = client.post(
        "/api/similarity",
        content=b"x" * (16 * 1024 + 1),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.headers["x-content-type-options"] == "nosniff"


def test_trace_method_is_rejected():
    response = client.request("TRACE", "/api/health")
    assert response.status_code == 405


def test_api_rate_limit_is_enforced():
    headers = {"x-real-ip": "203.0.113.42"}
    for _ in range(120):
        assert client.get("/api/health", headers=headers).status_code == 200
    response = client.get("/api/health", headers=headers)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
