from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Anlam Atölyesi" in response.text


def test_health_reports_model():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "Word2Vec"
    assert body["vocabulary_size"] > 100


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

