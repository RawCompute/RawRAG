#now to find the part from the pdf which will answer the users question to the best of our knowledge I am creating and  using the query part. joh chunks mein divide kiya hai closest chunks aayenge 
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


def fake_embed(texts):
    """Deterministic: 'one' -> axis 0, 'two' -> axis 1, 'three' -> axis 2."""
    out = np.zeros((len(texts), main.DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for axis, word in enumerate(["one", "two", "three"]):
            if word in t:
                out[i, axis] = 1.0
    return out


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "test.db"))
    main.store.matrix = np.empty((0, main.DIM), dtype=np.float32)
    monkeypatch.setattr(main, "chunk_pdf", lambda path: [
        {"text": "chunk one", "page": 1},
        {"text": "chunk two", "page": 2},
        {"text": "chunk three", "page": 3},
    ])
    monkeypatch.setattr(main, "embed_texts", fake_embed)
    monkeypatch.setattr(main, "generate_answer", lambda prompt: "ANSWER")
    with TestClient(main.app) as c:
        yield c


def ingest(client):
    return client.post("/ingest", files={"file": ("doc.pdf", b"%PDF-1.4 x", "application/pdf")})


def test_returns_best_chunk_first(client):
    ingest(client)
    r = client.post("/query", json={"question": "tell me about two", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ANSWER"
    assert body["sources"][0]["text"] == "chunk two"
    assert body["sources"][0]["page"] == 2
    assert body["sources"][0]["document"] == "doc.pdf"


def test_top_k_limits_results(client):
    ingest(client)
    r = client.post("/query", json={"question": "one", "top_k": 1})
    assert len(r.json()["sources"]) == 1


def test_top_k_larger_than_index(client):
    ingest(client)
    r = client.post("/query", json={"question": "one", "top_k": 20})
    assert r.status_code == 200
    assert len(r.json()["sources"]) == 3


def test_prompt_contains_context_and_question(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "generate_answer", lambda p: seen.update(p=p) or "ok")
    ingest(client)
    client.post("/query", json={"question": "what about three", "top_k": 1})
    assert "chunk three" in seen["p"]
    assert "what about three" in seen["p"]
    assert "chunk one" not in seen["p"]


def test_empty_index(client):
    r = client.post("/query", json={"question": "anything"})
    assert r.status_code == 409


def test_blank_question(client):
    ingest(client)
    assert client.post("/query", json={"question": "   "}).status_code == 400


def test_missing_question(client):
    assert client.post("/query", json={}).status_code == 422


def test_llm_failure(client, monkeypatch):
    def boom(prompt): raise RuntimeError("quota")
    monkeypatch.setattr(main, "generate_answer", boom)
    ingest(client)
    assert client.post("/query", json={"question": "one"}).status_code == 502


def test_min_score_filters_everything(client, monkeypatch):
    monkeypatch.setattr(main, "MIN_SCORE", 0.99)
    ingest(client)
    r = client.post("/query", json={"question": "unrelated words"})
    assert r.json()["answer"] == main.NOT_FOUND
    assert r.json()["sources"] == []


def test_query_after_restart(client):
    ingest(client)
    main.store.matrix = np.empty((0, main.DIM), dtype=np.float32)
    with TestClient(main.app) as c2:
        r = c2.post("/query", json={"question": "two", "top_k": 1})
        assert r.json()["sources"][0]["text"] == "chunk two"