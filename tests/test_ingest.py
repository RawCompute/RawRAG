import numpy as np
import pytest
from fastapi.testclient import TestClient
import main

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "test.db"))  # throwaway DB
    main.store.matrix = np.empty((0, main.DIM), dtype=np.float32)    # reset global matrix
    monkeypatch.setattr(main, "chunk_pdf", lambda path: [
        {"text": "chunk one", "page": 1},
        {"text": "chunk two", "page": 2},
    ])
    monkeypatch.setattr(main, "embed_texts",
                        lambda texts: np.random.rand(len(texts), main.DIM))
    with TestClient(main.app) as c:  # `with` makes startup (lifespan) run
        yield c


def pdf(content=b"%PDF-1.4 fake"):
    return {"file": ("doc.pdf", content, "application/pdf")}


def test_happy_path(client):
    r = client.post("/ingest", files=pdf())
    assert r.status_code == 200
    assert r.json()["status"] == "ingested"
    assert r.json()["chunks"] == 2
    assert main.store.size == 2


def test_duplicate(client):
    client.post("/ingest", files=pdf())
    r = client.post("/ingest", files=pdf())
    assert r.json()["status"] == "already_ingested"
    assert main.store.size == 2  # no new rows added


def test_rejects_non_pdf(client):
    r = client.post("/ingest", files={"file": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_rejects_empty(client):
    assert client.post("/ingest", files=pdf(b"")).status_code == 400


def test_embedding_failure_stores_nothing(client, monkeypatch):
    def boom(texts):
        raise RuntimeError("rate limit")
    monkeypatch.setattr(main, "embed_texts", boom)
    assert client.post("/ingest", files=pdf()).status_code == 502
    assert main.store.size == 0


def test_no_text_extracted(client, monkeypatch):
    monkeypatch.setattr(main, "chunk_pdf", lambda path: [])
    assert client.post("/ingest", files=pdf()).status_code == 422


def test_survives_restart(client):
    client.post("/ingest", files=pdf())
    main.store.matrix = np.empty((0, main.DIM), dtype=np.float32)  # simulate memory wipe
    with TestClient(main.app):  # startup reloads from SQLite
        assert main.store.size == 2