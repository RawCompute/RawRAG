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


def test_rejects_garbled_font_encoding(client, monkeypatch):
    monkeypatch.setattr(main, "chunk_pdf", lambda path: [
        {"text": " ".join(["/i255/1 /2 /3 /4"] * 20), "page": 1},
    ])
    monkeypatch.setattr(
        main, "embed_texts",
        lambda texts: pytest.fail("Garbled PDF text must not be embedded."),
    )

    response = client.post("/ingest", files=pdf())
    assert response.status_code == 422
    assert "OCR could not recover" in response.json()["detail"]


def test_reingests_unreadable_duplicate_and_rebuilds_index(client):
    first = client.post("/ingest", files=pdf())
    second = client.post(
        "/ingest",
        files={"file": ("other.pdf", b"%PDF-1.4 other", "application/pdf")},
    )
    assert first.status_code == second.status_code == 200
    assert main.store.size == 4

    conn = main.connect()
    try:
        with conn:
            conn.execute(
                "UPDATE chunks SET text = ? WHERE doc_id = ?",
                (" ".join(["/i255/1 /2 /3 /4"] * 20), first.json()["document_id"]),
            )
    finally:
        conn.close()

    response = client.post("/ingest", files=pdf())
    assert response.status_code == 200
    assert response.json()["status"] == "reingested"
    assert main.store.size == 4

    conn = main.connect()
    try:
        rows = conn.execute(
            "SELECT row_idx, text FROM chunks ORDER BY row_idx"
        ).fetchall()
    finally:
        conn.close()
    assert [row[0] for row in rows] == list(range(4))
    assert all(main._has_readable_text(row[1]) for row in rows)


def test_survives_restart(client):
    client.post("/ingest", files=pdf())
    main.store.matrix = np.empty((0, main.DIM), dtype=np.float32)  # simulate memory wipe
    with TestClient(main.app):  # startup reloads from SQLite
        assert main.store.size == 2

def test_unreadable_pdf_returns_422(client, monkeypatch):
    def broken(path):
        raise ValueError("EOF marker not found")
    monkeypatch.setattr(main, "chunk_pdf", broken)
    r = client.post("/ingest", files=pdf())
    assert r.status_code == 422
    assert main.store.size == 0