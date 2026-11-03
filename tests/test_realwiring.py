# fake mode off ho toh yeh work karega to check real chunking, batching 
# this is optional only but covers one more aspect so good to keep.
import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

import main


def make_pdf(path, pages):
    c = canvas.Canvas(str(path))
    for text in pages:
        c.drawString(72, 700, text)
        c.showPage()
    c.save()


@pytest.fixture
def fake_gemini(monkeypatch):
    calls = {"embed_sizes": [], "answer_args": None}
    mod = types.ModuleType("core.gemini_client")

    def get_embeddings(chunks):
        calls["embed_sizes"].append(len(chunks))
        out = []
        for t in chunks:
            v = [0.0] * main.DIM
            v[0 if "alpha" in t else 1 if "beta" in t else 2] = 1.0
            out.append(v)
        return out

    def generate_answer(query, context_chunks):
        calls["answer_args"] = (query, context_chunks)
        return "MOCK ANSWER"

    mod.get_embeddings = get_embeddings
    mod.generate_answer = generate_answer
    monkeypatch.setitem(sys.modules, "core.gemini_client", mod)
    monkeypatch.delenv("RAWRAG_FAKE", raising=False)
    return calls


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", str(tmp_path / "test.db"))
    main.store.matrix = np.empty((0, main.DIM), dtype=np.float32)
    with TestClient(main.app) as c:
        yield c


def test_real_chunker_keeps_page_numbers(tmp_path):
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, ["alpha page one", "beta page two", "gamma page three"])
    chunks = main.chunk_pdf(str(pdf))
    assert [c["page"] for c in chunks] == [1, 2, 3]
    assert "beta" in chunks[1]["text"]


def test_ocr_ingests_and_repairs_unreadable_duplicate(client, fake_gemini):
    pdf_path = Path(__file__).resolve().parent.parent / "sample.pdf"
    pdf_data = pdf_path.read_bytes()
    digest = hashlib.sha256(pdf_data).hexdigest()

    conn = main.connect()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO documents (filename, sha256) VALUES (?, ?)",
                (pdf_path.name, digest),
            )
            conn.execute(
                "INSERT INTO chunks (doc_id, row_idx, page, text, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    cur.lastrowid,
                    0,
                    1,
                    " ".join(["/i255/1 /2 /3 /4"] * 20),
                    np.zeros(main.DIM, dtype=np.float32).tobytes(),
                ),
            )
    finally:
        conn.close()
    main.store.load(np.zeros((1, main.DIM), dtype=np.float32))

    response = client.post(
        "/ingest",
        files={"file": (pdf_path.name, pdf_data, "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "reingested"
    assert response.json()["chunks"] > 0

    conn = main.connect()
    try:
        rows = conn.execute(
            "SELECT row_idx, text FROM chunks ORDER BY row_idx"
        ).fetchall()
    finally:
        conn.close()
    text = " ".join(row[1] for row in rows)
    assert [row[0] for row in rows] == list(range(len(rows)))
    assert main._has_readable_text(text)
    assert "rule value approach" in text.lower()
    assert main.store.size == len(rows)


def test_embed_texts_batches_requests(fake_gemini):
    vectors = main.embed_texts([f"alpha {i}" for i in range(120)])
    assert vectors.shape == (120, main.DIM)
    assert fake_gemini["embed_sizes"] == [50, 50, 20]


def test_full_flow_without_fake_mode(tmp_path, client, fake_gemini):
    pdf = tmp_path / "book.pdf"
    make_pdf(pdf, ["alpha is the first topic", "beta is the second topic"])
    with open(pdf, "rb") as f:
        r = client.post("/ingest", files={"file": ("book.pdf", f.read(), "application/pdf")})
    assert r.status_code == 200 and r.json()["chunks"] == 2

    r = client.post("/query", json={"question": "tell me about beta", "top_k": 1})
    body = r.json()
    assert body["answer"] == "MOCK ANSWER"
    assert body["sources"][0]["page"] == 2
    assert body["sources"][0]["document"] == "book.pdf"
    question, chunks = fake_gemini["answer_args"]
    assert question == "tell me about beta"
    assert len(chunks) == 1 and "beta" in chunks[0]


def test_missing_api_key_gives_clean_502(tmp_path, client, monkeypatch):
    monkeypatch.delenv("RAWRAG_FAKE", raising=False)

    def no_key(*a, **k):
        raise ValueError("Missing key inputs argument")

    mod = types.ModuleType("core.gemini_client")
    mod.__getattr__ = lambda name: no_key
    monkeypatch.setitem(sys.modules, "core.gemini_client", mod)
    pdf = tmp_path / "x.pdf"
    make_pdf(pdf, ["alpha"])
    with open(pdf, "rb") as f:
        r = client.post("/ingest", files={"file": ("x.pdf", f.read(), "application/pdf")})
    assert r.status_code == 502