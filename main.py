import hashlib
import os
import sqlite3
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

load_dotenv()

DIM = 3072  # every Gemini embedding has 3072 numbers
EMBED_BATCH = 50  # chunks sent to Gemini per request


def is_fake() -> bool:
    return os.getenv("RAWRAG_FAKE") == "1"


# The three functions below connect main.py to core/. With RAWRAG_FAKE=1 they
# return fake data, so you can test without a Gemini key.
# The core imports are inside the functions so that fake mode (and the tests)
# work even when GEMINI_API_KEY is not set.
def chunk_pdf(path: str) -> list[dict]:
    if is_fake():
        return [{"text": "fake chunk one", "page": 1},
                {"text": "fake chunk two", "page": 2}]

    import PyPDF2
    from core.chunker import chunk_text

    chunks = []
    reader = PyPDF2.PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").replace("\n", " ")
        for piece in chunk_text(page_text):
            chunks.append({"text": piece, "page": page_number})
    return chunks


def embed_texts(texts: list[str]) -> np.ndarray:
    if is_fake():
        return np.random.rand(len(texts), DIM)

    from core.gemini_client import get_embeddings

    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors.extend(get_embeddings(texts[i:i + EMBED_BATCH]))
    return np.array(vectors, dtype=np.float32)


def generate_answer(query: str, context_chunks: list[str]) -> str:
    if is_fake():
        return "FAKE ANSWER (no LLM was called)."

    from core.gemini_client import generate_answer as gemini_generate_answer

    return gemini_generate_answer(query, context_chunks)


class VectorStoreAdapter:
    def __init__(self):
        self.matrix = np.empty((0, DIM), dtype=np.float32)

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    def load(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def add(self, vectors: np.ndarray) -> None:
        self.matrix = np.vstack([self.matrix, vectors])


DB_PATH = str(Path(__file__).resolve().parent / "rawrag.db")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

MIN_SCORE = 0.0
NOT_FOUND = "I couldn't find that in the provided documents."

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    filename   TEXT NOT NULL,
    sha256     TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id  INTEGER NOT NULL REFERENCES documents(id),
    row_idx INTEGER NOT NULL UNIQUE,
    page    INTEGER,
    text    TEXT NOT NULL,
    vector  BLOB NOT NULL
);
"""

store = VectorStoreAdapter()
write_lock = threading.Lock()
DEFAULT_HOST = os.getenv("HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        rows = conn.execute("SELECT vector FROM chunks ORDER BY row_idx").fetchall()
    finally:
        conn.close()
    if rows:
        store.load(np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in rows]))
    else:
        store.load(np.empty((0, DIM), dtype=np.float32))
    yield


app = FastAPI(title="RawRAG", lifespan=lifespan)


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large.")

    digest = hashlib.sha256(data).hexdigest()
    conn = connect()
    try:
        existing = conn.execute(
            "SELECT id FROM documents WHERE sha256 = ?", (digest,)
        ).fetchone()
    finally:
        conn.close()
    if existing:
        return {"status": "already_ingested", "document_id": existing[0],
                "filename": file.filename}

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        chunks = chunk_pdf(tmp_path)
    except Exception as exc:
        raise HTTPException(422, f"Could not read this PDF: {exc}")
    finally:
        os.remove(tmp_path)
    if not chunks:
        raise HTTPException(422, "No extractable text found (is this a scanned PDF?).")

    try:
        vectors = np.asarray(embed_texts([c["text"] for c in chunks]), dtype=np.float32)
    except Exception as exc:
        raise HTTPException(502, f"Embedding service failed: {exc}")
    if vectors.shape != (len(chunks), DIM):
        raise HTTPException(
            502, f"Embedding service returned shape {vectors.shape}, "
                 f"expected {(len(chunks), DIM)}.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-12, None)

    with write_lock:
        conn = connect()
        try:
            existing = conn.execute(
                "SELECT id FROM documents WHERE sha256 = ?", (digest,)
            ).fetchone()
            if existing:
                return {"status": "already_ingested", "document_id": existing[0],
                        "filename": file.filename}

            start = store.size
            with conn:
                cur = conn.execute(
                    "INSERT INTO documents (filename, sha256) VALUES (?, ?)",
                    (file.filename, digest),
                )
                doc_id = cur.lastrowid
                conn.executemany(
                    "INSERT INTO chunks (doc_id, row_idx, page, text, vector) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (doc_id, start + i, c.get("page"), c["text"], vectors[i].tobytes())
                        for i, c in enumerate(chunks)
                    ],
                )
        finally:
            conn.close()
        store.add(vectors)

    pages = {c.get("page") for c in chunks if c.get("page") is not None}
    return {"status": "ingested", "document_id": doc_id, "filename": file.filename,
            "pages": len(pages), "chunks": len(chunks)}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


@app.post("/query")
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be blank.")

    matrix = store.matrix
    if matrix.shape[0] == 0:
        raise HTTPException(409, "No documents have been ingested yet.")

    try:
        q = np.asarray(embed_texts([question]), dtype=np.float32)[0]
    except Exception as exc:
        raise HTTPException(502, f"Embedding service failed: {exc}")
    if q.shape != (DIM,):
        raise HTTPException(502, f"Embedding service returned shape {q.shape}, expected {(DIM,)}.")
    q = q / max(float(np.linalg.norm(q)), 1e-12)

    scores = matrix @ q
    k = min(req.top_k, matrix.shape[0])
    top = [int(i) for i in np.argsort(scores)[::-1][:k] if scores[i] >= MIN_SCORE]
    if not top:
        return {"answer": NOT_FOUND, "sources": []}

    placeholders = ",".join("?" * len(top))
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT c.row_idx, c.page, c.text, d.filename FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.row_idx IN ({placeholders})", top,
        ).fetchall()
    finally:
        conn.close()
    by_row = {r[0]: r for r in rows}
    ordered = [by_row[i] for i in top if i in by_row]

    try:
        answer = generate_answer(question, [text for _, _, text, _ in ordered])
    except Exception as exc:
        raise HTTPException(502, f"Language model failed: {exc}")

    return {
        "answer": answer,
        "sources": [
            {"document": filename, "page": page,
             "score": round(float(scores[row_idx]), 4), "text": text[:300]}
            for row_idx, page, text, filename in ordered
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=DEFAULT_HOST, port=DEFAULT_PORT)