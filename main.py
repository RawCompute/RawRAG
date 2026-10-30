import hashlib
import os
import sqlite3
import tempfile
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

DIM = 3072  #gemini embedding model, 3072 vectors ko handle



def chunk_pdf(path: str) -> list[dict]:
    """ADAPT: call your sliding-window chunker.
    Must return [{"text": str, "page": int}, ...]."""
    if os.getenv("RAWRAG_FAKE") == "1":
        return [{"text": "fake chunk one", "page": 1},
                {"text": "fake chunk two", "page": 2}]
    raise NotImplementedError("Wire chunk_pdf to core/chunker.py")


def embed_texts(texts: list[str]) -> np.ndarray:
    """ADAPT: call your Gemini client.
    Must return an array of shape (len(texts), 3072)."""
    if os.getenv("RAWRAG_FAKE") == "1":
        return np.random.rand(len(texts), DIM)
    raise NotImplementedError("Wire embed_texts to core/gemini_client.py")


class VectorStoreAdapter:
    """ADAPT: thin wrapper around your VectorDatabase class (or keep as is)."""

    def __init__(self):
        self.matrix = np.empty((0, DIM), dtype=np.float32)

    @property
    def size(self) -> int:
        return self.matrix.shape[0]

    def load(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def add(self, vectors: np.ndarray) -> None:
        self.matrix = np.vstack([self.matrix, vectors])

DB_PATH = "rawrag.db"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

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
    row_idx INTEGER NOT NULL UNIQUE,   -- position in the NumPy matrix
    page    INTEGER,
    text    TEXT NOT NULL,
    vector  BLOB NOT NULL              -- float32 bytes
);
"""

store = VectorStoreAdapter()
write_lock = threading.Lock()  # one writer at a time keeps SQLite and matrix aligned


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables, then rebuild the in-memory matrix from SQLite.
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

    # 1. Validate
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is too large.")

    # 2. Duplicate check (hash of the file contents)
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

    # 3. Extract and chunk
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        chunks = chunk_pdf(tmp_path)
    finally:
        os.remove(tmp_path)
    if not chunks:
        raise HTTPException(422, "No extractable text found (is this a scanned PDF?).")

    # 4. Embed everything BEFORE touching storage, so a failure leaves nothing half-written
    try:
        vectors = np.asarray(embed_texts([c["text"] for c in chunks]), dtype=np.float32)
    except Exception as exc:
        raise HTTPException(502, f"Embedding service failed: {exc}")

    # Normalise so a plain dot product equals cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.clip(norms, 1e-12, None)

    # 5. Store: SQLite first, then the matrix, all under one lock
    with write_lock:
        start = store.size
        conn = connect()
        try:
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

    # 6. Respond
    pages = {c.get("page") for c in chunks if c.get("page") is not None}
    return {"status": "ingested", "document_id": doc_id, "filename": file.filename,
            "pages": len(pages), "chunks": len(chunks)}